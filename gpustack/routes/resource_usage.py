"""Read API for the unified resource metering framework (``metered_usage``).

Endpoints (mounted under ``/usage``):

* ``POST /usage/resource/breakdown``       — by resource_type
* ``POST /usage/gpu-instances/breakdown``  — by instance_type(sku) / instance / user
* ``POST /usage/storage/breakdown``        — by type(sku) / volume / user
* ``GET  /usage/summary``                  — cross-resource KPIs (tokens unioned in)
* ``GET  /usage/resource-events``          — resource-events lifecycle log

The token tabs keep using ``gpustack/routes/usage.py`` (``model_usages``)
unchanged; this module only serves the time-based resources.
"""

import json
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, case, desc, func, literal_column, or_
from sqlmodel import select

from gpustack import envs
from gpustack.api.exceptions import InvalidException
from gpustack.routes.usage import _resolve_effective_scope
from gpustack.schemas.common import Pagination
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    RESOURCE_TYPE_CPU_INSTANCE,
    RESOURCE_TYPE_GPU_INSTANCE,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    MeteredUsage,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
)
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.principals import Principal
from gpustack.schemas.usage import (
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_MONTH,
    USAGE_GRANULARITY_WEEK,
    USAGE_SCOPE_ALL,
    USAGE_SCOPE_SELF,
)
from gpustack.schemas.resource_events import ResourceEvent
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep
from gpustack.utils.rollup_tz import (
    resolve_rollup_tz,
    rollup_fixed_tz,
    rollup_offset_minutes,
    to_rollup_aware,
)

router = APIRouter()

# metered_usage is hourly; token usage (model_usages) has no hour granularity.
USAGE_GRANULARITY_HOUR = "hour"


# ---------------------------------------------------------------------------
# Metric definitions — each pins a meter + a quantity→display conversion.
# Expressed as SQL so SUM happens in the DB.
# ---------------------------------------------------------------------------

_UPTIME = MeteredUsage.meter_key == METER_INSTANCE_UPTIME
_STORAGE = MeteredUsage.meter_key == METER_STORAGE_CAPACITY


def _sum_case(condition, value) -> Any:
    return func.coalesce(func.sum(case((condition, value), else_=0)), 0)


def _metric_columns() -> Dict[str, Any]:
    q = MeteredUsage.quantity
    return {
        # instance.uptime seconds → hours
        "instance_hours": _sum_case(_UPTIME, q) / 3600.0,
        # uptime seconds × card count → GPU-hours. Filtered to GPU instances:
        # CPU rows carry sku_count=1 (whole machine), so without the resource
        # filter their instance-seconds would leak into GPU-Hours.
        "gpu_hours": _sum_case(
            and_(_UPTIME, MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE),
            q * MeteredUsage.sku_count,
        )
        / 3600.0,
        # storage.capacity mib-seconds → GB·days
        "gb_days": _sum_case(_STORAGE, q) / 1024.0 / 86400.0,
        "gb_hours": _sum_case(_STORAGE, q) / 1024.0 / 3600.0,
    }


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ResourceBreakdownRequest(BaseModel):
    scope: str = USAGE_SCOPE_ALL
    start_date: date
    end_date: date
    # One or more grouping dimensions, combined left-to-right (mirrors the token
    # usage API's ``group_by: ["date", "user"]``). A trend splits a bucketed
    # series via ["date", "<dim>"]; a table uses a single ["<dim>"].
    group_by: List[str] = Field(default_factory=lambda: ["resource_type"])
    granularity: str = USAGE_GRANULARITY_DAY
    order_by: Optional[str] = None  # a metric key; default = first metric
    descending: bool = True
    # Optional "filter by user" — restricts to these creator (principal) ids.
    # Safe in any scope: in SELF scope it only ever intersects with the
    # caller's own rows, so it can't widen visibility.
    creator_ids: Optional[List[int]] = None
    # Optional "filter by resource" — instance ids (GPU tab) / volume ids
    # (Storage tab). Both narrow ``resource_id``; an endpoint only ever
    # receives the kind that matches its ``base_filter``.
    instance_ids: Optional[List[int]] = None
    volume_ids: Optional[List[int]] = None
    # ``-1`` is the no-pagination sentinel (return all buckets) used by the
    # trend chart and the export path; any other value must be positive.
    page: int = 1
    # Generous ceiling (abuse cap only). Kept at 10000 for backward compat:
    # older UIs fetch the whole filtered set via perPage=10000, so lowering it
    # would 400 their trend/export requests during a rolling upgrade. New
    # callers fetch the full series via page=-1 instead.
    perPage: int = Field(default=20, ge=1, le=10000)

    @field_validator("page")
    @classmethod
    def validate_page(cls, value: int) -> int:
        if value == 0:
            raise ValueError("page must be a positive number or -1 (no pagination)")
        return value


def _parse_id_csv(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated id list from a GET query param.

    GET endpoints take ``creator_ids`` as a CSV string (e.g. ``"3,7"``) rather
    than repeated params so the frontend doesn't have to fight axios array
    serialization. Returns ``None`` for empty/blank so callers can skip the
    filter entirely."""
    if not value:
        return None
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out or None


# group_by token → (label column expr factory). resource/user/sku/date.
def _group_columns(
    group_by: str, session, granularity: str = USAGE_GRANULARITY_DAY
) -> Tuple[List, List]:
    if group_by == "resource_type":
        return [MeteredUsage.resource_type.label("group_key")], [
            MeteredUsage.resource_type
        ]
    if group_by == "instance_type":
        # Group by the actual resource *shape* = the flavor ``sku`` plus
        # ``sku_count`` (GPU card count / CPU base-flavor unit count). A generic
        # CPU flavor hosts 1c2g / 2c4g / 3c6g and a GPU flavor hosts different
        # card counts, all under one sku; sku_count splits them. Both are indexed
        # columns — ``dimensions`` is a display blob and grouping on its JSON
        # would force a full scan, so the shape's specs are read from a
        # representative row per (sku, sku_count) instead (see _attach_dimensions).
        return (
            [
                MeteredUsage.sku.label("group_key"),
                MeteredUsage.sku_count.label("group_sku_count"),
            ],
            [MeteredUsage.sku, MeteredUsage.sku_count],
        )
    if group_by in ("type", "sku"):
        return [MeteredUsage.sku.label("group_key")], [MeteredUsage.sku]
    if group_by in ("instance", "volume", "resource"):
        return (
            [
                MeteredUsage.resource_id.label("group_id"),
                MeteredUsage.resource_name.label("group_key"),
            ],
            [MeteredUsage.resource_id, MeteredUsage.resource_name],
        )
    if group_by == "user":
        return (
            [
                MeteredUsage.creator_id.label("group_id"),
                MeteredUsage.creator_name.label("group_key"),
            ],
            [MeteredUsage.creator_id, MeteredUsage.creator_name],
        )
    if group_by == "date":
        expr = _bucket_expr(session, granularity)
        return [expr.label("group_date")], [expr]
    raise InvalidException(message=f"Unsupported group_by: {group_by}")


def _rollup_shift(col, dialect: str, offset_min: int):
    """Add the rollup-tz UTC offset (minutes) to a UTC timestamp column so SQL
    date bucketing groups by the rollup-tz calendar. A plain interval add →
    portable across PostgreSQL / MySQL with no tz tables (fixed offset, so
    DST-naive)."""
    if offset_min == 0:
        return col
    if dialect == "postgresql":
        return col + literal_column(f"interval '{offset_min} minutes'")
    if dialect == "mysql":
        return func.date_add(col, literal_column(f"interval {offset_min} minute"))
    # Fallback for the in-memory sqlite test engine / any other backend.
    sign = "+" if offset_min >= 0 else "-"
    return func.datetime(col, f"{sign}{abs(offset_min)} minutes")


def _bucket_expr(session, granularity: str):
    """Time-bucket expression over ``bucket_start`` for the requested
    granularity. Rows are stored at hourly UTC granularity; we shift to the
    rollup timezone (so buckets follow the operator's calendar, consistent with
    the token rollup) then truncate via date_trunc (PostgreSQL) / date functions
    (MySQL).

    For a half-hour-offset rollup tz (e.g. ``+05:30``) the hour labels land on
    the UTC ``:30`` boundary, so one stored UTC hour straddles two displayed
    hours. Harmless for aggregation (totals are unchanged) — only the hour-axis
    labels look offset by 30 minutes."""
    dialect = session.get_bind().dialect.name
    col = _rollup_shift(MeteredUsage.bucket_start, dialect, rollup_offset_minutes())
    if granularity == USAGE_GRANULARITY_HOUR:
        if dialect == "postgresql":
            return func.date_trunc("hour", col)
        if dialect == "mysql":
            return func.date_format(col, "%Y-%m-%d %H:00:00")
        return func.strftime("%Y-%m-%d %H:00:00", col)  # sqlite test engine
    if dialect == "postgresql":
        unit = {
            USAGE_GRANULARITY_WEEK: "week",
            USAGE_GRANULARITY_MONTH: "month",
        }.get(granularity, "day")
        return func.date_trunc(unit, col)
    # MySQL (and the sqlite test engine)
    if granularity == USAGE_GRANULARITY_MONTH:
        return func.date_format(col, "%Y-%m-01")
    if granularity == USAGE_GRANULARITY_WEEK:
        # Monday-start week, matching PostgreSQL's date_trunc('week').
        return func.subdate(func.date(col), func.weekday(col))
    return func.date(col)  # day


def _localize_bucket(value: Any, granularity: str, fixed_tz=None) -> Any:
    """Label an hour bucket with the rollup offset so it serializes
    self-describing (e.g. ``2026-06-10T14:00:00+08:00``).

    ``_bucket_expr`` already shifted ``bucket_start`` into the rollup wall clock;
    this just attaches the matching fixed offset. The SQL value is a naive
    datetime (PostgreSQL ``date_trunc``) or an ``"YYYY-MM-DD HH:00:00"`` string
    (MySQL ``date_format``). Day/week/month buckets are calendar dates with no
    time-of-day, so they carry no offset and pass through unchanged.

    Pass ``fixed_tz`` to reuse one resolved per request (avoids re-reading the
    env for every row); defaults to resolving it."""
    if granularity != USAGE_GRANULARITY_HOUR or value is None:
        return value
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=fixed_tz or rollup_fixed_tz())
    return value


def _rollup_day_window(
    start_date: Optional[date], end_date: Optional[date]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Map an inclusive rollup-tz day range to a half-open UTC ``[start, end)``
    datetime window. Both ends are shifted back by the rollup offset, so a stored
    UTC timestamp compares against the rollup calendar day the UI selected — the
    same shift ``_bucket_expr`` applies to the displayed buckets. Half-open +
    naive datetimes keep the bucket_start / occurred_at index usable (no
    per-row ``cast(..., Date)``). A ``None`` bound yields ``None`` (skip it)."""
    offset = timedelta(minutes=rollup_offset_minutes())
    start_dt = (
        datetime(start_date.year, start_date.month, start_date.day) - offset
        if start_date is not None
        else None
    )
    end_dt = (
        datetime(end_date.year, end_date.month, end_date.day)
        + timedelta(days=1)
        - offset
        if end_date is not None
        else None
    )
    return start_dt, end_dt


def _bucket_in_range(statement, start_date: date, end_date: date):
    """Filter ``MeteredUsage.bucket_start`` to the inclusive rollup-tz day range
    [start_date, end_date] (see :func:`_rollup_day_window`)."""
    start_dt, end_dt = _rollup_day_window(start_date, end_date)
    return statement.where(MeteredUsage.bucket_start >= start_dt).where(
        MeteredUsage.bucket_start < end_dt
    )


def _apply_scope(
    statement,
    *,
    user,
    ctx,
    scope: str,
    base_filter=None,
    creator_ids=None,
    resource_ids=None,
):
    if base_filter is not None:
        statement = statement.where(base_filter)
    if scope == USAGE_SCOPE_SELF:
        statement = statement.where(MeteredUsage.creator_id == user.id)
    if ctx is not None and ctx.current_principal_id is not None:
        # Tenant scope follows the consumer (the tenant that ran the resource);
        # owner is the cluster provider. "My org's usage" = consumer_principal_id
        # — same as the token side (model_usages.consumer_principal_id).
        statement = statement.where(
            MeteredUsage.consumer_principal_id == ctx.current_principal_id
        )
    if creator_ids:
        statement = statement.where(MeteredUsage.creator_id.in_(creator_ids))
    if resource_ids:
        statement = statement.where(MeteredUsage.resource_id.in_(resource_ids))
    return statement


async def _enrich_items(session, gb: str, items: List[dict]) -> None:
    """Post-query enrichment of breakdown rows (mutates ``items`` in place).

    * entity groupings (user / instance / volume): resolve display names
      (``metered_usage`` doesn't snapshot them) and flag rows whose entity was
      since deleted — the same "(Deleted)" treatment the token breakdown gives.
    * instance-type / per-instance groupings: attach the flavor's display fields
      (pretty product name + per-card cpu/mem/vram) so the UI renders them like
      the GPU Instances list instead of the raw flavor slug. Dimensions are
      flavor-constant per sku, so one representative row per sku is enough.
    """
    if not items:
        return
    if gb in ("user", "instance", "volume"):
        ids = [i["id"] for i in items if i.get("id") is not None]
        existing: set[int] = set()
        names: dict[int, str] = {}
        if ids:
            if gb == "user":
                principals = (
                    await session.exec(select(Principal).where(Principal.id.in_(ids)))
                ).all()
                names = {p.id: (p.display_name or p.name) for p in principals}
                existing = set(names)
            else:
                model = GPUInstance if gb == "instance" else GPUInstancePersistentVolume
                existing = set(
                    (
                        await session.exec(select(model.id).where(model.id.in_(ids)))
                    ).all()
                )
        for i in items:
            rid = i.get("id")
            if rid is None:
                continue
            if gb == "user" and names.get(rid):
                i["key"] = names[rid]
            i["deleted"] = rid not in existing

    await _attach_dimensions(session, gb, items)


async def _dims_by_representative(session, *, group_col, keys, extra_filter):
    """``{group value: dimensions}`` from the latest row per group.

    Picks MAX(id) per ``group_col`` among rows matching ``extra_filter`` (e.g.
    uptime / storage-capacity meter) and returns that representative row's
    ``dimensions`` blob — dimensions are constant per group so any row will do.
    """
    if not keys:
        return {}
    rep_ids = (
        select(func.max(MeteredUsage.id))
        .where(extra_filter)
        .where(group_col.in_(keys))
        .group_by(group_col)
    )
    rows = (
        await session.exec(
            select(group_col, MeteredUsage.dimensions).where(
                MeteredUsage.id.in_(rep_ids)
            )
        )
    ).all()
    return {r[0]: (r[1] or {}) for r in rows}


async def _dims_by_shape(session, shapes):
    """``{(sku, sku_count): dimensions}`` from the latest row per shape.

    Instance types are grouped by ``(sku, sku_count)``; every row of one shape
    has the same specs (cpu/mem/cards) and flavor fields, so one MAX(id) per
    ``(sku, sku_count)`` supplies the whole display blob. Grouping is on indexed
    columns, not JSON.
    """
    skus = {s for s, _ in shapes if s}
    if not skus:
        return {}
    rep_ids = (
        select(func.max(MeteredUsage.id))
        .where(_UPTIME)
        .where(MeteredUsage.sku.in_(skus))
        .group_by(MeteredUsage.sku, MeteredUsage.sku_count)
    )
    rows = (
        await session.exec(
            select(
                MeteredUsage.sku,
                MeteredUsage.sku_count,
                MeteredUsage.dimensions,
            ).where(MeteredUsage.id.in_(rep_ids))
        )
    ).all()
    return {(r[0], r[1]): (r[2] or {}) for r in rows}


async def _attach_dimensions(session, gb: str, items: List[dict]) -> None:
    """Attach the ``dimensions`` the UI needs per grouping (in place).

    Only the ``instance_type`` / ``instance`` / ``volume`` groupings carry
    dimensions; for any other grouping (e.g. ``user`` / ``date``) this is a
    no-op.
    """
    if gb == "instance_type":
        # One representative row per (sku, sku_count) supplies the whole shape's
        # display blob (product / per-unit specs / VRAM are flavor-constant;
        # cpu/mem/cards are constant within the shape).
        dims = await _dims_by_shape(
            session,
            {(i.get("sku"), i.get("_sku_count")) for i in items},
        )
        for i in items:
            d = dims.get((i.get("sku"), i.pop("_sku_count", None))) or {}
            i["dimensions"] = {
                "product": d.get("product"),
                "unit_cpu_milli": d.get("unit_cpu_milli"),
                "unit_memory_mib": d.get("unit_memory_mib"),
                "vram_mib": d.get("vram_mib"),
                # GPU card count (0 for CPU) — the UI uses it to tell GPU from
                # CPU and to render "<product> x <cards>".
                "gpu_count": d.get("gpu_count"),
                # Instance totals (requested cpu/ram) so each row shows its real
                # size — "CPU Only · 3 vCPU · 6 GB" / "NVIDIA-A100-80GB x 4".
                "cpu_milli": d.get("cpu_milli"),
                "memory_mib": d.get("memory_mib"),
            }
    elif gb == "instance":
        # Per-instance dims: gpu_count varies per instance (unlike the flavor),
        # so the Instances table renders "<product> x <count>" plus the spec
        # popover like the GPU Instances list. Keyed by resource_id since the
        # sku is count-independent. ``persistent_mib`` was snapshotted at
        # metering time, so it survives the PV being deleted later.
        dims = await _dims_by_representative(
            session,
            group_col=MeteredUsage.resource_id,
            keys=[i.get("id") for i in items if i.get("id") is not None],
            extra_filter=_UPTIME,
        )
        for i in items:
            d = dims.get(i.get("id")) or {}
            i["dimensions"] = {
                "product": d.get("product"),
                "unit_cpu_milli": d.get("unit_cpu_milli"),
                "unit_memory_mib": d.get("unit_memory_mib"),
                "vram_mib": d.get("vram_mib"),
                "gpu_count": d.get("gpu_count"),
                # Instance totals (requested cpu/ram) — what the row actually
                # holds, vs the per-unit flavor specs above. The headline spec
                # for CPU instances (gpu_count=0), where unit_* alone would
                # understate a multi-unit request (e.g. 4c8g on a 1c2g flavor).
                "cpu_milli": d.get("cpu_milli"),
                "memory_mib": d.get("memory_mib"),
                "ephemeral_mib": d.get("ephemeral_mib"),
                "local_storage_mib": d.get("local_storage_mib"),
                "persistent_mib": d.get("persistent_mib"),
            }
    elif gb == "volume":
        # Per-volume storage type + provisioned capacity (constant per volume),
        # for the Storage tab's Type / Capacity columns.
        dims = await _dims_by_representative(
            session,
            group_col=MeteredUsage.resource_id,
            keys=[i.get("id") for i in items if i.get("id") is not None],
            extra_filter=MeteredUsage.meter_key == METER_STORAGE_CAPACITY,
        )
        for i in items:
            d = dims.get(i.get("id")) or {}
            i["dimensions"] = {
                "storage_type": d.get("storage_type"),
                "capacity_mib": d.get("capacity_mib"),
            }


async def _run_breakdown(
    session,
    *,
    user,
    ctx,
    request: ResourceBreakdownRequest,
    base_filter,
    metric_keys: List[str],
    join_instances: bool = True,
    join_volumes: bool = True,
):
    effective_scope = _resolve_effective_scope(user, ctx, request.scope)
    metrics = _metric_columns()
    metric_keys = [k for k in metric_keys if k in metrics]

    resource_ids = (request.instance_ids or []) + (request.volume_ids or [])
    base = select().select_from(MeteredUsage)
    base = _apply_scope(
        base,
        user=user,
        ctx=ctx,
        scope=effective_scope,
        base_filter=base_filter,
        creator_ids=request.creator_ids,
        resource_ids=resource_ids or None,
    )
    base = _bucket_in_range(base, request.start_date, request.end_date)

    # "resources" = Active Instances/Volumes, so it must EXCLUDE rows whose
    # resource was since deleted (metered_usage keeps deleted rows for the record,
    # but the UI counts only live ones). LEFT JOIN the source tables on the
    # resource_id — type-qualified so the instance/volume id spaces don't cross —
    # and count only ids that still resolve. Usage metrics (gpu_hours, …) keep
    # summing every row, deleted or not. The 1:1 join (id PK) doesn't fan rows.
    #
    # Only join the table(s) the endpoint can actually return: the GPU tab never
    # has PV rows and vice versa, so a single-resource endpoint skips the dead
    # join (the resource breakdown spans both and keeps both). Avoids a join the
    # planner may not always elide.
    live_conds = []
    if join_instances:
        base = base.join(
            GPUInstance,
            and_(
                GPUInstance.id == MeteredUsage.resource_id,
                MeteredUsage.resource_type.in_(
                    [RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE]
                ),
            ),
            isouter=True,
        )
        live_conds.append(GPUInstance.id.isnot(None))
    if join_volumes:
        base = base.join(
            GPUInstancePersistentVolume,
            and_(
                GPUInstancePersistentVolume.id == MeteredUsage.resource_id,
                MeteredUsage.resource_type == RESOURCE_TYPE_PERSISTENT_VOLUME,
            ),
            isouter=True,
        )
        live_conds.append(GPUInstancePersistentVolume.id.isnot(None))
    active_resource_id = case(
        (or_(*live_conds), MeteredUsage.resource_id),
        else_=None,
    )

    # summary row (no grouping)
    summary_cols = [metrics[k].label(k) for k in metric_keys]
    summary_cols += [
        func.count(func.distinct(active_resource_id)).label("resources"),
        func.count(func.distinct(MeteredUsage.creator_id)).label("active_users"),
    ]
    summary_row = (await session.exec(base.with_only_columns(*summary_cols))).first()

    # Combine each grouping dimension left-to-right (e.g. ["date",
    # "instance_type"] → one (date, sku) row per bucket per group). Only "date"
    # is ever paired with another dimension, so the labels (group_date vs
    # group_key/group_id) don't clash.
    select_cols: List[Any] = []
    group_cols: List[Any] = []
    for gb in request.group_by:
        sc, gc = _group_columns(gb, session, request.granularity)
        select_cols.extend(sc)
        group_cols.extend(gc)
    agg_cols = [metrics[k].label(k) for k in metric_keys]
    agg_cols += [
        func.count(func.distinct(active_resource_id)).label("resources"),
        func.count(func.distinct(MeteredUsage.creator_id)).label("active_users"),
        func.max(MeteredUsage.bucket_start).label("last_active"),
        # One sku per resource — carry it so per-instance / per-volume rows can
        # show their Instance Type / Storage Type even when grouped by resource.
        func.max(MeteredUsage.sku).label("sku"),
    ]
    grouped = base.with_only_columns(*select_cols, *agg_cols).group_by(*group_cols)

    order_key = request.order_by or (metric_keys[0] if metric_keys else None)
    if order_key and order_key in metrics:
        order_expr = metrics[order_key]
        grouped = grouped.order_by(
            desc(order_expr) if request.descending else order_expr
        )

    count_stmt = select(func.count()).select_from(grouped.subquery())
    total = (await session.exec(count_stmt)).first() or 0

    # No-pagination returns the whole series; ``total`` (bucket count) is known
    # here, so reject an over-large result before fetching its rows rather than
    # silently truncating (which would reintroduce the chart-gap bug).
    if request.page <= 0 and total > envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS:
        raise InvalidException(
            message=(
                f"Result set too large ({total} buckets, limit "
                f"{envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS}). Narrow the "
                "date range or add filters."
            )
        )

    # ``page <= 0`` is the project-wide "no pagination" sentinel (mirrors
    # ActiveRecordMixin.page_query): the trend chart needs every bucket, so an
    # order-by-metric page would drop low-usage (often most recent) buckets and
    # leave gaps. Replaces the old ``perPage=10000`` workaround on the client.
    items_stmt = grouped
    if request.page > 0:
        items_stmt = items_stmt.offset((request.page - 1) * request.perPage).limit(
            request.perPage
        )
    rows = (await session.exec(items_stmt)).all()

    def metrics_of(row) -> Dict[str, Any]:
        out = {k: round(float(getattr(row, k, 0) or 0), 2) for k in metric_keys}
        out["resources"] = int(getattr(row, "resources", 0) or 0)
        out["active_users"] = int(getattr(row, "active_users", 0) or 0)
        return out

    # Resolve the rollup tz once per request (not per row): the DST-correct tz
    # for instants, plus the fixed-offset tz that labels the SQL-shifted buckets.
    aware_tz = resolve_rollup_tz()
    fixed_tz = rollup_fixed_tz()
    items = []
    for row in rows:
        item = {"metrics": metrics_of(row)}
        # Carry whichever grouping columns this query selected — a compound
        # (date + dimension) trend row has both ``group_date`` and ``group_key``.
        if hasattr(row, "group_date"):
            item["date"] = _localize_bucket(
                getattr(row, "group_date", None), request.granularity, fixed_tz
            )
        if hasattr(row, "group_key"):
            item["key"] = getattr(row, "group_key", None)
        if hasattr(row, "group_id"):
            item["id"] = getattr(row, "group_id", None)
        # instance_type rows are grouped by (sku, sku_count); carry sku_count so
        # enrichment can fetch the right per-shape representative for display.
        if hasattr(row, "group_sku_count"):
            item["_sku_count"] = getattr(row, "group_sku_count", None)
        item["sku"] = getattr(row, "sku", None)
        # max(bucket_start) is a UTC instant → show it in the rollup tz, aware
        # (carries an offset) so the API is self-describing and the UI renders
        # the rollup wall clock via parseZone without re-converting.
        item["metrics"]["last_active"] = to_rollup_aware(
            getattr(row, "last_active", None), aware_tz
        )
        items.append(item)

    # Resolve display fields for the secondary dimension regardless of whether
    # a date axis is present — a grouped trend (["date", <dim>]) needs them too,
    # else e.g. instance_type series carry the raw flavor slug instead of the
    # pretty product name in the chart legend (#5700). ``granularity`` only
    # changes the time bucket, never the group_by tokens, so this is granularity
    # agnostic (hour/day/week/month all share the ["date", <dim>] shape).
    dims = [g for g in request.group_by if g != "date"]
    if dims:
        await _enrich_items(session, dims[0], items)

    return {
        "summary": metrics_of(summary_row) if summary_row is not None else {},
        "group_by": request.group_by,
        "pagination": Pagination(
            page=request.page,
            perPage=request.perPage,
            total=total,
            totalPage=(
                ceil(total / request.perPage)
                if request.page > 0 and total
                else (1 if total else 0)
            ),
        ),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/resource/breakdown")
async def resource_breakdown(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    request: ResourceBreakdownRequest,
):
    return await _run_breakdown(
        session,
        user=user,
        ctx=ctx,
        request=request,
        base_filter=None,
        metric_keys=["instance_hours", "gpu_hours", "gb_days"],
    )


@router.post("/gpu-instances/breakdown")
async def gpu_instances_breakdown(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    request: ResourceBreakdownRequest,
):
    allowed = {"instance_type", "instance", "user", "date"}
    request.group_by = [g for g in request.group_by if g in allowed] or [
        "instance_type"
    ]
    return await _run_breakdown(
        session,
        user=user,
        ctx=ctx,
        request=request,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            # CPU-only instances are metered as ``cpu_instance`` — the tab
            # shows both; they just contribute 0 to gpu_hours (see
            # _metric_columns).
            MeteredUsage.resource_type.in_(
                [RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE]
            ),
        ),
        metric_keys=["gpu_hours", "instance_hours"],
        join_volumes=False,  # GPU rows never resolve against the PV table
    )


@router.post("/storage/breakdown")
async def storage_breakdown(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    request: ResourceBreakdownRequest,
):
    allowed = {"type", "volume", "user", "date"}
    request.group_by = [g for g in request.group_by if g in allowed] or ["type"]
    return await _run_breakdown(
        session,
        user=user,
        ctx=ctx,
        request=request,
        base_filter=(MeteredUsage.meter_key == METER_STORAGE_CAPACITY),
        metric_keys=["gb_days", "gb_hours"],
        join_instances=False,  # PV rows never resolve against the instance table
    )


@router.get("/summary")
async def usage_summary(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    start_date: date,
    end_date: date,
    scope: str = USAGE_SCOPE_ALL,
    creator_ids: Optional[str] = None,
):
    effective_scope = _resolve_effective_scope(user, ctx, scope)
    org_id = ctx.current_principal_id if ctx is not None else None
    creator_id_list = _parse_id_csv(creator_ids)

    # metered_usage side (instances + storage)
    mu = select().select_from(MeteredUsage)
    mu = _apply_scope(
        mu, user=user, ctx=ctx, scope=effective_scope, creator_ids=creator_id_list
    )
    mu = _bucket_in_range(mu, start_date, end_date)
    metrics = _metric_columns()
    mu_row = (
        await session.exec(
            mu.with_only_columns(
                metrics["instance_hours"].label("instance_hours"),
                metrics["gpu_hours"].label("gpu_hours"),
                metrics["gb_days"].label("gb_days"),
                func.count(func.distinct(MeteredUsage.creator_id)).label(
                    "active_users"
                ),
            )
        )
    ).first()

    # token side (model_usages) — scoped by consumer_principal_id. The user
    # filter maps onto the token consumer (model_usages.user_id) so the Tokens
    # headline narrows in lockstep with the compute/storage figures.
    tu = select().select_from(ModelUsage)
    if effective_scope == USAGE_SCOPE_SELF:
        tu = tu.where(ModelUsage.user_id == user.id)
    if org_id is not None:
        tu = tu.where(ModelUsage.consumer_principal_id == org_id)
    if creator_id_list:
        tu = tu.where(ModelUsage.user_id.in_(creator_id_list))
    tu = tu.where(ModelUsage.date >= start_date).where(ModelUsage.date <= end_date)
    tu_row = (
        await session.exec(
            tu.with_only_columns(
                func.coalesce(func.sum(ModelUsage.prompt_token_count), 0).label(
                    "input_tokens"
                ),
                func.coalesce(func.sum(ModelUsage.completion_token_count), 0).label(
                    "output_tokens"
                ),
                func.count(func.distinct(ModelUsage.user_id)).label(
                    "token_active_users"
                ),
            )
        )
    ).first()

    input_tokens = int(getattr(tu_row, "input_tokens", 0) or 0)
    output_tokens = int(getattr(tu_row, "output_tokens", 0) or 0)

    return {
        # Input = prompt tokens, Output = completion tokens (cached is folded
        # into prompt). total = input + output.
        "total_tokens": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_active_users": int(getattr(tu_row, "token_active_users", 0) or 0),
        "gpu_hours": round(float(getattr(mu_row, "gpu_hours", 0) or 0), 2),
        "instance_hours": round(float(getattr(mu_row, "instance_hours", 0) or 0), 2),
        "storage_gb_days": round(float(getattr(mu_row, "gb_days", 0) or 0), 2),
        "active_users": int(getattr(mu_row, "active_users", 0) or 0),
    }


def _phase_message_of(spec_snapshot) -> Optional[str]:
    """``status.phaseMessage`` from the event's spec snapshot — the detail behind
    a failure phase (the UI shows it as the event's error message). Reads both
    the snake / camel key since serialization differs by path."""
    if isinstance(spec_snapshot, str):
        # Some drivers / replay paths hand back the JSON column as a raw string.
        try:
            spec_snapshot = json.loads(spec_snapshot)
        except (ValueError, TypeError):
            return None
    if not isinstance(spec_snapshot, dict):
        return None
    status = spec_snapshot.get("status")
    if not isinstance(status, dict):
        return None
    return status.get("phase_message") or status.get("phaseMessage")


@router.get("/resource-events")
async def resource_events(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    resource_type: Optional[str] = None,
    resource_name: Optional[str] = None,
    event_types: Optional[str] = None,
    scope: str = USAGE_SCOPE_ALL,
    creator_ids: Optional[str] = None,
    page: int = 1,
    perPage: int = 50,
):
    effective_scope = _resolve_effective_scope(user, ctx, scope)
    org_id = ctx.current_principal_id if ctx is not None else None
    creator_id_list = _parse_id_csv(creator_ids)
    # Comma-separated event types (e.g. "created,deleted"); the only emitted
    # types are created / deleted / phase_to_metered / phase_left_metered.
    event_type_list = (
        [e.strip() for e in event_types.split(",") if e.strip()]
        if event_types
        else None
    )

    stmt = select(ResourceEvent)
    if effective_scope == USAGE_SCOPE_SELF:
        stmt = stmt.where(ResourceEvent.creator_id == user.id)
    if org_id is not None:
        # Tenant scope follows the consumer, same as metered_usage.
        stmt = stmt.where(ResourceEvent.consumer_principal_id == org_id)
    if creator_id_list:
        stmt = stmt.where(ResourceEvent.creator_id.in_(creator_id_list))
    if resource_type:
        stmt = stmt.where(ResourceEvent.resource_type == resource_type)
    if event_type_list:
        stmt = stmt.where(ResourceEvent.event_type.in_(event_type_list))
    if resource_name and resource_name.strip():
        # Case-insensitive substring ("fuzzy") match on the resource name —
        # func.lower(...) LIKE keeps it portable across PostgreSQL and MySQL.
        needle = f"%{resource_name.strip().lower()}%"
        stmt = stmt.where(func.lower(ResourceEvent.resource_name).like(needle))
    # Filter by the rollup-tz calendar day, matching how occurred_at is
    # displayed (and the breakdown buckets) — NOT the raw UTC day. Otherwise an
    # event at 2026-05-26 20:00 UTC, shown as 2026-05-27 04:00+08:00, would be
    # missed by a 2026-05-27 filter and wrongly returned by a 2026-05-26 one
    # (the #5523 cross-boundary bug). Half-open UTC window, no per-row cast.
    ev_start, ev_end = _rollup_day_window(start_date, end_date)
    if ev_start is not None:
        stmt = stmt.where(ResourceEvent.occurred_at >= ev_start)
    if ev_end is not None:
        stmt = stmt.where(ResourceEvent.occurred_at < ev_end)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.exec(count_stmt)).first() or 0

    rows = (
        await session.exec(
            stmt.order_by(desc(ResourceEvent.occurred_at))
            .offset((page - 1) * perPage)
            .limit(perPage)
        )
    ).all()

    aware_tz = resolve_rollup_tz()  # resolved once, not per row
    return {
        "pagination": Pagination(
            page=page,
            perPage=perPage,
            total=total,
            totalPage=ceil(total / perPage) if total else 0,
        ),
        "items": [
            {
                "id": r.id,
                # Show the event instant in the rollup tz, aware (carries an
                # offset) so the API is self-describing and the UI renders the
                # rollup wall clock via parseZone without re-converting.
                "occurred_at": to_rollup_aware(r.occurred_at, aware_tz),
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "resource_name": r.resource_name,
                "event_type": r.event_type,
                "event_message": r.event_message,
                "phase": r.phase,
                "phase_message": _phase_message_of(r.spec_snapshot),
                "creator_id": r.creator_id,
                "creator_name": r.creator_name,
            }
            for r in rows
        ],
    }


@router.get("/resource/meta")
async def resource_meta(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    scope: str = USAGE_SCOPE_ALL,
):
    """Filter dropdown source for the resource tabs:

    * ``creators`` — distinct creators (principals) in the caller's scope, union
      of ``metered_usage`` (metered rollups) and ``resource_events`` (lifecycle
      log superset). Drives the "filter by user" select on every tab.
    * ``instances`` / ``volumes`` — distinct resources in scope (id + snapshot
      name), driving the per-entity select on the GPU Instances / Storage tabs.

    Mirrors the Tokens tab's ``/usage/meta`` filter lists."""
    effective_scope = _resolve_effective_scope(user, ctx, scope)
    org_id = ctx.current_principal_id if ctx is not None else None

    def _scoped(stmt, creator_col, scope_col):
        if effective_scope == USAGE_SCOPE_SELF:
            stmt = stmt.where(creator_col == user.id)
        if org_id is not None:
            # Tenant scope = consumer_principal_id.
            stmt = stmt.where(scope_col == org_id)
        return stmt

    async def _distinct_creators(creator_col, scope_col) -> Set[Optional[int]]:
        stmt = _scoped(
            select(creator_col).where(creator_col.isnot(None)).distinct(),
            creator_col,
            scope_col,
        )
        return set((await session.exec(stmt)).all())

    ids = await _distinct_creators(
        MeteredUsage.creator_id, MeteredUsage.consumer_principal_id
    )
    ids |= await _distinct_creators(
        ResourceEvent.creator_id, ResourceEvent.consumer_principal_id
    )
    ids.discard(None)

    creators = []
    if ids:
        principals = (
            await session.exec(select(Principal).where(Principal.id.in_(ids)))
        ).all()
        name_by_id = {p.id: (p.display_name or p.name) for p in principals}
        # A creator id that no longer resolves to a principal was deleted —
        # flag it so the filter shows a "(Deleted)" tag (same as the Tokens tab).
        creators = [
            {
                "id": cid,
                "label": name_by_id.get(cid) or f"User {cid}",
                "deleted": cid not in name_by_id,
            }
            for cid in ids
        ]
        creators.sort(key=lambda c: (c["label"] or "").lower())

    # Distinct resources of one type — id + latest snapshot name. metered_usage
    # snapshots the name, so deleted resources still resolve a label; flag the
    # ones no longer live (not in the source table) so the filter can tag them.
    async def _resources(
        resource_types: List[str],
        model: Type[Union[GPUInstance, GPUInstancePersistentVolume]],
    ) -> List[Dict[str, Any]]:
        stmt = _scoped(
            select(
                MeteredUsage.resource_id,
                func.max(MeteredUsage.resource_name).label("name"),
            )
            .where(MeteredUsage.resource_type.in_(resource_types))
            .where(MeteredUsage.resource_id.isnot(None))
            .group_by(MeteredUsage.resource_id),
            MeteredUsage.creator_id,
            MeteredUsage.consumer_principal_id,
        )
        rows = (await session.exec(stmt)).all()
        rids = [r.resource_id for r in rows]
        live: set[int] = set()
        if rids:
            live = set(
                (await session.exec(select(model.id).where(model.id.in_(rids)))).all()
            )
        out = [
            {
                "id": r.resource_id,
                "label": r.name or f"#{r.resource_id}",
                "deleted": r.resource_id not in live,
            }
            for r in rows
        ]
        out.sort(key=lambda x: (x["label"] or "").lower())
        return out

    instances = await _resources(
        [RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_CPU_INSTANCE], GPUInstance
    )
    volumes = await _resources(
        [RESOURCE_TYPE_PERSISTENT_VOLUME], GPUInstancePersistentVolume
    )

    return {"creators": creators, "instances": instances, "volumes": volumes}
