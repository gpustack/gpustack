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

from datetime import date, datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import Date, and_, case, cast, desc, func, or_
from sqlmodel import select

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
    group_by: str = "resource_type"
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
    page: int = Field(default=1, ge=1)
    # Upper bound is generous (not 100) because the export path fetches the
    # whole filtered set in one page (perPage=10000); cap only blocks abuse.
    perPage: int = Field(default=20, ge=1, le=10000)


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
    if group_by in ("instance_type", "type", "sku"):
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


def _bucket_expr(session, granularity: str):
    """Time-bucket expression over ``bucket_start`` for the requested
    granularity. Rows are stored at hourly granularity; coarser buckets are
    derived via date_trunc (PostgreSQL) / date functions (MySQL)."""
    col = MeteredUsage.bucket_start
    if granularity == USAGE_GRANULARITY_HOUR:
        return col  # already hour-truncated by the collector
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        unit = {
            USAGE_GRANULARITY_WEEK: "week",
            USAGE_GRANULARITY_MONTH: "month",
        }.get(granularity, "day")
        return func.date_trunc(unit, col)
    # MySQL
    if granularity == USAGE_GRANULARITY_MONTH:
        return func.date_format(col, "%Y-%m-01")
    if granularity == USAGE_GRANULARITY_WEEK:
        # Monday-start week, matching PostgreSQL's date_trunc('week').
        return func.subdate(func.date(col), func.weekday(col))
    return func.date(col)  # day


def _bucket_in_range(statement, start_date: date, end_date: date):
    """Filter ``bucket_start`` (datetime) to the inclusive [start_date, end_date]
    day range, as half-open datetime bounds so the index is usable."""
    start_dt = datetime(start_date.year, start_date.month, start_date.day)
    end_dt = datetime(end_date.year, end_date.month, end_date.day) + timedelta(days=1)
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
    if gb in ("user", "instance", "volume"):
        ids = [i["id"] for i in items if i.get("id") is not None]
        existing: set = set()
        names: dict = {}
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

    if gb in ("instance_type", "instance"):
        skus = [i.get("sku") for i in items if i.get("sku")]
        dims_by_sku: dict = {}
        if skus:
            rep_ids = (
                select(func.max(MeteredUsage.id))
                .where(_UPTIME)
                .where(MeteredUsage.sku.in_(skus))
                .group_by(MeteredUsage.sku)
            )
            rows = (
                await session.exec(
                    select(MeteredUsage.sku, MeteredUsage.dimensions).where(
                        MeteredUsage.id.in_(rep_ids)
                    )
                )
            ).all()
            dims_by_sku = {r[0]: (r[1] or {}) for r in rows}
        for i in items:
            d = dims_by_sku.get(i.get("sku")) or {}
            i["dimensions"] = {
                "product": d.get("product"),
                "unit_cpu_milli": d.get("unit_cpu_milli"),
                "unit_memory_mib": d.get("unit_memory_mib"),
                "vram_mib": d.get("vram_mib"),
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

    select_cols, group_cols = _group_columns(
        request.group_by, session, request.granularity
    )
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

    items_stmt = grouped.offset((request.page - 1) * request.perPage).limit(
        request.perPage
    )
    rows = (await session.exec(items_stmt)).all()

    def metrics_of(row) -> Dict[str, Any]:
        out = {k: round(float(getattr(row, k, 0) or 0), 2) for k in metric_keys}
        out["resources"] = int(getattr(row, "resources", 0) or 0)
        out["active_users"] = int(getattr(row, "active_users", 0) or 0)
        return out

    items = []
    for row in rows:
        item = {"metrics": metrics_of(row)}
        if request.group_by == "date":
            item["date"] = getattr(row, "group_date", None)
        else:
            item["key"] = getattr(row, "group_key", None)
            if hasattr(row, "group_id"):
                item["id"] = getattr(row, "group_id", None)
        item["sku"] = getattr(row, "sku", None)
        item["metrics"]["last_active"] = getattr(row, "last_active", None)
        items.append(item)

    await _enrich_items(session, request.group_by, items)

    return {
        "summary": metrics_of(summary_row) if summary_row is not None else {},
        "group_by": request.group_by,
        "pagination": Pagination(
            page=request.page,
            perPage=request.perPage,
            total=total,
            totalPage=ceil(total / request.perPage) if total else 0,
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
    if request.group_by not in ("instance_type", "instance", "user", "date"):
        request.group_by = "instance_type"
    return await _run_breakdown(
        session,
        user=user,
        ctx=ctx,
        request=request,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
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
    if request.group_by not in ("type", "volume", "user", "date"):
        request.group_by = "type"
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


@router.get("/resource-events")
async def resource_events(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    resource_type: Optional[str] = None,
    scope: str = USAGE_SCOPE_ALL,
    creator_ids: Optional[str] = None,
    page: int = 1,
    perPage: int = 50,
):
    effective_scope = _resolve_effective_scope(user, ctx, scope)
    org_id = ctx.current_principal_id if ctx is not None else None
    creator_id_list = _parse_id_csv(creator_ids)

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
    if start_date is not None:
        stmt = stmt.where(cast(ResourceEvent.occurred_at, Date) >= start_date)
    if end_date is not None:
        stmt = stmt.where(cast(ResourceEvent.occurred_at, Date) <= end_date)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.exec(count_stmt)).first() or 0

    rows = (
        await session.exec(
            stmt.order_by(desc(ResourceEvent.occurred_at))
            .offset((page - 1) * perPage)
            .limit(perPage)
        )
    ).all()

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
                "occurred_at": r.occurred_at,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "resource_name": r.resource_name,
                "event_type": r.event_type,
                "event_message": r.event_message,
                "phase": r.phase,
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
    async def _resources(resource_type: str, model) -> List[Dict[str, Any]]:
        stmt = _scoped(
            select(
                MeteredUsage.resource_id,
                func.max(MeteredUsage.resource_name).label("name"),
            )
            .where(MeteredUsage.resource_type == resource_type)
            .where(MeteredUsage.resource_id.isnot(None))
            .group_by(MeteredUsage.resource_id),
            MeteredUsage.creator_id,
            MeteredUsage.consumer_principal_id,
        )
        rows = (await session.exec(stmt)).all()
        rids = [r.resource_id for r in rows]
        live: set = set()
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

    instances = await _resources(RESOURCE_TYPE_GPU_INSTANCE, GPUInstance)
    volumes = await _resources(
        RESOURCE_TYPE_PERSISTENT_VOLUME, GPUInstancePersistentVolume
    )

    return {"creators": creators, "instances": instances, "volumes": volumes}
