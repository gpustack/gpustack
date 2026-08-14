"""Integration tests for the metered_usage read API SQL — exercises the real
aggregation (case/coalesce/group-by) against an in-memory sqlite engine."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ForbiddenException
from gpustack.routes.resource_usage import (
    ResourceBreakdownRequest,
    _phase_message_of,
    _run_breakdown,
    usage_summary,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    RESOURCE_TYPE_CPU_INSTANCE,
    RESOURCE_TYPE_GPU_INSTANCE,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    MeteredUsage,
)
from gpustack.schemas.model_usage import ModelUsage

D = date(2026, 5, 26)
BUCKET = datetime(2026, 5, 26, 10, 0, 0)  # an hour bucket within day D
NOW = datetime(2026, 5, 26, 12, 0, 0)
USER = SimpleNamespace(id=7, is_admin=True)
CTX = SimpleNamespace(current_principal_id=None)


def _mu(**kw):
    base = dict(
        owner_principal_id=1,
        creator_id=7,
        bucket_start=BUCKET,
        created_at=NOW,
        updated_at=NOW,
        sku_count=1,
    )
    base.update(kw)
    return MeteredUsage(**base)


def _gi(id_: int, name: str):
    """A minimal *live* GPUInstance so its id resolves as 'active' (not deleted)
    in the breakdown's resources count."""
    from gpustack.schemas.gpu_instances import GPUInstance

    return GPUInstance(
        id=id_,
        name=name,
        owner_principal_id=1,
        cluster_id=2,
        spec={"type_": "gpu", "image": "busybox"},
    )


async def _seed(session):
    session.add_all(
        [
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=501,
                resource_name="gpu-1",
                sku="h100x2",
                sku_count=2,
                quantity=49795,
                unit="seconds",
            ),
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=502,
                resource_name="gpu-2",
                sku="a100x1",
                sku_count=1,
                quantity=3600,
                unit="seconds",
            ),
            _mu(
                meter_key=METER_STORAGE_CAPACITY,
                resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
                resource_id=88,
                resource_name="pv-models",
                sku="ssd",
                quantity=204800 * 25200,
                unit="mib_seconds",
            ),
        ]
    )
    await session.commit()


@pytest_asyncio.fixture
async def session():
    from gpustack.schemas.principals import Principal, PrincipalMembership
    from gpustack.schemas.gpu_instances import GPUInstance
    from gpustack.schemas.gpu_instance_persistent_volumes import (
        GPUInstancePersistentVolume,
    )
    from gpustack.schemas.resource_events import ResourceEvent

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for table in (
            MeteredUsage.__table__,
            ModelUsage.__table__,
            Principal.__table__,
            PrincipalMembership.__table__,
            GPUInstance.__table__,
            GPUInstancePersistentVolume.__table__,
            ResourceEvent.__table__,
        ):
            await conn.run_sync(table.create)
    async with AsyncSession(engine) as s:
        await _seed(s)
        s.add(Principal(id=7, kind="user", name="alice", display_name="Alice"))
        # gpu-2 (502) is a live instance → counts as active; gpu-1 (501) is left
        # unseeded → since-deleted (its usage stays, but it's not an "active"
        # instance). pv-models (88) is left unseeded too.
        s.add(_gi(502, "gpu-2"))
        await s.commit()
        yield s
    await engine.dispose()


def _req(group_by):
    return ResourceBreakdownRequest(
        scope="self", start_date=D, end_date=D, group_by=[group_by]
    )


@pytest.mark.asyncio
async def test_gpu_instances_by_instance_carries_sku(session):
    # grouping by instance must still surface each instance's sku (Instance Type)
    from sqlalchemy import and_

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours", "instance_hours"],
    )
    by_name = {i["key"]: i for i in out["items"]}
    assert by_name["gpu-1"]["sku"] == "h100x2"
    assert by_name["gpu-2"]["sku"] == "a100x1"
    # instances aren't seeded into gpu_instances → flagged deleted
    assert by_name["gpu-1"]["deleted"] is True


@pytest.mark.asyncio
async def test_instance_grouping_carries_creator_resource_type_omits(session):
    """Resource-dimension groupings carry the owner (constant per resource);
    coarser groupings that can span multiple creators omit it entirely rather
    than emit a MAX(id)/MAX(name) mismatch."""
    from sqlalchemy import and_

    gpu_filter = and_(
        MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
        MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
    )
    by_instance = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=gpu_filter,
        metric_keys=["gpu_hours"],
    )
    # every seeded row is creator 7 → each per-instance row carries that owner
    assert by_instance["items"]
    for i in by_instance["items"]:
        assert i["creator_id"] == 7
        assert "creator_name" in i

    by_type = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("resource_type"),
        base_filter=None,
        metric_keys=["instance_hours"],
    )
    # non-resource grouping (a group may span many creators) → fields absent
    assert by_type["items"]
    for i in by_type["items"]:
        assert "creator_id" not in i
        assert "creator_name" not in i


@pytest.mark.asyncio
async def test_instance_creator_deleted_flag(session):
    """Per-instance rows flag whether the owner principal still exists so the
    UI can tag a since-deleted creator '(Deleted)', keeping the snapshot name."""
    from sqlalchemy import and_

    from gpustack.schemas.principals import Principal

    # gpu-3 is owned by creator 99, which has no Principal row (deleted user).
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=503,
            resource_name="gpu-3",
            sku="a100x1",
            quantity=3600,
            unit="seconds",
            creator_id=99,
            creator_name="bob",
        )
    )
    # gpu-4 is owned by creator 8, whose principal row still exists but is
    # soft-deleted (``deleted_at`` set) — must flag deleted just like a gone id.
    session.add(
        Principal(id=8, kind="user", name="carol", display_name="Carol", deleted_at=NOW)
    )
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=504,
            resource_name="gpu-4",
            sku="a100x1",
            quantity=3600,
            unit="seconds",
            creator_id=8,
            creator_name="carol",
        )
    )
    await session.commit()

    # scope="all" so creator 99's row isn't clamped out by the self filter.
    req = ResourceBreakdownRequest(
        scope="all", start_date=D, end_date=D, group_by=["instance"]
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    by_name = {i["key"]: i for i in out["items"]}
    # creator 7 (alice) still exists → not deleted
    assert by_name["gpu-1"]["creator_deleted"] is False
    # creator 99 has no principal → flagged deleted, snapshot name preserved
    assert by_name["gpu-3"]["creator_deleted"] is True
    assert by_name["gpu-3"]["creator_name"] == "bob"
    # creator 8 exists but is soft-deleted → also flagged deleted
    assert by_name["gpu-4"]["creator_deleted"] is True


@pytest.mark.asyncio
async def test_user_grouping_resolves_principal_name(session):
    from sqlalchemy import and_

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("user"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    # creator_id 7 → principal login name "alice" (not display name), matching
    # the Tokens tab which groups users by login name.
    assert any(i.get("key") == "alice" for i in out["items"])


@pytest.mark.asyncio
async def test_resource_breakdown_by_type(session):
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("resource_type"),
        base_filter=None,
        metric_keys=["instance_hours", "gpu_hours", "gb_days"],
    )
    by_key = {i["key"]: i["metrics"] for i in out["items"]}
    assert set(by_key) == {RESOURCE_TYPE_GPU_INSTANCE, RESOURCE_TYPE_PERSISTENT_VOLUME}
    gpu = by_key[RESOURCE_TYPE_GPU_INSTANCE]
    # instance-hours = (49795 + 3600) / 3600
    assert gpu["instance_hours"] == pytest.approx((49795 + 3600) / 3600, abs=0.01)
    # gpu-hours = (49795*2 + 3600*1) / 3600 — includes gpu-1's usage even though
    # gpu-1 is since-deleted (usage is metering truth and stays).
    assert gpu["gpu_hours"] == pytest.approx((49795 * 2 + 3600) / 3600, abs=0.01)
    # resources = Active Instances: only gpu-2 (502) is live; gpu-1 (501) was
    # deleted, so it drops out of the count (but not the hours above).
    assert gpu["resources"] == 1
    pv = by_key[RESOURCE_TYPE_PERSISTENT_VOLUME]
    assert pv["gb_days"] == pytest.approx(204800 * 25200 / 1024 / 86400, abs=0.01)
    # summary sums everything
    assert out["summary"]["gpu_hours"] == pytest.approx(
        (49795 * 2 + 3600) / 3600, abs=0.01
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["day", "hour"])
async def test_trend_enriches_instance_type_product(session, granularity):
    # A grouped trend ["date", "instance_type"] must carry the flavor's pretty
    # product name (dimensions.product), not just the raw sku slug, so the chart
    # legend matches the GPU Instances list (#5700). Enrichment is granularity
    # agnostic — date/hour share the ["date", <dim>] shape (week/month use the
    # same path but their bucket SQL isn't sqlite-portable, so aren't run here).
    from sqlalchemy import and_

    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=701,
            resource_name="gpu-x",
            sku="flavor-a100x4",
            sku_count=4,
            quantity=3600,
            unit="seconds",
            dimensions={
                "product": "NVIDIA-A100-80G",
                "unit_cpu_milli": 1000,
                "unit_memory_mib": 2048,
                "vram_mib": 81920,
            },
        )
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(
            scope="self",
            start_date=D,
            end_date=D,
            group_by=["date", "instance_type"],
            granularity=granularity,
        ),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    by_sku = {i.get("sku"): i for i in out["items"]}
    assert by_sku["flavor-a100x4"]["dimensions"]["product"] == "NVIDIA-A100-80G"


def test_resource_breakdown_request_rejects_page_zero():
    with pytest.raises(ValueError):
        ResourceBreakdownRequest(
            scope="self",
            start_date=D,
            end_date=D,
            group_by=["resource_type"],
            page=0,
        )


def test_resource_breakdown_request_rejects_other_negative_page():
    # Only -1 is the no-pagination sentinel; other negatives are rejected.
    for bad in (-2, -42):
        with pytest.raises(ValueError):
            ResourceBreakdownRequest(
                scope="self",
                start_date=D,
                end_date=D,
                group_by=["resource_type"],
                page=bad,
            )


@pytest.mark.asyncio
async def test_breakdown_no_pagination_rejects_oversized(session, monkeypatch):
    from gpustack.api.exceptions import InvalidException

    # resource_type yields two buckets; cap at 1 so page=-1 is rejected rather
    # than silently truncated.
    monkeypatch.setattr(
        "gpustack.routes.resource_usage.envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS",
        1,
    )
    with pytest.raises(InvalidException):
        await _run_breakdown(
            session,
            user=USER,
            ctx=CTX,
            request=ResourceBreakdownRequest(
                scope="self",
                start_date=D,
                end_date=D,
                group_by=["resource_type"],
                page=-1,
            ),
            base_filter=None,
            metric_keys=["gb_days"],
        )


@pytest.mark.asyncio
async def test_breakdown_no_pagination_returns_all_groups(session):
    # resource_type yields two buckets (GPU instance + persistent volume).
    base = dict(scope="self", start_date=D, end_date=D, group_by=["resource_type"])
    # perPage=1 truncates to a single bucket...
    paged = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(**base, page=1, perPage=1),
        base_filter=None,
        metric_keys=["gb_days"],
    )
    assert len(paged["items"]) == 1
    assert paged["pagination"].total == 2
    assert paged["pagination"].totalPage == 2

    # ...page=-1 (no-pagination sentinel) returns every bucket.
    full = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(**base, page=-1),
        base_filter=None,
        metric_keys=["gb_days"],
    )
    assert len(full["items"]) == 2
    assert full["pagination"].page == -1
    assert full["pagination"].totalPage == 1


@pytest.mark.asyncio
async def test_instance_type_splits_by_actual_shape(session):
    # Same flavor sku, different actual sizes → separate Instance Types rows,
    # each carrying its own cpu/mem (read from dimensions, so it works on
    # historical rows). A 1c2g and a 3c6g instance of "generic-1c-2g" must not
    # collapse into one row.
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        sku="generic-1c-2g",
        quantity=3600,
        unit="seconds",
    )
    session.add_all(
        [
            _mu(
                resource_id=801,
                resource_name="cpu-small",
                sku_count=1,  # 1c2g = 1 unit
                dimensions={
                    "gpu_count": 0,
                    "cpu_milli": 1000,
                    "memory_mib": 2048,
                    "unit_cpu_milli": 1000,
                    "unit_memory_mib": 2048,
                },
                **common,
            ),
            _mu(
                resource_id=802,
                resource_name="cpu-big",
                sku_count=3,  # 3c6g = 3 units
                dimensions={
                    "gpu_count": 0,
                    "cpu_milli": 3000,
                    "memory_mib": 6144,
                    "unit_cpu_milli": 1000,
                    "unit_memory_mib": 2048,
                },
                **common,
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance_type"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["instance_hours"],
    )
    by_cpu = {
        i["dimensions"]["cpu_milli"]: i["dimensions"]
        for i in out["items"]
        if i.get("sku") == "generic-1c-2g"
    }
    assert set(by_cpu) == {1000, 3000}
    assert by_cpu[1000]["memory_mib"] == 2048
    assert by_cpu[3000]["memory_mib"] == 6144


@pytest.mark.asyncio
async def test_unit_hours_exposes_the_billed_quantity_for_cpu_instances(session):
    """``instance_hours`` is unweighted wall clock and ``gpu_hours`` is zero for
    a CPU instance, so between them NOTHING on the page reflected a CPU row's
    ``sku_count`` — a 4-unit instance looked identical to a 1-unit one, and even
    *smaller* when it ran for less wall time. ``unit_hours`` is the quantity the
    invoice multiplies, so it must invert that.
    """
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        sku="sha1:" + "7" * 40,
        unit="seconds",
        instance_type_name="generic",
    )
    session.add_all(
        [
            # c1: 1 unit, runs LONGER.
            _mu(
                resource_id=931,
                resource_name="c1",
                sku_count=1,
                quantity=3698,
                **common,
            ),
            # c2: 4 units, runs shorter — but bills far more.
            _mu(
                resource_id=932,
                resource_name="c2",
                sku_count=4,
                quantity=3467,
                **common,
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
        ),
        metric_keys=["gpu_hours", "unit_hours", "instance_hours"],
    )
    by_name = {i["key"]: i["metrics"] for i in out["items"]}
    # Wall clock says c1 is the bigger consumer …
    assert by_name["c1"]["instance_hours"] > by_name["c2"]["instance_hours"]
    # … and GPU-Hours says nothing at all about either.
    assert by_name["c1"]["gpu_hours"] == 0
    assert by_name["c2"]["gpu_hours"] == 0
    # unit_hours restores the truth: c2 bills ~3.7x c1.
    assert by_name["c2"]["unit_hours"] > by_name["c1"]["unit_hours"] * 3
    assert by_name["c1"]["unit_hours"] == by_name["c1"]["instance_hours"]  # 1 unit


@pytest.mark.asyncio
async def test_unit_hours_equals_gpu_hours_for_gpu_rows(session):
    """``unit_hours`` is ``gpu_hours`` without the GPU filter — a generalization,
    not a second, competing number. If the two ever disagree on a GPU row, one of
    them is wrong."""
    from sqlalchemy import and_

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours", "unit_hours"],
    )
    assert out["items"]
    for i in out["items"]:
        assert i["metrics"]["unit_hours"] == i["metrics"]["gpu_hours"]


@pytest.mark.asyncio
async def test_reconfigured_instance_reports_its_shape_history(session):
    """A per-instance row shows the LATEST shape but sums the whole period, so a
    mid-period reconfiguration makes the cell lie. ``shapes`` carries the split
    that ``metered_usage`` already stores, so the popover can say so."""
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        resource_id=941,
        resource_name="c2",
        sku="sha1:" + "8" * 40,
        unit="seconds",
        instance_type_name="generic",
    )
    session.add_all(
        [
            _mu(
                sku_count=2,
                quantity=2534,
                dimensions={"cpu_milli": 2000, "memory_mib": 4096, "gpu_count": 0},
                **common,
            ),
            _mu(
                bucket_start=BUCKET + timedelta(hours=1),
                sku_count=4,
                quantity=933,
                dimensions={"cpu_milli": 4000, "memory_mib": 8192, "gpu_count": 0},
                **{k: v for k, v in common.items() if k != "bucket_start"},
            ),
        ]
    )
    # A single-shape instance in the same query must NOT get a shapes array.
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_CPU_INSTANCE,
            resource_id=942,
            resource_name="c1",
            sku="sha1:" + "8" * 40,
            sku_count=1,
            quantity=3698,
            unit="seconds",
        )
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
        ),
        metric_keys=["unit_hours", "instance_hours"],
    )
    by_name = {i["key"]: i for i in out["items"]}

    # A single-shape instance still gets a shapes array — the UI renders EVERY
    # row's usage as an explicit formula, so it needs the count and per-unit spec
    # of every row, not only reconfigured ones.
    assert len(by_name["c1"]["shapes"]) == 1
    assert by_name["c1"]["shapes"][0]["sku_count"] == 1

    shapes = by_name["c2"]["shapes"]
    assert len(shapes) == 2
    # Chronological, so it reads as a sequence of changes.
    assert [s["cpu_milli"] for s in shapes] == [2000, 4000]
    # The parts must reconcile with the row's totals — that is the whole point.
    assert (
        round(sum(s["instance_hours"] for s in shapes), 2)
        == by_name["c2"]["metrics"]["instance_hours"]
    )
    assert (
        round(sum(s["unit_hours"] for s in shapes), 2)
        == by_name["c2"]["metrics"]["unit_hours"]
    )


@pytest.mark.asyncio
async def test_each_shape_carries_its_own_per_unit_spec(session):
    """The usage cell renders ``unit × count × hours = value``, so every shape
    needs the spec of ONE unit — and its own, not the row's.

    Two traps this pins:
      * the multiplicand cannot be derived as total/count (that inverts a
        ``round()``, so it breaks on non-integer requests);
      * changing the instance TYPE gives the shapes different per-unit specs,
        while the row-level ``dimensions`` only describes the latest one.
    """
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        resource_id=961,
        resource_name="c4",
        unit="seconds",
        quantity=3600,
    )
    session.add_all(
        [
            # Type A: one unit is 1c2g, instance holds 2 of them.
            _mu(
                sku="sha1:" + "a" * 40,
                sku_count=2,
                dimensions={"cpu_milli": 2000, "unit_cpu_milli": 1000},
                **common,
            ),
            # Moved to type B: one unit is 2c4g, instance holds 2 of them. Note
            # the count is IDENTICAL, so the change is invisible from count.
            _mu(
                bucket_start=BUCKET + timedelta(hours=1),
                sku="sha1:" + "b" * 40,
                sku_count=2,
                dimensions={"cpu_milli": 4000, "unit_cpu_milli": 2000},
                **{k: v for k, v in common.items() if k != "bucket_start"},
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
        ),
        metric_keys=["unit_hours", "instance_hours"],
    )
    row = next(i for i in out["items"] if i["key"] == "c4")
    shapes = row["shapes"]
    assert len(shapes) == 2
    # Each shape's own per-unit spec — NOT the row's latest one for both.
    assert [s["unit_cpu_milli"] for s in shapes] == [1000, 2000]
    # The row-level dimensions only knows the latest, which is exactly why the
    # per-shape copy is required.
    assert row["dimensions"]["unit_cpu_milli"] == 2000


@pytest.mark.asyncio
async def test_shape_history_is_scoped_to_the_queried_range(session):
    """Reporting a reconfiguration that happened OUTSIDE the selected period
    would be worse than reporting none."""
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        resource_id=951,
        resource_name="c3",
        sku="sha1:" + "9" * 40,
        unit="seconds",
        quantity=3600,
    )
    session.add_all(
        [
            # Last month, at 2 units — outside the queried day.
            _mu(bucket_start=BUCKET - timedelta(days=40), sku_count=2, **common),
            _mu(sku_count=4, **common),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
        ),
        metric_keys=["unit_hours"],
    )
    row = next(i for i in out["items"] if i["key"] == "c3")
    # Positive assertions, not "shapes is absent" — that would also hold if the
    # feature were removed entirely (it did, silently, once). Exactly ONE shape
    # is in range, and it is the in-range one (count 4), not the old count 2.
    assert len(row["shapes"]) == 1
    assert row["shapes"][0]["sku_count"] == 4
    assert row["shapes"][0]["instance_hours"] == 1.0
    # And the out-of-range hour is not folded into the row's totals either.
    assert row["metrics"]["unit_hours"] == 4.0


@pytest.mark.asyncio
async def test_instance_type_rows_are_identifiable_not_just_labelled(session):
    """``sku`` became an opaque hash, so the row LABEL is the snapshotted type
    name — and a label is not an identity: the same definition on two clusters is
    two priced rows sharing one name. Each row must therefore carry the full
    ``(sku, sku_count)`` key plus ``definition_snapshot``, or a client keying on
    what it can see collides."""
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        quantity=3600,
        unit="seconds",
        instance_type_name="gpustack--generic--nvidia-a100-linux-amd64",
        definition_snapshot="sha1:" + "d" * 40,
        dimensions={"gpu_count": 1},
    )
    session.add_all(
        [
            # Same definition, two clusters -> two skus, ONE name.
            _mu(resource_id=901, resource_name="a", sku="sha1:" + "1" * 40, **common),
            _mu(resource_id=902, resource_name="b", sku="sha1:" + "2" * 40, **common),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance_type"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["instance_hours"],
    )
    items = [i for i in out["items"] if str(i.get("sku", "")).startswith("sha1:")]
    assert len(items) == 2
    # Readable label, identical on both — exactly why it cannot be the key.
    assert {i["key"] for i in items} == {"gpustack--generic--nvidia-a100-linux-amd64"}
    # The real identity is distinct, and both halves of it are exposed.
    assert len({i["sku"] for i in items}) == 2
    assert all(i["sku_count"] == 1 for i in items)
    # Cross-cluster folding stays possible.
    assert {i["definition_snapshot"] for i in items} == {"sha1:" + "d" * 40}


@pytest.mark.asyncio
async def test_instance_type_shapes_share_a_name_but_not_a_key(session):
    """Within ONE cluster, a whole-card row and a sliced row of the same type
    also share the label — ``sku_count`` is what separates them."""
    from sqlalchemy import and_

    sku = "sha1:" + "3" * 40
    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        sku=sku,
        quantity=3600,
        unit="seconds",
        instance_type_name="a100-pool",
    )
    session.add_all(
        [
            _mu(
                resource_id=911,
                resource_name="whole",
                sku_count=Decimal("1"),
                dimensions={"gpu_count": 1, "slice_mode": "whole"},
                **common,
            ),
            _mu(
                resource_id=912,
                resource_name="sliced",
                sku_count=Decimal("0.25"),
                dimensions={
                    "gpu_count": 1,
                    "slice_mode": "ratio",
                    "sliced_memory_percentage": 25,
                    "slice_share_milli": 250,
                },
                **common,
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance_type"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["instance_hours"],
    )
    items = [i for i in out["items"] if i.get("sku") == sku]
    assert len(items) == 2
    assert {i["key"] for i in items} == {"a100-pool"}  # same label
    # Compared numerically, not as text: the column has scale 8, so the value
    # reads back as Decimal("0.25000000"). That is the same shape as 0.25 — the
    # very reason the shape lives in a NUMERIC column instead of a hashed string.
    ordered = sorted(items, key=lambda i: i["sku_count"])
    assert ordered[0]["sku_count"] == Decimal("0.25")
    assert ordered[1]["sku_count"] == 1
    # The sliced row's facets travel, so the page can show WHY it costs less.
    assert ordered[0]["dimensions"]["slice_mode"] == "ratio"
    assert ordered[0]["dimensions"]["slice_share_milli"] == 250


@pytest.mark.asyncio
async def test_gpu_instances_breakdown_by_instance_type(session):
    from gpustack.schemas.metered_usage import METER_INSTANCE_UPTIME as UPTIME
    from sqlalchemy import and_

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance_type"),
        base_filter=and_(
            MeteredUsage.meter_key == UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours", "instance_hours"],
    )
    by_key = {i["key"]: i["metrics"] for i in out["items"]}
    assert set(by_key) == {"h100x2", "a100x1"}
    assert by_key["h100x2"]["gpu_hours"] == pytest.approx(49795 * 2 / 3600, abs=0.01)
    assert by_key["a100x1"]["gpu_hours"] == pytest.approx(3600 / 3600, abs=0.01)


@pytest.mark.asyncio
async def test_gpu_instances_breakdown_includes_cpu_instances(session):
    # CPU-only instances are metered as ``cpu_instance`` — the GPU Instances
    # tab must show them too (contributing instance_hours but 0 gpu_hours).
    from gpustack.routes.resource_usage import gpu_instances_breakdown
    from gpustack.schemas.metered_usage import RESOURCE_TYPE_CPU_INSTANCE

    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_CPU_INSTANCE,
            resource_id=601,
            resource_name="cpu-1",
            sku="gpustack--generic-31-ln-a64-1c-2g",
            sku_count=1,
            quantity=7200,
            unit="seconds",
            # A 4c8g request on a 1c2g flavor — the breakdown must surface the
            # totals (cpu_milli/memory_mib), not only the per-unit specs.
            dimensions={
                "gpu_count": 0,
                "cpu_milli": 4000,
                "memory_mib": 8192,
                "unit_cpu_milli": 1000,
                "unit_memory_mib": 2048,
            },
        )
    )
    await session.commit()

    out = await gpu_instances_breakdown(session, USER, CTX, _req("instance"))
    items_by_name = {i["key"]: i for i in out["items"]}
    by_name = {k: i["metrics"] for k, i in items_by_name.items()}
    assert set(by_name) == {"gpu-1", "gpu-2", "cpu-1"}
    assert by_name["cpu-1"]["instance_hours"] == pytest.approx(2.0, abs=0.01)
    # CPU rows never leak into GPU-Hours (sku_count=1 is the whole machine).
    assert by_name["cpu-1"]["gpu_hours"] == 0
    cpu_dims = items_by_name["cpu-1"]["dimensions"]
    assert cpu_dims["cpu_milli"] == 4000
    assert cpu_dims["memory_mib"] == 8192
    assert cpu_dims["unit_cpu_milli"] == 1000
    assert cpu_dims["unit_memory_mib"] == 2048
    assert out["summary"]["instance_hours"] == pytest.approx(
        (49795 + 3600 + 7200) / 3600, abs=0.01
    )
    assert out["summary"]["gpu_hours"] == pytest.approx(
        (49795 * 2 + 3600) / 3600, abs=0.01
    )


@pytest.mark.asyncio
async def test_storage_breakdown_by_volume(session):
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("volume"),
        base_filter=(MeteredUsage.meter_key == METER_STORAGE_CAPACITY),
        metric_keys=["gb_days"],
    )
    assert len(out["items"]) == 1
    item = out["items"][0]
    assert item["key"] == "pv-models"
    assert item["id"] == 88
    assert item["metrics"]["gb_days"] == pytest.approx(
        204800 * 25200 / 1024 / 86400, abs=0.01
    )


@pytest.mark.asyncio
async def test_breakdown_filters_by_creator_ids(session):
    # A row from a different creator (id 9). Use scope="all" so the self-clamp
    # doesn't already hide it, isolating the creator_ids filter.
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=503,
            resource_name="gpu-3",
            sku="a100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            creator_id=9,
        )
    )
    # gpu-3 is a live instance too, so the count it would add is real — the
    # creator filter is what must exclude it.
    session.add(_gi(503, "gpu-3"))
    await session.commit()

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=D,
        end_date=D,
        group_by=["resource_type"],
        creator_ids=[7],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=None,
        metric_keys=["instance_hours", "gpu_hours", "gb_days"],
    )
    gpu = {i["key"]: i["metrics"] for i in out["items"]}[RESOURCE_TYPE_GPU_INSTANCE]
    # creator 9's gpu-3 is filtered out; of the creator-7 instances only the
    # live one (gpu-2 / 502) counts — gpu-1 (501) is since-deleted.
    assert gpu["resources"] == 1


@pytest.mark.asyncio
async def test_breakdown_tenant_scope_follows_consumer_not_owner(session):
    # Shared-cluster row: provider (owner) = org 99, consumer = org 1.
    # A tenant scoped to org 1 must SEE it (they pay), even though they don't own
    # the cluster. A row consumed by org 2 must be hidden from org 1.
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=601,
            resource_name="shared-gpu",
            sku="h100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            owner_principal_id=99,  # cluster provider
            consumer_principal_id=1,  # consumer = org 1
            creator_id=7,
        )
    )
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=602,
            resource_name="other-gpu",
            sku="h100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            owner_principal_id=99,
            consumer_principal_id=2,  # paid by a different org
            creator_id=7,
        )
    )
    await session.commit()

    ctx_org1 = SimpleNamespace(current_principal_id=1)
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=ctx_org1,
        request=ResourceBreakdownRequest(
            scope="all", start_date=D, end_date=D, group_by=["instance"]
        ),
        base_filter=(MeteredUsage.meter_key == METER_INSTANCE_UPTIME),
        metric_keys=["gpu_hours"],
    )
    names = {i["key"] for i in out["items"]}
    # org 1 sees the row it pays for (consumer=1) + the seeded self-owned rows
    # (consumer NULL is not == 1, so only shared-gpu among the new rows);
    # crucially org 2's row is excluded.
    assert "shared-gpu" in names
    assert "other-gpu" not in names


@pytest.mark.asyncio
async def test_breakdown_filters_by_instance_ids(session):
    from sqlalchemy import and_

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=D,
        end_date=D,
        group_by=["instance"],
        instance_ids=[501],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    # only the selected instance 501 (gpu-1) survives
    assert [i["id"] for i in out["items"]] == [501]


@pytest.mark.asyncio
async def test_resource_meta_lists_creators_instances_volumes(session):
    from gpustack.routes.resource_usage import resource_meta

    out = await resource_meta(session, USER, CTX, scope="all")
    labels = {c["id"]: c["label"] for c in out["creators"]}
    # creator_id 7 → principal login name "alice" (not display name)
    assert labels.get(7) == "alice"
    # instances / volumes resolve their snapshot names
    assert {i["id"]: i["label"] for i in out["instances"]} == {
        501: "gpu-1",
        502: "gpu-2",
    }
    assert {v["id"]: v["label"] for v in out["volumes"]} == {88: "pv-models"}


@pytest.mark.asyncio
async def test_resource_meta_deleted_creator_falls_back_to_snapshot_name(session):
    """A creator whose principal row is gone shows the ``creator_name`` login
    snapshot (marked deleted), not a bare ``User <id>``."""
    from gpustack.routes.resource_usage import resource_meta

    # creator_id 99 has no Principal row; its metered_usage rows carry the
    # login-name snapshot captured at event time.
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=777,
            resource_name="gpu-x",
            creator_id=99,
            creator_name="bob",
            sku="a100x1",
            quantity=3600,
            unit="seconds",
        )
    )
    await session.commit()

    out = await resource_meta(session, USER, CTX, scope="all")
    creators = {c["id"]: c for c in out["creators"]}
    assert creators[99]["label"] == "bob"
    assert creators[99]["deleted"] is True
    # the live creator (7) is not flagged deleted
    assert creators[7]["deleted"] is False


@pytest.mark.asyncio
async def test_resource_meta_instances_include_cpu_instances(session):
    from gpustack.routes.resource_usage import resource_meta
    from gpustack.schemas.metered_usage import RESOURCE_TYPE_CPU_INSTANCE

    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_CPU_INSTANCE,
            resource_id=601,
            resource_name="cpu-1",
            sku="gpustack--generic-31-ln-a64-1c-2g",
            quantity=7200,
            unit="seconds",
        )
    )
    await session.commit()

    out = await resource_meta(session, USER, CTX, scope="all")
    assert {i["id"]: i["label"] for i in out["instances"]} == {
        501: "gpu-1",
        502: "gpu-2",
        601: "cpu-1",
    }


@pytest.mark.asyncio
async def test_summary_unions_tokens_and_metered(session):
    session.add(
        ModelUsage(
            user_id=7,
            consumer_principal_id=None,
            model_name="qwen",
            date=D,
            prompt_token_count=1_000_000,
            completion_token_count=250_000,
            prompt_cached_token_count=0,
            request_count=10,
        )
    )
    await session.commit()
    out = await usage_summary(
        session, USER, CTX, start_date=D, end_date=D, scope="self"
    )
    assert out["total_tokens"] == 1_250_000
    assert out["input_tokens"] == 1_000_000
    assert out["output_tokens"] == 250_000
    assert out["token_active_users"] == 1
    assert out["gpu_hours"] == pytest.approx((49795 * 2 + 3600) / 3600, abs=0.01)
    assert out["storage_gb_days"] == pytest.approx(
        204800 * 25200 / 1024 / 86400, abs=0.01
    )


def test_phase_message_of():
    # camelCase (model_dump by alias) and snake_case (by field name) both read
    assert _phase_message_of({"status": {"phaseMessage": "boom"}}) == "boom"
    assert _phase_message_of({"status": {"phase_message": "kaboom"}}) == "kaboom"
    # raw JSON string (some drivers / replay paths) is parsed defensively
    assert _phase_message_of('{"status": {"phaseMessage": "oops"}}') == "oops"
    # missing / malformed → None (no crash)
    assert _phase_message_of({"status": {}}) is None
    assert _phase_message_of({"status": "not-a-dict"}) is None
    assert _phase_message_of(None) is None
    assert _phase_message_of("not json") is None


@pytest.mark.asyncio
async def test_resource_events_filters_by_event_type_and_name(session):
    from datetime import datetime
    from gpustack.routes.resource_usage import resource_events
    from gpustack.schemas.resource_events import (
        EVENT_TYPE_PHASE_LEFT_METERED,
        EVENT_TYPE_PHASE_TO_METERED,
        ResourceEvent,
    )

    at = datetime(2026, 5, 26, 10, 0, 0)
    session.add_all(
        [
            ResourceEvent(
                occurred_at=at,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=701,
                resource_name="michelia-gpu",
                event_type=EVENT_TYPE_PHASE_TO_METERED,
                creator_id=7,
            ),
            ResourceEvent(
                occurred_at=at,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=701,
                resource_name="michelia-gpu",
                event_type=EVENT_TYPE_PHASE_LEFT_METERED,
                creator_id=7,
            ),
            ResourceEvent(
                occurred_at=at,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=702,
                resource_name="other-box",
                event_type=EVENT_TYPE_PHASE_TO_METERED,
                creator_id=7,
            ),
        ]
    )
    await session.commit()

    # event_types (CSV) → only the Stopped (phase_left_metered) row.
    out = await resource_events(
        session,
        USER,
        CTX,
        scope="all",
        event_types=EVENT_TYPE_PHASE_LEFT_METERED,
    )
    assert [e["event_type"] for e in out["items"]] == [EVENT_TYPE_PHASE_LEFT_METERED]

    # resource_name → case-insensitive substring (fuzzy) match.
    out = await resource_events(
        session,
        USER,
        CTX,
        scope="all",
        resource_name="MICHELIA",
    )
    assert {e["resource_name"] for e in out["items"]} == {"michelia-gpu"}
    assert len(out["items"]) == 2


@pytest.mark.asyncio
async def test_breakdown_date_sub_group_splits_series(session):
    # group_by="date" + sub_group_by="instance_type" → one (date, sku) row per
    # bucket per group, so the trend chart can build one series per instance type.
    from sqlalchemy import and_

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=D,
        end_date=D,
        group_by=["date", "instance_type"],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    # every compound row carries BOTH the date bucket and the group key
    assert out["items"]
    assert all(
        i.get("date") is not None and i.get("key") is not None for i in out["items"]
    )
    by_sku = {i["key"]: i["metrics"]["gpu_hours"] for i in out["items"]}
    assert by_sku["h100x2"] == pytest.approx(49795 * 2 / 3600, abs=0.01)
    assert by_sku["a100x1"] == pytest.approx(3600 / 3600, abs=0.01)


@pytest.mark.asyncio
async def test_buckets_and_last_active_use_rollup_tz(session, monkeypatch):
    # Pin the rollup tz to +08:00 (no DST). A 20:00 UTC bucket is 04:00 the
    # NEXT day there → it must bucket on 05-27, and Last Active reads 05-27.
    from gpustack import envs
    from sqlalchemy import and_

    monkeypatch.setattr(envs, "TIMEZONE", "Asia/Shanghai")

    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=901,
            resource_name="tz-gpu",
            sku="tz-sku",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            bucket_start=datetime(2026, 5, 26, 20, 0, 0),
        )
    )
    await session.commit()

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=date(2026, 5, 26),
        end_date=date(2026, 5, 27),
        group_by=["date", "instance_type"],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    tz_rows = [i for i in out["items"] if i.get("key") == "tz-sku"]
    assert tz_rows, "tz-sku row missing"
    # Bucketed on the rollup-tz day (05-27), not UTC's 05-26.
    assert str(tz_rows[0]["date"]).startswith("2026-05-27")
    # Last Active is the same instant rendered in the rollup tz (04:00 on 05-27),
    # and aware (carries the +08:00 offset) so the API is self-describing.
    last_active = tz_rows[0]["metrics"]["last_active"]
    assert last_active.utcoffset() == timedelta(hours=8)
    assert str(last_active).startswith("2026-05-27 04:00:00+08:00")

    # Hour granularity → the bucket itself is an aware, offset-carrying instant.
    hour_req = ResourceBreakdownRequest(
        scope="all",
        start_date=date(2026, 5, 26),
        end_date=date(2026, 5, 27),
        group_by=["date"],
        granularity="hour",
    )
    hour_out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=hour_req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    hour_dates = {str(i["date"]) for i in hour_out["items"]}
    assert "2026-05-27 04:00:00+08:00" in hour_dates


@pytest.mark.asyncio
async def test_resource_events_filter_uses_rollup_tz_day(session, monkeypatch):
    # #5523 regression: an event at 2026-05-26 20:00 UTC is 2026-05-27 04:00 in
    # +08:00, so it must match a 05-27 filter and NOT a 05-26 one — matching how
    # occurred_at is displayed (the bug was filtering on the raw UTC day).
    from gpustack import envs
    from gpustack.routes.resource_usage import resource_events
    from gpustack.schemas.resource_events import EVENT_TYPE_CREATED, ResourceEvent

    monkeypatch.setattr(envs, "TIMEZONE", "Asia/Shanghai")
    session.add(
        ResourceEvent(
            occurred_at=datetime(2026, 5, 26, 20, 0, 0),
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=801,
            resource_name="tz-evt",
            event_type=EVENT_TYPE_CREATED,
            creator_id=7,
        )
    )
    await session.commit()

    def names(out):
        return {e["resource_name"] for e in out["items"]}

    # The rollup-tz day (05-27) finds it, displayed at 04:00+08:00.
    out = await resource_events(
        session,
        USER,
        CTX,
        scope="all",
        start_date=date(2026, 5, 27),
        end_date=date(2026, 5, 27),
        resource_name="tz-evt",
    )
    assert names(out) == {"tz-evt"}
    occurred = out["items"][0]["occurred_at"]
    assert occurred.utcoffset() == timedelta(hours=8)
    assert str(occurred).startswith("2026-05-27 04:00:00+08:00")

    # The raw UTC day (05-26) must NOT return it anymore.
    out = await resource_events(
        session,
        USER,
        CTX,
        scope="all",
        start_date=date(2026, 5, 26),
        end_date=date(2026, 5, 26),
        resource_name="tz-evt",
    )
    assert names(out) == set()


@pytest.mark.asyncio
async def test_buckets_use_negative_offset_rollup_tz(session, monkeypatch):
    # Negative-offset zone (America/Bogota, UTC-05:00, no DST). A 02:00 UTC
    # bucket is 21:00 the PREVIOUS day there → it must bucket on 05-25 and Last
    # Active carries the -05:00 offset.
    from gpustack import envs
    from sqlalchemy import and_

    monkeypatch.setattr(envs, "TIMEZONE", "America/Bogota")
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=902,
            resource_name="neg-gpu",
            sku="neg-sku",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            bucket_start=datetime(2026, 5, 26, 2, 0, 0),
        )
    )
    await session.commit()

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=date(2026, 5, 25),
        end_date=date(2026, 5, 26),
        group_by=["date", "instance_type"],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    rows = [i for i in out["items"] if i.get("key") == "neg-sku"]
    assert rows, "neg-sku row missing"
    assert str(rows[0]["date"]).startswith("2026-05-25")
    last_active = rows[0]["metrics"]["last_active"]
    assert last_active.utcoffset() == timedelta(hours=-5)
    assert str(last_active).startswith("2026-05-25 21:00:00-05:00")


@pytest.mark.asyncio
async def test_breakdown_range_boundary_includes_shifted_edge(session, monkeypatch):
    # The row sits at exactly start_day − offset: 2026-05-26 16:00 UTC is
    # 2026-05-27 00:00 in +08:00, the first instant of the selected start day.
    # The half-open shifted window must include it (off-by-one guard).
    from gpustack import envs
    from sqlalchemy import and_

    monkeypatch.setattr(envs, "TIMEZONE", "Asia/Shanghai")
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=903,
            resource_name="edge-gpu",
            sku="edge-sku",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            bucket_start=datetime(2026, 5, 26, 16, 0, 0),
        )
    )
    await session.commit()

    req = ResourceBreakdownRequest(
        scope="all",
        start_date=date(2026, 5, 27),  # selecting only 05-27 (rollup tz)
        end_date=date(2026, 5, 27),
        group_by=["instance_type"],
    )
    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=req,
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_GPU_INSTANCE,
        ),
        metric_keys=["gpu_hours"],
    )
    assert "edge-sku" in {i.get("key") for i in out["items"]}


@pytest.mark.asyncio
async def test_breakdown_groups_by_organization_for_admin(session):
    # Two consumer Orgs; the consumer principal id is resolved to a display
    # name live, and a gone principal (no Principal row) is flagged deleted.
    from gpustack.schemas.principals import Principal

    session.add(Principal(id=100, kind="org", name="acme", display_name="Acme"))
    # org 200 intentionally left unseeded → since-deleted.
    session.add_all(
        [
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=701,
                resource_name="org-a-gpu",
                sku="h100x1",
                sku_count=1,
                quantity=3600,
                unit="seconds",
                consumer_principal_id=100,
                creator_id=7,
            ),
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=702,
                resource_name="org-b-gpu",
                sku="h100x1",
                sku_count=1,
                quantity=3600,
                unit="seconds",
                consumer_principal_id=200,
                creator_id=7,
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,  # admin, cross-org (current_principal_id=None)
        request=ResourceBreakdownRequest(
            scope="all", start_date=D, end_date=D, group_by=["organization"]
        ),
        base_filter=(MeteredUsage.meter_key == METER_INSTANCE_UPTIME),
        metric_keys=["gpu_hours"],
    )
    by_id = {i["id"]: i for i in out["items"]}
    assert by_id[100]["key"] == "acme"
    assert by_id[100]["deleted"] is False
    assert by_id[200]["deleted"] is True


@pytest.mark.asyncio
async def test_breakdown_filters_by_organization(session):
    session.add_all(
        [
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=711,
                resource_name="org-a-gpu",
                sku="h100x1",
                sku_count=1,
                quantity=3600,
                unit="seconds",
                consumer_principal_id=100,
                creator_id=7,
            ),
            _mu(
                meter_key=METER_INSTANCE_UPTIME,
                resource_type=RESOURCE_TYPE_GPU_INSTANCE,
                resource_id=712,
                resource_name="org-b-gpu",
                sku="h100x1",
                sku_count=1,
                quantity=3600,
                unit="seconds",
                consumer_principal_id=200,
                creator_id=7,
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(
            scope="all",
            start_date=D,
            end_date=D,
            group_by=["instance"],
            organization_ids=[100],
        ),
        base_filter=(MeteredUsage.meter_key == METER_INSTANCE_UPTIME),
        metric_keys=["gpu_hours"],
    )
    names = {i["key"] for i in out["items"]}
    assert "org-a-gpu" in names
    assert "org-b-gpu" not in names


@pytest.mark.asyncio
async def test_breakdown_filters_by_user_group_members(session):
    # Group 50 has one direct user member (creator 7); a row from creator 9
    # must be excluded when filtering by that group.
    from gpustack.schemas.principals import Principal, PrincipalMembership

    session.add(Principal(id=50, kind="group", name="eng", display_name="Eng"))
    session.add(Principal(id=9, kind="user", name="bob", display_name="Bob"))
    session.add(PrincipalMembership(parent_principal_id=50, member_principal_id=7))
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=721,
            resource_name="bob-gpu",
            sku="h100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            creator_id=9,
        )
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(
            scope="all",
            start_date=D,
            end_date=D,
            group_by=["user"],
            user_group_ids=[50],
        ),
        base_filter=(MeteredUsage.meter_key == METER_INSTANCE_UPTIME),
        metric_keys=["gpu_hours"],
    )
    creator_ids = {i["id"] for i in out["items"]}
    assert creator_ids == {7}  # only the group member; bob (9) excluded


@pytest.mark.asyncio
async def test_breakdown_rejects_organization_group_by_for_non_admin(session):
    regular = SimpleNamespace(id=7, is_admin=False)
    with pytest.raises(ForbiddenException):
        await _run_breakdown(
            session,
            user=regular,
            ctx=SimpleNamespace(
                current_principal_id=1,
                org_role=None,
                current_is_personal_scope=False,
            ),
            request=ResourceBreakdownRequest(
                scope="all", start_date=D, end_date=D, group_by=["organization"]
            ),
            base_filter=None,
            metric_keys=["gpu_hours"],
        )


@pytest.mark.asyncio
async def test_breakdown_rejects_user_group_filter_in_self_scope(session):
    regular = SimpleNamespace(id=7, is_admin=False)
    with pytest.raises(ForbiddenException):
        await _run_breakdown(
            session,
            user=regular,
            ctx=SimpleNamespace(
                current_principal_id=1,
                org_role=None,
                current_is_personal_scope=False,
            ),
            request=ResourceBreakdownRequest(
                scope="self",
                start_date=D,
                end_date=D,
                group_by=["user"],
                user_group_ids=[50],
            ),
            base_filter=None,
            metric_keys=["gpu_hours"],
        )


@pytest.mark.asyncio
async def test_resource_meta_lists_organizations_and_user_groups(session):
    from gpustack.routes.resource_usage import resource_meta
    from gpustack.schemas.principals import Principal

    session.add(Principal(id=100, kind="org", name="acme", display_name="Acme"))
    session.add(Principal(id=50, kind="group", name="eng", display_name="Eng"))
    session.add(
        Principal(id=51, kind="group", name="system/authenticated", display_name="All")
    )
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=731,
            resource_name="org-a-gpu",
            sku="h100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            consumer_principal_id=100,
            creator_id=7,
        )
    )
    await session.commit()

    out = await resource_meta(session, USER, CTX, scope="all")
    assert [g["label"] for g in out["user_groups"]] == ["Eng"]  # reserved hidden
    assert [o["id"] for o in out["organizations"]] == [100]
    assert out["organizations"][0]["label"] == "acme"


@pytest.mark.asyncio
async def test_resource_meta_hides_org_and_group_when_pinned_to_org(session):
    # An admin pinned to a single Org (current_principal_id set) is NOT
    # platform-wide: both the org and the user-group filter sources are empty
    # (cross-Org filtering is meaningless within one tenant).
    from gpustack.routes.resource_usage import resource_meta
    from gpustack.schemas.principals import Principal

    session.add(Principal(id=100, kind="org", name="acme", display_name="Acme"))
    session.add(Principal(id=50, kind="group", name="eng", display_name="Eng"))
    session.add(
        _mu(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=741,
            resource_name="org-a-gpu",
            sku="h100x1",
            sku_count=1,
            quantity=3600,
            unit="seconds",
            consumer_principal_id=100,
            creator_id=7,
        )
    )
    await session.commit()

    pinned_ctx = SimpleNamespace(current_principal_id=100)
    out = await resource_meta(session, USER, pinned_ctx, scope="all")
    assert out["organizations"] == []
    assert out["user_groups"] == []


@pytest.mark.asyncio
async def test_row_identity_and_specs_describe_the_same_shape(session):
    """A per-instance row's type identity must come from the shape its
    ``dimensions`` describe.

    Both used to be aggregated, but differently: ``dimensions`` from the row with
    the greatest ``id`` (the LATEST shape) and ``sku`` / ``instance_type_name``
    from ``func.max`` over the column (the lexicographic maximum, which for a hash
    is an arbitrary member). A reconfigured instance is where they part company —
    the row below then named the old CPU type while describing 4 A100s. Callers
    have no way to notice, since both fields look authoritative.
    """
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        resource_id=961,
        resource_name="switched",
        unit="seconds",
        quantity=3600,
    )
    session.add_all(
        [
            # First the CPU shape, whose sku sorts HIGHER than the GPU one that
            # replaced it — so ``func.max`` picks the superseded row.
            _mu(
                sku="sha1:" + "f" * 40,
                instance_type_name="zz-cpu-2c4g",
                definition_snapshot="sha1:" + "c" * 40,
                sku_count=2,
                dimensions={"cpu_milli": 2000, "memory_mib": 4096, "gpu_count": 0},
                **common,
            ),
            _mu(
                bucket_start=BUCKET + timedelta(hours=1),
                sku="sha1:" + "a" * 40,
                instance_type_name="aa-gpu-a100",
                definition_snapshot="sha1:" + "g" * 40,
                sku_count=4,
                dimensions={"gpu_count": 4, "gpu_type": "nvidia-a100"},
                **{k: v for k, v in common.items() if k != "bucket_start"},
            ),
        ]
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=and_(
            MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
            MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
        ),
        metric_keys=["unit_hours"],
    )
    row = next(i for i in out["items"] if i["key"] == "switched")

    # The latest shape, consistently: label, identity and specs all agree.
    assert row["dimensions"]["gpu_count"] == 4
    assert row["instance_type_name"] == "aa-gpu-a100"
    assert row["sku"] == "sha1:" + "a" * 40
    assert row["definition_snapshot"] == "sha1:" + "g" * 40
    # Nothing is lost by picking one: the full history is what ``shapes`` is for.
    assert [s["sku_count"] for s in row["shapes"]] == [2, 4]


@pytest.mark.asyncio
async def test_bucketed_rows_carry_no_shape_breakdown(session):
    """``["date", "instance"]`` rows cover ONE bucket, so a whole-window shape
    breakdown does not describe them.

    ``shapes`` is aggregated over the request's date range. On a per-day row that
    made the segments sum to the instance's whole-range hours while the row's own
    metrics covered a single day — a 30-day query put ~720 hours of segments under
    a 24-hour row. The scoping the ``window`` argument adds one level up does not
    reach the date axis, so the field is simply not emitted there.
    """
    from sqlalchemy import and_

    common = dict(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_CPU_INSTANCE,
        resource_id=971,
        resource_name="daily",
        sku="sha1:" + "7" * 40,
        unit="seconds",
        quantity=3600,
    )
    session.add_all(
        [
            _mu(sku_count=2, **common),
            _mu(
                bucket_start=BUCKET + timedelta(days=1),
                sku_count=4,
                **{k: v for k, v in common.items() if k != "bucket_start"},
            ),
        ]
    )
    await session.commit()

    base = and_(
        MeteredUsage.meter_key == METER_INSTANCE_UPTIME,
        MeteredUsage.resource_type == RESOURCE_TYPE_CPU_INSTANCE,
    )
    bucketed = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(
            scope="self",
            start_date=D,
            end_date=D + timedelta(days=1),
            group_by=["date", "instance"],
        ),
        base_filter=base,
        metric_keys=["unit_hours"],
    )
    rows = [i for i in bucketed["items"] if i["key"] == "daily"]
    assert len(rows) == 2
    assert all("shapes" not in r for r in rows)
    # The enrichment a bucketed row DOES need still happens — this is a skip of
    # one field, not of the branch.
    assert all(r["dimensions"] is not None for r in rows)

    # Without the date axis the same request does carry the breakdown.
    whole = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=ResourceBreakdownRequest(
            scope="self",
            start_date=D,
            end_date=D + timedelta(days=1),
            group_by=["instance"],
        ),
        base_filter=base,
        metric_keys=["unit_hours"],
    )
    row = next(i for i in whole["items"] if i["key"] == "daily")
    assert [s["sku_count"] for s in row["shapes"]] == [2, 4]


@pytest.mark.asyncio
async def test_a_storage_row_keeps_its_own_sku_under_the_generic_endpoint(session):
    """``/resource/breakdown`` has no ``base_filter``, and this grouping keys on
    ``resource_id`` alone — so a volume lands in the "instance" branch too.

    The realignment above must not touch it. It looks up an uptime representative,
    which a volume has none of, so overwriting unconditionally replaced a perfectly
    readable ``volume--<kind>--<type>`` sku with ``None``. For a volume the
    aggregate MAX is exact (one sku per volume), so the untouched value is right.
    """
    session.add(
        _mu(
            meter_key=METER_STORAGE_CAPACITY,
            resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
            resource_id=981,
            resource_name="pv-mixed",
            sku="volume--nfs--aws",
            quantity=204800 * 3600,
            unit="mib_seconds",
        )
    )
    await session.commit()

    out = await _run_breakdown(
        session,
        user=USER,
        ctx=CTX,
        request=_req("instance"),
        base_filter=None,  # what the generic endpoint passes
        metric_keys=["unit_hours", "instance_hours", "gpu_hours", "gb_days"],
    )
    by_name = {i["key"]: i for i in out["items"]}
    assert by_name["pv-mixed"]["sku"] == "volume--nfs--aws"
    # A real instance in the same response is still realigned off its
    # representative — the guard narrows the overwrite, it does not remove it.
    assert by_name["gpu-1"]["sku"] == "h100x2"
