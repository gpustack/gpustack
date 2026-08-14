"""Metering against the instance-type catalog: ``sku = type_snapshot``,
fractional ``sku_count`` for sliced accelerators, and the per-shape natural key.

Three things are pinned here because getting any of them wrong is a billing bug
that no test elsewhere would catch:

* ``sku`` is the type snapshot BYTE-FOR-BYTE — not a derived or prefixed string.
* A sliced card bills its VRAM share, and an unresolvable share is NOT settled
  (never silently rounded up to a whole card).
* Two shapes inside one hour become two rows; a shape-neutral change does not.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from fnmatch import fnmatch
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import gpustack.server.resource_usage_collector as rc
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile,
    GPUInstanceTypeDetail,
    GPUInstanceTypeSpec,
    GPUInstanceTypeStatusPublic,
    GPUInstanceTypeUnitResources,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    MeteredUsage,
)
from gpustack.schemas.resource_events import (
    EVENT_TYPE_PHASE_TO_METERED,
    RESOURCE_TYPE_GPU_INSTANCE,
    ResourceEvent,
)
from gpustack.server.resource_usage_collector import (
    SKU_SOURCE_DESCRIPTION,
    SKU_SOURCE_TYPE_SNAPSHOT,
    ResourceUsageCollector,
    _open_window_from_event,
    _resolve_instance_type,
)
from gpustack.utils.resource_usage import volume_sku

A100_VRAM_MIB = 81920
# Module-level so it is not a call in a default argument (flake8 B008).
_T0 = datetime(2026, 8, 5, 10, 0, 0)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_type(
    *,
    cluster_id: int = 1,
    name: str = "gpustack--generic--nvidia-a100-linux-amd64",
    accelerator_group: str = "nvidia-a100",
    display_name: str | None = None,
    acceleratable: bool = True,
    unit_cpu: str = "4000m",
    unit_ram: str = "16384Mi",
    with_detail: bool = True,
    card_memory_mib: int = A100_VRAM_MIB,
    profiles: list[tuple[str, int]] | None = None,
    deleted_at: datetime | None = None,
) -> GPUInstanceType:
    spec = GPUInstanceTypeSpec(
        display_name=display_name,
        accelerator_group=accelerator_group,
        general_group="generic",
        acceleratable=acceleratable,
        os="linux",
        arch="amd64",
        unit_resources=GPUInstanceTypeUnitResources(cpu=unit_cpu, ram=unit_ram),
        local_storage="100Gi",
    )
    status = None
    if with_detail:
        sliced = None
        if profiles is not None:
            sliced = GPUInstanceTypeAcceleratorSlicedDetail(
                physical=GPUInstanceTypeAcceleratorSlicedPhysicalDetail(
                    profiles=[
                        GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile(
                            name=n, memory_mib=m, count=1
                        )
                        for n, m in profiles
                    ],
                )
            )
        status = GPUInstanceTypeStatusPublic(
            detail=GPUInstanceTypeDetail(
                manufacturer="nvidia",
                product="NVIDIA-A100-SXM4-80GB",
                family="ampere",
                memory=f"{card_memory_mib}Mi",
                sliced_detail=sliced,
            )
        )
    row = GPUInstanceType(cluster_id=cluster_id, name=name, spec=spec, status=status)
    row.snapshot = row.compute_snapshot()
    row.definition_snapshot = row.compute_definition_snapshot()
    row.deleted_at = deleted_at
    return row


def make_event(
    *,
    type_snapshot: str | None,
    resources: dict,
    occurred_at: datetime = _T0,
    resource_id: int = 1,
    type_name: str = "gpustack--generic--nvidia-a100-linux-amd64",
    description: str | None = None,
) -> ResourceEvent:
    snap: dict = {
        "name": f"gpu-{resource_id}",
        "display_name": "GPU 1",
        "type_snapshot": type_snapshot,
        "spec": {"type_": type_name, "resources": resources},
    }
    if description is not None:
        snap["description"] = description
    return ResourceEvent(
        occurred_at=occurred_at,
        owner_principal_id=42,
        owner_name="acme-org",
        creator_id=7,
        creator_name="bob",
        cluster_id=1,
        cluster_name="default",
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=resource_id,
        resource_name=f"gpu-{resource_id}",
        event_type=EVENT_TYPE_PHASE_TO_METERED,
        phase="Ready",
        spec_snapshot=snap,
    )


@pytest_asyncio.fixture
async def catalog_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
        await conn.run_sync(MeteredUsage.__table__.create)
    # ``expire_on_commit=False``: these tests read the rows they just committed
    # (e.g. ``row.snapshot``), and an expired attribute would trigger a lazy
    # reload from a sync context, which async SQLAlchemy cannot service.
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


@asynccontextmanager
async def _yield(session):
    yield session


# ---------------------------------------------------------------------------
# sku identity
# ---------------------------------------------------------------------------


def test_sku_is_the_type_snapshot_verbatim():
    """No prefix, no truncation, no transformation — the sku must be joinable
    against ``gpu_instance_types.snapshot`` with a plain string equality."""
    row = make_type()
    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "2"})
    w = _open_window_from_event(evt)
    assert w is not None
    assert w.sku == row.snapshot
    assert w.sku.startswith("sha1:")
    assert len(w.sku) == len("sha1:") + 40
    # No source tag: the authoritative key was available.
    assert "sku_source" not in w.dimensions


def test_display_name_edit_does_not_change_the_sku():
    """``display_name`` is the one mutable spec field, so it is excluded from the
    snapshot — renaming a pool must not re-sku (and re-price) it."""
    assert make_type(display_name="A100 Pool").snapshot == make_type().snapshot


def test_recreated_type_keeps_or_changes_the_sku_by_definition():
    # Same definition recreated -> same sku (sha1 is a pure function), so the
    # existing price still applies, which is correct: nothing changed.
    assert make_type().snapshot == make_type().snapshot
    # A different definition -> a different sku, so it MUST be re-priced.
    assert make_type().snapshot != make_type(unit_cpu="8000m").snapshot


def test_same_definition_on_two_clusters_shares_definition_snapshot():
    """Per-cluster pricing stays expressible (different ``snapshot``) while
    cross-cluster aggregation stays possible (same ``definition_snapshot``)."""
    a, b = make_type(cluster_id=1), make_type(cluster_id=2)
    assert a.snapshot != b.snapshot
    assert a.definition_snapshot == b.definition_snapshot


def test_sku_glob_separates_instances_from_volumes():
    """``sha1:*`` is the only usable "all instance types" pattern, and it must
    not spill onto storage skus."""
    assert fnmatch(make_type().snapshot, "sha1:*")
    assert not fnmatch(volume_sku("nfs", "aws"), "sha1:*")


# ---------------------------------------------------------------------------
# Catalog resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_fills_identity_and_facets(catalog_session):
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "2"})
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.definition_snapshot == row.definition_snapshot
    assert w.instance_type_name == row.name
    assert w.dimensions["sku_source"] == SKU_SOURCE_TYPE_SNAPSHOT
    # gpu_type comes from accelerator_group, NOT from the regex over the name
    # (which mangles the operator's ``--``-separated naming).
    assert w.dimensions["gpu_type"] == "nvidia-a100"
    assert w.dimensions["product"] == "NVIDIA-A100-SXM4-80GB"
    assert w.dimensions["manufacturer"] == "nvidia"
    assert w.dimensions["unit_cpu_milli"] == 4000
    assert w.dimensions["unit_memory_mib"] == 16384
    assert w.dimensions["vram_mib"] == A100_VRAM_MIB
    assert w.sku_count == 2
    assert w.needs_type_lookup is False


@pytest.mark.asyncio
async def test_resolve_ignores_soft_delete(catalog_session):
    """A retired type must stay resolvable — instances keep running on it, which
    is exactly why the projection soft-deletes instead of hard-deleting."""
    row = make_type(deleted_at=datetime(2026, 8, 1, 0, 0, 0))
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "1"})
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.definition_snapshot == row.definition_snapshot
    assert w.instance_type_name == row.name


@pytest.mark.asyncio
async def test_definition_snapshot_derived_when_column_is_null(catalog_session):
    """Upgrade path: the migration adds the column NULL and only ACTIVE rows are
    backfilled by the watch re-LIST, so a soft-deleted row still has to yield the
    right value — derived on read from (name, spec)."""
    row = make_type(deleted_at=datetime(2026, 8, 1, 0, 0, 0))
    expected = row.definition_snapshot
    row.definition_snapshot = None
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "1"})
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)
    assert w.definition_snapshot == expected


@pytest.mark.asyncio
async def test_missing_type_row_keeps_retrying(catalog_session):
    """The catalog may not have projected the type yet (or the deployment has no
    operator). The sku is still correct — it IS the snapshot — so metering
    continues with legacy facets while the lookup retries."""
    evt = make_event(type_snapshot="sha1:" + "0" * 40, resources={"accelerator": "1"})
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.sku == "sha1:" + "0" * 40
    assert w.definition_snapshot is None
    assert w.needs_type_lookup is True
    assert w.sku_count == 1  # a whole card is still billable


@pytest.mark.asyncio
async def test_missing_type_row_is_flagged_then_cleared(catalog_session):
    """A hash sku sitting next to legacy-derived facets must be recognizable as
    degraded, or it just looks like corrupt data."""
    row = make_type()
    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "1"})
    w = _open_window_from_event(evt)

    await _resolve_instance_type(catalog_session, w)
    assert w.dimensions["type_unresolved"] is True

    catalog_session.add(row)
    await catalog_session.commit()
    await _resolve_instance_type(catalog_session, w)
    assert "type_unresolved" not in w.dimensions


@pytest.mark.asyncio
async def test_whole_card_bills_without_detail_but_keeps_retrying(catalog_session):
    """``status.detail`` arrives asynchronously. A whole-card request does not
    need it to be BILLED (its share is a whole card regardless of the hardware),
    so metering must not stall — but the display facets are still missing, so the
    lookup has to keep retrying instead of declaring itself done."""
    row = make_type(with_detail=False)
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "2"})
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.sku_count == 2  # billable now
    assert w.definition_snapshot == row.definition_snapshot
    assert w.needs_type_lookup is True  # facets still owed
    assert "product" not in w.dimensions

    # Detail lands; the same snapshot still identifies the row (status is outside
    # the hashed spec).
    row.status = make_type().status
    catalog_session.add(row)
    await catalog_session.commit()
    await _resolve_instance_type(catalog_session, w)

    assert w.needs_type_lookup is False
    assert w.dimensions["product"] == "NVIDIA-A100-SXM4-80GB"
    assert w.dimensions["vram_mib"] == A100_VRAM_MIB


@pytest.mark.asyncio
async def test_retry_gives_up_after_the_bound(catalog_session):
    evt = make_event(type_snapshot="sha1:" + "0" * 40, resources={"accelerator": "1"})
    w = _open_window_from_event(evt)
    for _ in range(rc._TYPE_LOOKUP_MAX_ATTEMPTS):
        await _resolve_instance_type(catalog_session, w)
    assert w.needs_type_lookup is False


@pytest.mark.asyncio
async def test_an_unbillable_window_is_never_given_up_on(catalog_session):
    """Past the bound, a window with no ``sku_count`` must keep being retried.

    Clearing ``needs_type_lookup`` here looks like "stop chasing display facets",
    but nothing else ever sets a ``sku_count`` — so it drops the instance out of
    the usage report for the rest of its life, leaving a single log line as the
    only trace. A missing row is far harder to notice than a wrong number, and the
    seconds are recoverable as long as the retry continues: whatever has not sealed
    is still billed once the share resolves.
    """
    row = make_type(profiles=[("1g.10gb", 9728)])
    catalog_session.add(row)
    await catalog_session.commit()
    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_partitioned_profile": "4g.40gb"},
    )
    w = _open_window_from_event(evt)

    for _ in range(rc._TYPE_LOOKUP_MAX_ATTEMPTS * 2):
        await _resolve_instance_type(catalog_session, w)

    assert w.sku_count is None
    assert w.needs_type_lookup is True  # still retrying, well past the bound
    assert w.type_lookup_attempts > rc._TYPE_LOOKUP_MAX_ATTEMPTS

    # And it recovers: the operator backfills the profile, the very next retry
    # makes the window billable.
    row.status = make_type(profiles=[("4g.40gb", 40960)]).status
    catalog_session.add(row)
    await catalog_session.commit()
    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count == Decimal("0.5")
    assert w.needs_type_lookup is False


@pytest.mark.asyncio
async def test_a_billable_window_still_stops_chasing_display_facets(catalog_session):
    """The bound keeps its original meaning where it is safe.

    A whole-card request is billed correctly without ``status.detail``, so once the
    retries are exhausted there is nothing left to lose by stopping — only the
    product name / card VRAM stay missing.
    """
    row = make_type(with_detail=False)
    catalog_session.add(row)
    await catalog_session.commit()
    evt = make_event(type_snapshot=row.snapshot, resources={"accelerator": "2"})
    w = _open_window_from_event(evt)

    for _ in range(rc._TYPE_LOOKUP_MAX_ATTEMPTS):
        await _resolve_instance_type(catalog_session, w)

    assert w.sku_count == 2  # billable throughout
    assert w.needs_type_lookup is False  # gave up on the facets
    assert "product" not in w.dimensions


@pytest.mark.asyncio
async def test_the_retry_and_deferral_warnings_are_logged_once_not_per_tick(
    catalog_session, caplog
):
    """These two conditions are re-evaluated every tick (300s).

    Logging per evaluation put 288 identical lines a day per stuck instance into
    the log — unbounded now that an unbillable window is retried indefinitely —
    and buried the one line that carried information. Both are latched to the
    state change: entering the state, and (for the deferral) leaving it.
    """
    row = make_type(profiles=[("1g.10gb", 9728)])
    catalog_session.add(row)
    await catalog_session.commit()
    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_partitioned_profile": "4g.40gb"},
    )
    w = _open_window_from_event(evt)

    with caplog.at_level(logging.WARNING, logger=rc.logger.name):
        for _ in range(rc._TYPE_LOOKUP_MAX_ATTEMPTS * 2):
            await _resolve_instance_type(catalog_session, w)
    lookup_lines = [
        r
        for r in caplog.records
        if "resolved" in r.getMessage() or "NOT being metered" in r.getMessage()
    ]
    # One "retrying" line and one "NOT being metered" line, whatever the tick count.
    assert len(lookup_lines) == 2, [r.getMessage() for r in lookup_lines]
    assert any(r.levelno == logging.ERROR for r in lookup_lines)

    caplog.clear()
    c = ResourceUsageCollector()
    with caplog.at_level(logging.WARNING, logger=rc.logger.name):
        with patch.object(rc, "async_session", lambda: _yield(catalog_session)):
            for minute in (10, 20, 30):
                await c._settle_locked(w, datetime(2026, 8, 5, 11, minute, 0))
    deferrals = [r for r in caplog.records if "deferring settlement" in r.getMessage()]
    assert len(deferrals) == 1, [r.getMessage() for r in deferrals]


@pytest.mark.asyncio
async def test_a_cpu_type_resolves_on_the_first_attempt(catalog_session):
    """A card-less request must not wait for card VRAM.

    The retry above exists for ``status.detail``, whose ``memory`` is the
    ACCELERATOR's VRAM. A CPU-only type has no such field to fill, so treating it
    as "detail not backfilled yet" made every CPU instance re-query the catalog
    once per tick for the full bound (~100 minutes) and log a warning each time,
    before declaring it gave up on a type that had in fact resolved completely on
    the first attempt.
    """
    flavor = _cpu_type()
    catalog_session.add(flavor)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=flavor.snapshot, resources={"cpu": "2", "ram": "4Gi"}
    )
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.needs_type_lookup is False
    assert w.type_lookup_attempts == 1
    # Resolved, not abandoned: everything a CPU row bills on is present.
    assert w.sku_count == 2
    assert w.definition_snapshot == flavor.definition_snapshot
    assert w.dimensions["unit_cpu_milli"] == 1000
    assert w.dimensions["unit_memory_mib"] == 2048


@pytest.mark.asyncio
async def test_zero_accelerators_on_an_accelerated_type_is_refused(catalog_session):
    """An accelerated type's ``unit_resources`` means "what comes with one card",
    so a card-less request has no defensible billing unit. Refuse, don't guess."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "0", "cpu": "8", "ram": "32Gi"},
    )
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count is None


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------


def test_null_type_snapshot_falls_back_and_is_tagged():
    """Pre-v2.3.0 instances have no ``type_snapshot`` and it cannot be
    backfilled, so the legacy name-based sku stays — but a row that fell back
    must be distinguishable, or a hash-vs-name sku mix looks like corruption."""
    evt = make_event(
        type_snapshot=None,
        resources={"accelerator": "1"},
        description='{"spec": {"product": "NVIDIA-A100", "memory": "80Gi"}}',
    )
    w = _open_window_from_event(evt)
    assert w.sku == "gpustack--generic--nvidia-a100-linux-amd64"
    assert w.dimensions["sku_source"] == SKU_SOURCE_DESCRIPTION
    assert w.needs_type_lookup is False


# ---------------------------------------------------------------------------
# Sliced accelerators
# ---------------------------------------------------------------------------


def test_soft_slice_bills_the_vram_share():
    """2 cards at 25% each = half a card's worth of sellable capacity."""
    row = make_type()
    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "2", "accelerator_sliced_memory_percentage": 25},
    )
    w = _open_window_from_event(evt)
    assert w.sku_count == Decimal("0.5")
    assert w.dimensions["slice_mode"] == "ratio"
    assert w.dimensions["slice_share_milli"] == 250


def test_zero_percentage_means_whole_card_not_zero_share():
    row = make_type()
    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_sliced_memory_percentage": 0},
    )
    w = _open_window_from_event(evt)
    assert w.dimensions["slice_mode"] == "whole"
    assert w.sku_count == 1


def test_cpu_floor_does_not_inflate_the_weight():
    """A tiny slice gets a 1c1g floor for usability; CPU is overcommitted on an
    accelerator node and excluded from the accelerated queue, so it is given
    away, not sold. The weight stays the VRAM share."""
    row = make_type(unit_cpu="8000m", unit_ram="32768Mi")
    evt = make_event(
        type_snapshot=row.snapshot,
        resources={
            "accelerator": "1",
            "accelerator_sliced_memory_percentage": 10,
            # 12.5% of the unit CPU — above the 10% VRAM share.
            "cpu": "1",
            "ram": "1Gi",
        },
    )
    w = _open_window_from_event(evt)
    assert w.dimensions["slice_share_milli"] == 100
    assert w.sku_count == Decimal("0.1")


@pytest.mark.asyncio
async def test_partition_share_uses_memory_mib_not_the_profile_name(catalog_session):
    """``1g.10gb`` really has ~9728 MiB on an A100-80GB: parsing the name would
    over-state it by ~5%. The reported ``memoryMib`` is also the number the
    operator folds into the Kueue credit request, so quota and bill agree."""
    row = make_type(profiles=[("1g.10gb", 9728), ("3g.40gb", 40192)])
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_partitioned_profile": "1g.10gb"},
    )
    w = _open_window_from_event(evt)
    assert w.sku_count is None  # not resolvable before the catalog lookup
    await _resolve_instance_type(catalog_session, w)

    # ceil(9728 * 1000 / 81920) = 119, not 10/80 = 125.
    assert w.dimensions["slice_share_milli"] == 119
    assert w.sku_count == Decimal("0.119")
    assert w.dimensions["slice_mode"] == "profile"


@pytest.mark.asyncio
async def test_seven_small_partitions_sum_below_a_whole_card(catalog_session):
    """MIG's 7-way split leaves partition overhead that physically cannot be
    sold, so 7 x ``1g.10gb`` summing to <1 card is correct, not a rounding bug."""
    row = make_type(profiles=[("1g.10gb", 9728)])
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "7", "accelerator_partitioned_profile": "1g.10gb"},
    )
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count == Decimal("0.833")
    assert w.sku_count < 1


@pytest.mark.asyncio
async def test_unresolvable_partition_is_not_billed_as_a_whole_card(catalog_session):
    """The failure mode that matters: an unknown / not-yet-backfilled profile
    must defer settlement, never fall back to a whole card (up to 8x overcharge)."""
    row = make_type(profiles=[("1g.10gb", 9728)])
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_partitioned_profile": "4g.40gb"},
    )
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)

    assert w.sku_count is None
    assert w.needs_type_lookup is True

    # And the settle path must actually skip it — no row, no seconds.
    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(catalog_session)):
        await c._settle_locked(w, datetime(2026, 8, 5, 10, 30, 0))
    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert rows == []


@pytest.mark.asyncio
async def test_partition_resolves_once_detail_is_backfilled(catalog_session):
    """``status.detail`` arrives asynchronously (a MODIFIED event, not ADDED), so
    an instance can be metered before it lands. The deferred seconds must not be
    lost — they are settled by the first pass after the backfill."""
    row = make_type(with_detail=False)
    catalog_session.add(row)
    await catalog_session.commit()

    evt = make_event(
        type_snapshot=row.snapshot,
        resources={"accelerator": "1", "accelerator_partitioned_profile": "1g.10gb"},
    )
    w = _open_window_from_event(evt)
    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count is None

    # The operator backfills the detail; the same snapshot still identifies the
    # row (status is outside the hashed spec).
    backfilled = make_type(profiles=[("1g.10gb", 9728)])
    assert backfilled.snapshot == row.snapshot
    row.status = backfilled.status
    catalog_session.add(row)
    await catalog_session.commit()

    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count == Decimal("0.119")

    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(catalog_session)):
        await c._settle_locked(w, datetime(2026, 8, 5, 10, 30, 0))
    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert len(rows) == 1
    # The whole window is billed from its start, not just from the resolve.
    assert rows[0].quantity == 1800


# ---------------------------------------------------------------------------
# Per-shape natural key
# ---------------------------------------------------------------------------


async def _settle(session, window, end_ts):
    c = ResourceUsageCollector()
    with patch.object(rc, "async_session", lambda: _yield(session)):
        await c._settle_locked(window, end_ts)


@pytest.mark.asyncio
async def test_reconfiguration_within_one_hour_splits_into_two_rows(catalog_session):
    """The M4 regression net: 4 cards for 30 min then 1 card for 20 min inside
    one UTC hour must bill 4x1800 + 1x1200, not 3000 seconds at one of the two
    shapes."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    big = _open_window_from_event(
        make_event(
            type_snapshot=row.snapshot,
            resources={"accelerator": "4"},
            occurred_at=datetime(2026, 8, 5, 10, 0, 0),
        )
    )
    await _resolve_instance_type(catalog_session, big)
    await _settle(catalog_session, big, datetime(2026, 8, 5, 10, 30, 0))

    # Stop -> change the card count -> start again, still inside hour 10.
    small = _open_window_from_event(
        make_event(
            type_snapshot=row.snapshot,
            resources={"accelerator": "1"},
            occurred_at=datetime(2026, 8, 5, 10, 35, 0),
        )
    )
    await _resolve_instance_type(catalog_session, small)
    await _settle(catalog_session, small, datetime(2026, 8, 5, 10, 55, 0))

    rows = (
        await catalog_session.exec(
            select(MeteredUsage).order_by(MeteredUsage.sku_count.desc())
        )
    ).all()
    assert len(rows) == 2
    assert [(r.sku_count, r.quantity) for r in rows] == [(4, 1800), (1, 1200)]
    # Card-seconds: 4*1800 + 1*1200 = 8400. The single-row behaviour would have
    # billed 3000 * 1 = 3000 (64% under) or 3000 * 4 = 12000 (82% over).
    assert sum(int(r.sku_count) * r.quantity for r in rows) == 8400


@pytest.mark.asyncio
async def test_switching_instance_type_within_one_hour_splits_too(catalog_session):
    """Different type -> different sku -> its own row, so an A100 half-hour is
    never priced at the H100 rate."""
    a100 = make_type()
    h100 = make_type(
        name="gpustack--generic--nvidia-h100-linux-amd64",
        accelerator_group="nvidia-h100",
    )
    catalog_session.add_all([a100, h100])
    await catalog_session.commit()

    for snapshot, start, end in (
        (
            a100.snapshot,
            datetime(2026, 8, 5, 10, 0, 0),
            datetime(2026, 8, 5, 10, 30, 0),
        ),
        (
            h100.snapshot,
            datetime(2026, 8, 5, 10, 35, 0),
            datetime(2026, 8, 5, 10, 55, 0),
        ),
    ):
        w = _open_window_from_event(
            make_event(
                type_snapshot=snapshot,
                resources={"accelerator": "1"},
                occurred_at=start,
            )
        )
        await _resolve_instance_type(catalog_session, w)
        await _settle(catalog_session, w, end)

    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert {r.sku for r in rows} == {a100.snapshot, h100.snapshot}
    assert len(rows) == 2


def test_cpu_instance_carries_no_card_facets():
    """A CPU instance has no card, so it records nothing that describes one.

    Every one of these used to be written unconditionally: ``gpu_type`` held
    the regex's leftovers from a CPU flavor name, ``vram_mib`` was 0, and
    ``slice_mode``/``slice_share_milli`` said a whole card was held
    exclusively. Absent means "not applicable"; a zero means "measured".
    """
    evt = make_event(
        type_snapshot="sha1:cpu",
        resources={"cpu": "1", "ram": "2Gi"},
        type_name="gpustack--generic-linux-amd64",
    )
    dims = _open_window_from_event(evt).dimensions

    for absent in (
        "gpu_type",
        "vram_mib",
        "slice_mode",
        "slice_share_milli",
        "sliced_memory_percentage",
        "partitioned_profile",
    ):
        assert absent not in dims, absent
    # What a CPU instance does have.
    assert dims["gpu_count"] == 0
    assert dims["cpu_milli"] == 1000
    assert dims["memory_mib"] == 2048


def test_gpu_instance_still_carries_its_card_facets():
    """The counterpart: gating the facets on the card count must not strip them
    from an instance that has cards."""
    evt = make_event(
        type_snapshot="sha1:gpu",
        resources={"accelerator": "1", "accelerator_sliced_memory_percentage": 25},
    )
    dims = _open_window_from_event(evt).dimensions

    assert dims["slice_mode"] == "ratio"
    assert dims["slice_share_milli"] == 250
    assert dims["sliced_memory_percentage"] == 25
    assert "gpu_type" in dims


def _cpu_type(
    *, name: str = "gpustack--generic-linux-amd64", unit_cpu: str = "1000m", **kw
) -> GPUInstanceType:
    """A CPU-only flavor: not acceleratable, so its ``unitResources`` describe
    one billable unit of CPU/RAM rather than the resources that accompany a
    card."""
    return make_type(
        name=name,
        accelerator_group=None,
        acceleratable=False,
        unit_cpu=unit_cpu,
        unit_ram="2048Mi",
        with_detail=False,
        **kw,
    )


async def _run_segment(session, snapshot, resources, start, end, *, type_name=None):
    """Meter one configuration of instance 1 over ``[start, end]``."""
    w = _open_window_from_event(
        make_event(
            type_snapshot=snapshot,
            resources=resources,
            occurred_at=start,
            **({"type_name": type_name} if type_name else {}),
        )
    )
    await _resolve_instance_type(session, w)
    await _settle(session, w, end)
    return w


@pytest.mark.asyncio
async def test_cpu_resize_splits_on_the_count_not_the_sku(catalog_session):
    """CPU -> CPU: the flavor is unchanged, so the sku is too — 2c4g then 4c8g
    on a 1c2g flavor is one sku at two counts, and the natural key has to carry
    the count or the whole hour reprices at whichever landed last."""
    flavor = _cpu_type()
    catalog_session.add(flavor)
    await catalog_session.commit()

    await _run_segment(
        catalog_session,
        flavor.snapshot,
        {"cpu": "2", "ram": "4Gi"},
        datetime(2026, 8, 5, 10, 0, 0),
        datetime(2026, 8, 5, 10, 30, 0),
    )
    await _run_segment(
        catalog_session,
        flavor.snapshot,
        {"cpu": "4", "ram": "8Gi"},
        datetime(2026, 8, 5, 10, 35, 0),
        datetime(2026, 8, 5, 10, 55, 0),
    )

    rows = (
        await catalog_session.exec(
            select(MeteredUsage).order_by(MeteredUsage.sku_count)
        )
    ).all()
    assert len(rows) == 2
    assert {r.sku for r in rows} == {flavor.snapshot}
    assert [(r.sku_count, r.quantity) for r in rows] == [(2, 1800), (4, 1200)]
    assert sum(int(r.sku_count) * r.quantity for r in rows) == 2 * 1800 + 4 * 1200


@pytest.mark.asyncio
async def test_cpu_to_gpu_switch_splits_and_keeps_each_shape(catalog_session):
    """CPU -> GPU: both the sku and the meaning of the count change (units of a
    flavor become cards), so each segment must keep its own dimensions — the
    aggregate row's ``dimensions`` only ever describe the latest."""
    flavor = _cpu_type()
    gpu = make_type()
    catalog_session.add_all([flavor, gpu])
    await catalog_session.commit()

    await _run_segment(
        catalog_session,
        flavor.snapshot,
        {"cpu": "2", "ram": "4Gi"},
        datetime(2026, 8, 5, 10, 0, 0),
        datetime(2026, 8, 5, 10, 30, 0),
        type_name=flavor.name,
    )
    await _run_segment(
        catalog_session,
        gpu.snapshot,
        {"accelerator": "2"},
        datetime(2026, 8, 5, 10, 35, 0),
        datetime(2026, 8, 5, 10, 55, 0),
    )

    rows = (
        await catalog_session.exec(
            select(MeteredUsage).order_by(MeteredUsage.bucket_start, MeteredUsage.id)
        )
    ).all()
    assert len(rows) == 2
    by_sku = {r.sku: r for r in rows}
    cpu_row, gpu_row = by_sku[flavor.snapshot], by_sku[gpu.snapshot]
    # Same count, entirely different meaning — 2 flavor units vs 2 cards. The
    # per-segment dimensions are what let the UI say which is which.
    assert cpu_row.sku_count == 2 and gpu_row.sku_count == 2
    assert not cpu_row.dimensions.get("gpu_count")
    assert gpu_row.dimensions.get("gpu_count") == 2
    assert cpu_row.dimensions.get("unit_cpu_milli") == 1000
    assert cpu_row.instance_type_name == flavor.name
    assert gpu_row.instance_type_name == gpu.name


@pytest.mark.asyncio
async def test_gpu_to_cpu_switch_splits_and_keeps_each_shape(catalog_session):
    """GPU -> CPU, the reverse direction, seen on live data: 2 cards then a
    1c2g CPU flavor. Both segments bill at their own rate."""
    gpu = make_type()
    flavor = _cpu_type()
    catalog_session.add_all([gpu, flavor])
    await catalog_session.commit()

    await _run_segment(
        catalog_session,
        gpu.snapshot,
        {"accelerator": "2"},
        datetime(2026, 8, 5, 10, 0, 0),
        datetime(2026, 8, 5, 10, 30, 0),
    )
    await _run_segment(
        catalog_session,
        flavor.snapshot,
        {"cpu": "1", "ram": "2Gi"},
        datetime(2026, 8, 5, 10, 35, 0),
        datetime(2026, 8, 5, 10, 55, 0),
        type_name=flavor.name,
    )

    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert len(rows) == 2
    by_sku = {r.sku: r for r in rows}
    assert by_sku[gpu.snapshot].sku_count == 2
    assert by_sku[flavor.snapshot].sku_count == 1
    # 2 cards x 1800s + 1 unit x 1200s
    assert sum(int(r.sku_count) * r.quantity for r in rows) == 2 * 1800 + 1200


@pytest.mark.asyncio
async def test_gpu_slice_ratio_change_splits_without_changing_the_sku(
    catalog_session,
):
    """GPU -> GPU by slice: the type is untouched, only the share moves, so the
    split rides entirely on the fractional count."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    await _run_segment(
        catalog_session,
        row.snapshot,
        {"accelerator": "1", "accelerator_sliced_memory_percentage": 25},
        datetime(2026, 8, 5, 10, 0, 0),
        datetime(2026, 8, 5, 10, 30, 0),
    )
    await _run_segment(
        catalog_session,
        row.snapshot,
        {"accelerator": "1", "accelerator_sliced_memory_percentage": 50},
        datetime(2026, 8, 5, 10, 35, 0),
        datetime(2026, 8, 5, 10, 55, 0),
    )

    rows = (
        await catalog_session.exec(
            select(MeteredUsage).order_by(MeteredUsage.sku_count)
        )
    ).all()
    assert len(rows) == 2
    assert {r.sku for r in rows} == {row.snapshot}
    assert [r.sku_count for r in rows] == [Decimal("0.25"), Decimal("0.5")]


@pytest.mark.asyncio
async def test_shape_neutral_change_stays_one_row(catalog_session):
    """Only the amount-deciding inputs are in the key: swapping the image leaves
    ``(sku, sku_count)`` alone, so the hour stays a single accumulating row."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    for start, end in (
        (datetime(2026, 8, 5, 10, 0, 0), datetime(2026, 8, 5, 10, 30, 0)),
        (datetime(2026, 8, 5, 10, 35, 0), datetime(2026, 8, 5, 10, 55, 0)),
    ):
        w = _open_window_from_event(
            make_event(
                type_snapshot=row.snapshot,
                resources={"accelerator": "2"},
                occurred_at=start,
            )
        )
        await _resolve_instance_type(catalog_session, w)
        await _settle(catalog_session, w, end)

    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert len(rows) == 1
    assert rows[0].quantity == 1800 + 1200


@pytest.mark.asyncio
async def test_decimal_representation_does_not_split_a_row(catalog_session):
    """``0.5`` and ``0.50`` are the same shape. The DB compares them numerically
    — which is precisely why the shape lives in the unique constraint rather than
    in a hashed fingerprint column, where the two would hash differently."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    for value, start, end in (
        (
            Decimal("0.5"),
            datetime(2026, 8, 5, 10, 0, 0),
            datetime(2026, 8, 5, 10, 30, 0),
        ),
        (
            Decimal("0.50"),
            datetime(2026, 8, 5, 10, 35, 0),
            datetime(2026, 8, 5, 10, 55, 0),
        ),
    ):
        w = _open_window_from_event(
            make_event(
                type_snapshot=row.snapshot,
                resources={
                    "accelerator": "2",
                    "accelerator_sliced_memory_percentage": 25,
                },
                occurred_at=start,
            )
        )
        await _resolve_instance_type(catalog_session, w)
        w.sku_count = value
        await _settle(catalog_session, w, end)

    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    assert len(rows) == 1
    assert rows[0].quantity == 3000


@pytest.mark.asyncio
async def test_shape_revised_mid_window_does_not_double_count(catalog_session):
    """The subtle hazard of putting the billed shape in the natural key.

    One window can change shape WITHOUT a phase transition — the operator revises
    a profile's ``memoryMib``, so the same running instance moves to a new
    ``sku_count`` and therefore a new row. That new row has no ``settled_until``
    of its own, so if the window replayed from ``window_start`` it would bill the
    already-settled time a second time. What prevents it is ``settled_through``
    being tracked per WINDOW, not per row/shape. Pin that.
    """
    row = make_type(profiles=[("1g.10gb", 9728)])
    catalog_session.add(row)
    await catalog_session.commit()

    w = _open_window_from_event(
        make_event(
            type_snapshot=row.snapshot,
            resources={
                "accelerator": "1",
                "accelerator_partitioned_profile": "1g.10gb",
            },
            occurred_at=datetime(2026, 8, 5, 10, 0, 0),
        )
    )
    await _resolve_instance_type(catalog_session, w)
    await _settle(catalog_session, w, datetime(2026, 8, 5, 10, 30, 0))
    assert w.sku_count == Decimal("0.119")

    # The operator re-detects the profile slightly larger; same window, still
    # running, no phase transition.
    row.status = make_type(profiles=[("1g.10gb", 10240)]).status
    catalog_session.add(row)
    await catalog_session.commit()
    w.needs_type_lookup = True
    w.share_milli = None
    await _resolve_instance_type(catalog_session, w)
    assert w.sku_count == Decimal("0.125")

    await _settle(catalog_session, w, datetime(2026, 8, 5, 10, 45, 0))

    rows = (await catalog_session.exec(select(MeteredUsage))).all()
    # Two shapes -> two rows, but the SECONDS are partitioned, not duplicated.
    assert len(rows) == 2
    assert sum(r.quantity for r in rows) == 2700  # 45 min, counted once
    assert sorted(r.quantity for r in rows) == [900, 1800]


@pytest.mark.asyncio
async def test_sealed_rows_are_never_rewritten(catalog_session):
    """Pre-upgrade history is protected by sealing, not by hoping the new logic
    leaves it alone. Nothing on a sealed row may move."""
    row = make_type()
    catalog_session.add(row)
    await catalog_session.commit()

    sealed = MeteredUsage(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=1,
        resource_name="gpu-1",
        # A legacy name-based sku and an integer count, as written before upgrade.
        sku="gpustack--generic-ln-x64-4c-16g-98g--nvidia-tesla-t4-1d",
        sku_count=4,
        bucket_start=datetime(2026, 8, 5, 10, 0, 0),
        quantity=3600,
        unit="seconds",
        settled_until=datetime(2026, 8, 5, 10, 30, 0),
        sealed_at=datetime(2026, 8, 5, 11, 5, 0),
    )
    catalog_session.add(sealed)
    await catalog_session.commit()

    w = _open_window_from_event(
        make_event(
            type_snapshot=row.snapshot,
            resources={"accelerator": "1"},
            occurred_at=datetime(2026, 8, 5, 10, 40, 0),
        )
    )
    await _resolve_instance_type(catalog_session, w)
    await _settle(catalog_session, w, datetime(2026, 8, 5, 10, 55, 0))

    await catalog_session.refresh(sealed)
    assert sealed.quantity == 3600
    assert sealed.sku_count == 4
    assert sealed.sku == "gpustack--generic-ln-x64-4c-16g-98g--nvidia-tesla-t4-1d"
    assert sealed.settled_until.replace(tzinfo=None) == datetime(2026, 8, 5, 10, 30, 0)


@pytest.mark.asyncio
async def test_legacy_integer_counts_read_back_numerically_equal(catalog_session):
    """The ``Integer -> Numeric`` change is a widening: no historical row's
    meaning may shift, so GPU-Hours over old rows must be unchanged."""
    catalog_session.add(
        MeteredUsage(
            meter_key=METER_INSTANCE_UPTIME,
            resource_type=RESOURCE_TYPE_GPU_INSTANCE,
            resource_id=1,
            resource_name="gpu-1",
            sku="legacy-flavor",
            sku_count=4,
            bucket_start=datetime(2026, 8, 5, 10, 0, 0),
            quantity=3600,
            unit="seconds",
        )
    )
    await catalog_session.commit()
    stored = (await catalog_session.exec(select(MeteredUsage))).first()
    assert stored.sku_count == 4
    assert int(stored.sku_count) * stored.quantity == 14400
