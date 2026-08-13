import contextlib
import io
import zipfile
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import ForbiddenException, InvalidException
from gpustack.routes.resource_usage import (
    ResourceExportRequest,
    estimate_gpu_instances_breakdown_export,
    export_gpu_instances_breakdown,
    export_storage_breakdown,
)
from gpustack.schemas.users import User
from gpustack.utils.usage_export import (
    UNTRACKED_ORGANIZATION_NAME,
    build_resource_export_columns,
    resource_export_column_keys,
    resource_export_row,
)

GPU_METRICS = ["gpu_hours", "instance_hours"]
STORAGE_METRICS = ["gb_days", "gb_hours"]


def _result(rows):
    result = MagicMock()
    result.all.return_value = rows
    result.first.return_value = rows[0] if rows else None
    return result


def _ctx_for(user):
    ctx = MagicMock()
    ctx.user = user
    ctx.is_platform_admin = bool(getattr(user, "is_admin", False))
    ctx.current_principal_id = None if ctx.is_platform_admin else 1
    ctx.org_role = None
    ctx.current_is_personal_scope = False
    return ctx


class _StreamResult:
    def __init__(self, rows):
        self._rows = rows

    async def partitions(self, size):
        for start in range(0, len(self._rows), size):
            yield self._rows[start : start + size]


def _session(*, counts, stream_rows):
    session = MagicMock()
    session.exec = AsyncMock(side_effect=[_result([count]) for count in counts])
    session.stream = AsyncMock(
        side_effect=[_StreamResult(rows) for rows in stream_rows]
    )
    return session


async def _collect(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    return b"".join(chunks)


@contextlib.contextmanager
def _stubbed_enrichment():
    """Neutralize the enrichment leg of the stream.

    Display names are resolved on a SECOND session because the export holds an
    open cursor on the first — so a unit test has to stub both the session
    factory and the lookup, not just the lookup.
    """
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("gpustack.routes.resource_usage.async_session", return_value=session_ctx),
        patch("gpustack.routes.resource_usage._enrich_items", new=AsyncMock()),
    ):
        yield


def _admin():
    return User(id=1, name="admin", is_admin=True)


def _instance_rows():
    return [
        SimpleNamespace(
            group_key="flavor-a",
            group_id=None,
            gpu_hours=12.0,
            instance_hours=6.0,
            resources=2,
            active_users=1,
            last_active=None,
            sku="flavor-a",
        )
    ]


# --------------------------------------------------------------------------
# Column layout
# --------------------------------------------------------------------------


def test_entity_dimension_spells_out_its_name_in_columns():
    # A resource item carries its dimension generically as id/key/deleted; the
    # exported columns name it, so the file is readable without the request
    # that produced it.
    keys = resource_export_column_keys(["instance"], GPU_METRICS)
    assert keys[:3] == ["instance_id", "instance_name", "instance_deleted"]
    # Per-resource rows carry an owner.
    assert {"owner_id", "owner_name", "owner_deleted"} <= set(keys)


def test_bucket_dimension_has_no_id_or_deleted_columns():
    # instance_type is a bucket, not an entity — it has no id and cannot be
    # "deleted", so emitting those columns would be dead weight.
    keys = resource_export_column_keys(["instance_type"], GPU_METRICS)
    assert keys[0] == "instance_type_name"
    assert not [k for k in keys if k.endswith("_id") or k.endswith("_deleted")]


def test_metrics_follow_the_endpoint_not_a_fixed_list():
    gpu = resource_export_column_keys(["date"], GPU_METRICS)
    storage = resource_export_column_keys(["date"], STORAGE_METRICS)
    assert "gpu_hours" in gpu and "gb_days" not in gpu
    assert "gb_days" in storage and "gpu_hours" not in storage


def test_resource_count_column_is_named_after_what_it_counts():
    """One server metric, the two names the product gives it.

    ``metrics.resources`` counts the distinct resources of a group that still
    exist. The page calls that Active Instances on the GPU tab and Active
    Volumes on Storage; a file column called "Resources" was neither, so a
    reader could not line the export up with the screen it came from. The
    sheet's metric set is what picks the name — ``group_by`` can't, since
    ``["user"]`` is a valid sheet on either tab.
    """
    gpu = resource_export_column_keys(["user"], GPU_METRICS)
    storage = resource_export_column_keys(["user"], STORAGE_METRICS)
    assert "active_instances" in gpu and "active_volumes" not in gpu
    assert "active_volumes" in storage and "active_instances" not in storage
    # The generic key never reaches the file, under either tab.
    assert "resources" not in gpu and "resources" not in storage

    titles = dict(
        zip(storage, build_resource_export_columns(["user"], STORAGE_METRICS))
    )
    assert titles["active_volumes"] == "Active Volumes"
    assert (
        dict(zip(gpu, build_resource_export_columns(["user"], GPU_METRICS)))[
            "active_instances"
        ]
        == "Active Instances"
    )

    # ...and the value still lands in it — the item keeps the generic name.
    item = {
        "id": 3,
        "key": "alice",
        "deleted": False,
        "metrics": {"gb_days": 5.0, "gb_hours": 120.0, "resources": 2},
    }
    row = dict(zip(storage, resource_export_row(item, ["user"], STORAGE_METRICS)))
    assert row["active_volumes"] == 2


def test_resource_export_dates_are_day_precision_like_the_token_export():
    """One report, one date format.

    ``metered_usage`` buckets hourly, so a resource row arrives with instants
    where the token export (a daily rollup) has plain days. Written raw they
    put "2026-08-02 00:00:00" in the Date column and pinned Last Active to
    whichever hour bucket happened to be last — extra precision that is an
    artifact of the storage layer, not something anyone asked for.
    """
    item = {
        "date": datetime(2026, 8, 2, 0, 0),
        "id": 7,
        "key": "vol-a",
        "deleted": False,
        "metrics": {
            "gb_days": 500,
            "gb_hours": 12000,
            "resources": 0,
            "active_users": 1,
            "last_active": datetime(2026, 8, 2, 23, 0, tzinfo=timezone.utc),
        },
    }
    group_by = ["date", "volume"]
    metrics = ["gb_days", "gb_hours"]

    row = dict(
        zip(
            resource_export_column_keys(group_by, metrics),
            resource_export_row(item, group_by, metrics),
        )
    )

    assert row["date"] == date(2026, 8, 2)
    assert row["last_active"] == date(2026, 8, 2)
    assert not isinstance(row["date"], datetime)
    assert not isinstance(row["last_active"], datetime)


def test_hour_buckets_keep_their_time_of_day():
    """Day precision is right for a day bucket and destroys an hour one.

    ``metered_usage`` is hourly, and the resource tabs expose that granularity.
    Collapsing those buckets to a calendar day (correct for day/week/month)
    would give all 24 rows of a volume's day the same Date cell — a file whose
    rows can't be told apart or re-sorted, and where an hourly series looks
    like duplicated data.
    """
    item = {
        "date": datetime(2026, 8, 2, 13, 0),
        "id": 7,
        "key": "vol-a",
        "deleted": False,
        "metrics": {"gb_days": 1, "gb_hours": 24, "resources": 1, "active_users": 1},
    }
    group_by = ["date", "volume"]

    row = dict(
        zip(
            resource_export_column_keys(group_by, STORAGE_METRICS),
            resource_export_row(item, group_by, STORAGE_METRICS, granularity="hour"),
        )
    )

    assert row["date"] == datetime(2026, 8, 2, 13, 0)


def test_resource_export_row_matches_columns():
    item = {
        "id": 7,
        "key": "inst-a",
        "deleted": False,
        "creator_id": 3,
        "creator_name": "alice",
        "creator_deleted": True,
        "metrics": {
            "gpu_hours": 12.0,
            "instance_hours": 6.0,
            "resources": 1,
            "active_users": 1,
            "last_active": None,
        },
    }
    columns = resource_export_column_keys(["instance"], GPU_METRICS)
    values = dict(zip(columns, resource_export_row(item, ["instance"], GPU_METRICS)))

    assert values["instance_id"] == 7
    assert values["instance_name"] == "inst-a"
    assert values["instance_deleted"] is False
    # The owner's deletion is tracked separately from the resource's.
    assert values["owner_name"] == "alice"
    assert values["owner_deleted"] is True
    assert values["gpu_hours"] == 12.0


def test_untracked_organization_row_is_named():
    item = {"id": None, "key": None, "deleted": False, "metrics": {}}
    columns = resource_export_column_keys(["organization"], GPU_METRICS)
    values = dict(
        zip(columns, resource_export_row(item, ["organization"], GPU_METRICS))
    )

    assert values["organization_id"] is None
    assert values["organization_name"] == UNTRACKED_ORGANIZATION_NAME


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_export_streams_csv():
    user = _admin()
    session = _session(counts=[1], stream_rows=[_instance_rows()])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["instance_type"],
        format="csv",
    )

    with _stubbed_enrichment():
        response = await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )
        body = await _collect(response)
    text = body.decode("utf-8-sig")

    assert response.media_type.startswith("text/csv")
    assert text.splitlines()[0].startswith("Instance Type,")
    assert "flavor-a" in text
    assert text.strip().splitlines()[-1].startswith("# rows=1 scope=all")


@pytest.mark.asyncio
async def test_storage_export_rejects_gpu_only_dimension():
    # instance_type belongs to the GPU tab; letting it through here would
    # silently produce an empty file instead of an error.
    user = _admin()
    session = _session(counts=[], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["instance_type"],
    )

    with pytest.raises(InvalidException):
        await export_storage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_export_refuses_two_entity_dimensions():
    """Two dimensions collide in the query and would fabricate a column.

    ``_group_columns`` labels every dimension ``group_id`` / ``group_key``, so
    a row grouped by volume AND user carries exactly one pair — the volume's.
    The export spells both dimensions out into their own columns, so it would
    print the volume's id and name under "User ID" / "User". Refusing is the
    only honest answer; the two are available as two sheets.
    """
    user = _admin()
    session = _session(counts=[], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["volume", "user"],
    )

    with pytest.raises(InvalidException):
        await export_storage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_date_plus_one_dimension_is_still_allowed():
    # The guard above must not catch the trend export, which is exactly
    # date + one dimension.
    user = _admin()
    session = _session(counts=[1], stream_rows=[_instance_rows()])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date", "instance"],
        format="csv",
    )

    with _stubbed_enrichment():
        response = await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )
        body = await _collect(response)

    assert body.decode("utf-8-sig").splitlines()[0].startswith("Date,Instance ID,")


@pytest.mark.asyncio
async def test_multi_sheet_resource_export_is_zipped():
    user = _admin()
    session = _session(counts=[1, 1], stream_rows=[_instance_rows(), _instance_rows()])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        format="csv",
        sheets=[
            {"key": "instance_type", "group_by": ["instance_type"]},
            {"key": "instance", "group_by": ["instance"]},
        ],
    )

    with _stubbed_enrichment():
        response = await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )
        body = await _collect(response)

    archive = zipfile.ZipFile(io.BytesIO(body))
    assert archive.testzip() is None
    assert archive.namelist() == ["by_instance_type.csv", "by_instance.csv"]


@pytest.mark.asyncio
async def test_resource_export_refuses_silent_scope_downgrade():
    user = User(id=2, name="member", is_admin=False)
    session = _session(counts=[], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["instance_type"],
        scope="all",
    )

    with pytest.raises(ForbiddenException):
        await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_resource_export_rejects_oversized_with_details(monkeypatch):
    # 90 rows at 10 per file is 9 parts — deliberately inside
    # USAGE_EXPORT_MAX_SPLIT_MEMBERS, so this stays a test of WHICH remedies an
    # ordinary over-limit query gets under the shipped defaults. A ratio that
    # overran the cap would silently turn it into a test of the cap instead.
    monkeypatch.setattr("gpustack.routes.resource_usage.envs.USAGE_EXPORT_MAX_ROWS", 10)
    user = _admin()
    session = _session(counts=[90], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        group_by=["instance_type"],
    )

    with pytest.raises(InvalidException) as excinfo:
        await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )

    details = excinfo.value.details
    assert details["kind"] == "export_too_large"
    actions = {item["action"] for item in details["suggestions"]}
    # Both remedies, for every grouping: parts are row slices, so a dateless
    # grouping is no longer a special case.
    assert actions == {"shorten_range", "split_export"}
    split = next(i for i in details["suggestions"] if i["action"] == "split_export")
    assert split["parts"] == 9  # 90 rows / 10


@pytest.mark.asyncio
async def test_resource_estimate_returns_per_sheet_totals():
    user = _admin()
    session = _session(counts=[40, 12], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        sheets=[
            {"key": "instance_type", "group_by": ["instance_type"]},
            {"key": "user", "group_by": ["user"]},
        ],
    )

    response = await estimate_gpu_instances_breakdown_export(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert [sheet.total for sheet in response.sheets] == [40, 12]
    assert response.total == 52


def test_export_request_requires_exactly_one_shape():
    # Neither shape, or both, is ambiguous about which one wins — reject rather
    # than silently pick.
    base = dict(start_date=date(2026, 4, 1), end_date=date(2026, 4, 2))
    with pytest.raises(ValidationError):
        ResourceExportRequest(**base)
    with pytest.raises(ValidationError):
        ResourceExportRequest(
            **base,
            group_by=["instance"],
            sheets=[{"key": "user", "group_by": ["user"]}],
        )


@pytest.mark.asyncio
async def test_oversized_date_grouping_does_offer_split(monkeypatch):
    # 9 parts — under the member cap, so the offer is not withheld for a
    # reason this test is not about.
    monkeypatch.setattr("gpustack.routes.resource_usage.envs.USAGE_EXPORT_MAX_ROWS", 10)
    user = _admin()
    session = _session(counts=[90], stream_rows=[])
    request = ResourceExportRequest(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        group_by=["date", "instance"],
    )

    with pytest.raises(InvalidException) as excinfo:
        await export_gpu_instances_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )

    actions = {i["action"] for i in excinfo.value.details["suggestions"]}
    assert "split_export" in actions
