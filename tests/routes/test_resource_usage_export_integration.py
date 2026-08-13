"""End-to-end resource-export tests against a real database and ASGI app.

The resource path has one structural risk the token path doesn't: display
names are resolved by a SECOND connection while the export holds an open
cursor on the first. A mocked session cannot show whether that actually works
— only a real engine can.
"""

import csv
import io
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import resource_usage as resource_routes
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    RESOURCE_TYPE_GPU_INSTANCE,
    MeteredUsage,
)
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.users import User
from gpustack.server.deps import get_current_user, get_session, get_tenant_context

NOW = datetime(2026, 4, 1, 0, 0, 0)


class _Ctx:
    user = None
    is_platform_admin = True
    current_principal_id = None
    org_role = None
    current_is_personal_scope = False


def _metered(resource_id: int, day: int, creator_id: int):
    return MeteredUsage(
        meter_key=METER_INSTANCE_UPTIME,
        resource_type=RESOURCE_TYPE_GPU_INSTANCE,
        resource_id=resource_id,
        resource_name=f"inst-{resource_id}",
        bucket_start=datetime(2026, 4, day, 0, 0, 0),
        quantity=3600.0 * resource_id,
        unit="seconds",
        sku="flavor-a",
        sku_count=2,
        creator_id=creator_id,
        creator_name=f"user{creator_id}",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(MeteredUsage.__table__.create)
        await conn.run_sync(Principal.__table__.create)
        await conn.run_sync(GPUInstance.__table__.create)

    async with AsyncSession(engine) as seed:
        seed.add(
            Principal(
                id=1,
                kind=PrincipalType.USER,
                name="user1",
                display_name="User One",
                source="local",
                is_admin=False,
                is_active=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        # 3 instances × 3 days.
        for resource_id in (1, 2, 3):
            for day in (1, 2, 3):
                seed.add(_metered(resource_id, day, creator_id=resource_id))
        await seed.commit()

    app = FastAPI()
    register_handlers(app)
    app.include_router(resource_routes.router, prefix="/usage")

    admin = User(id=1, name="admin", is_admin=True)
    ctx = _Ctx()
    ctx.user = admin

    async def _session_dep():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session_dep
    app.dependency_overrides[get_current_user] = lambda: admin
    app.dependency_overrides[get_tenant_context] = lambda: ctx

    @asynccontextmanager
    async def _enrichment_session():
        # Stands in for the app-wide session factory, which is only wired up
        # once the server has initialized the database.
        async with AsyncSession(engine) as session:
            yield session

    with patch.object(resource_routes, "async_session", _enrichment_session):
        yield app, engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app_and_engine):
    app, _ = app_and_engine
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# These exercise the CSV/zip payloads, so they pin the format rather than
# riding on the default (which is xlsx — what users have always received).
RANGE = {
    "start_date": "2026-04-01",
    "end_date": "2026-04-03",
    "scope": "all",
    "format": "csv",
}


def _parse_csv(body: bytes):
    text = body.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


@pytest.mark.asyncio
async def test_export_enriches_names_while_the_cursor_is_open(client):
    """The second connection resolves display names mid-stream.

    If enrichment shared the streaming connection this would raise; if it were
    skipped the name column would be empty.
    """
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["instance"]},
    )

    assert response.status_code == 200, response.text
    rows = _parse_csv(response.content)
    assert len(rows) == 3
    # Every instance was deleted (none exist in gpu_instances), so the flag
    # must be set rather than the row dropped — metered usage is still real.
    assert {r["Instance Deleted"] for r in rows} == {"True"}
    # The owner is tracked separately from the resource.
    assert {r["Owner"] for r in rows} == {"user1", "user2", "user3"}


@pytest.mark.asyncio
async def test_export_matches_the_breakdown_endpoint(client):
    listed = await client.post(
        "/usage/gpu-instances/breakdown",
        json={**RANGE, "group_by": ["instance"], "page": -1},
    )
    exported = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["instance"]},
    )

    assert listed.status_code == 200 and exported.status_code == 200
    items = listed.json()["items"]
    rows = _parse_csv(exported.content)

    assert len(rows) == len(items)
    exported_hours = sorted(round(float(r["GPU Hours"]), 2) for r in rows)
    # The JSON route nests metrics; the export flattens them into columns.
    listed_hours = sorted(round(float(i["metrics"]["gpu_hours"]), 2) for i in items)
    assert exported_hours == listed_hours


@pytest.mark.asyncio
async def test_streaming_survives_a_chunk_boundary(client, monkeypatch):
    monkeypatch.setattr(resource_routes.envs, "USAGE_EXPORT_STREAM_CHUNK_ROWS", 1)
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["instance"]},
    )

    rows = _parse_csv(response.content)
    assert len(rows) == 3
    assert len({r["Instance ID"] for r in rows}) == 3


@pytest.mark.asyncio
async def test_multi_sheet_export_returns_a_readable_zip(client):
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={
            **RANGE,
            "sheets": [
                {"key": "instance", "group_by": ["instance"]},
                {"key": "instance_type", "group_by": ["instance_type"]},
            ],
        },
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    assert archive.namelist() == ["by_instance.csv", "by_instance_type.csv"]
    assert len(_parse_csv(archive.read("by_instance.csv"))) == 3
    # All rows share one sku, so the instance_type sheet collapses to a row.
    assert len(_parse_csv(archive.read("by_instance_type.csv"))) == 1


@pytest.mark.asyncio
async def test_estimate_matches_the_exported_row_count(client):
    estimate = await client.post(
        "/usage/gpu-instances/breakdown/export/estimate",
        json={**RANGE, "group_by": ["instance"]},
    )
    exported = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["instance"]},
    )

    assert estimate.status_code == 200, estimate.text
    assert estimate.json()["total"] == len(_parse_csv(exported.content)) == 3


@pytest.mark.asyncio
async def test_storage_export_rejects_a_gpu_only_dimension(client):
    response = await client.post(
        "/usage/storage/breakdown/export",
        json={**RANGE, "group_by": ["instance_type"]},
    )

    assert response.status_code == 422
    assert "Unsupported group_by" in response.json()["message"]


# --------------------------------------------------------------------------
# Cross-repo contract
#
# ``toResourceExportRequest`` reuses the breakdown body builder, then strips
# pagination and swaps in the export knobs. Its output is flat (ids at the top
# level, not nested under ``filters``) — the opposite of the token payload —
# so the two shapes are pinned separately.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_the_chart_export_payload_the_ui_sends(client):
    """``resource-export-data.tsx`` → single-table export (entries 3/5)."""
    payload = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "scope": "all",
        "granularity": "day",
        "creator_ids": [1],
        "group_by": ["date", "instance"],
    }

    estimate = await client.post(
        "/usage/gpu-instances/breakdown/export/estimate", json=payload
    )
    exported = await client.post("/usage/gpu-instances/breakdown/export", json=payload)

    assert estimate.status_code == 200, estimate.text
    assert exported.status_code == 200, exported.text
    # creator_ids narrowed it to one instance across three days.
    assert estimate.json()["total"] == 3
    # The UI sends no ``format``, so this is the xlsx users have always had.
    assert estimate.json()["effective_format"] == "xlsx"
    assert "xl/workbook.xml" in zipfile.ZipFile(io.BytesIO(exported.content)).namelist()


@pytest.mark.asyncio
async def test_accepts_the_table_export_payload_the_ui_sends(client):
    """``handleExportTable`` → multi-sheet export (entries 4/6).

    ``gpu_type`` is the UI's name for the backend's ``instance_type``. The
    frontend maps BOTH the key and ``group_by`` through ``GROUP_BY_MAP``, so
    the member name is `by_instance_type.csv` — the same vocabulary the Tokens
    tab's `by_route.csv` uses. A file whose names depended on which tab
    produced it would break consumer scripts.
    """
    payload = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "scope": "all",
        "granularity": "day",
        "sheets": [
            {
                "key": "instance_type",
                "group_by": ["instance_type"],
                "name": "实例类型",
            },
            {"key": "instance", "group_by": ["instance"], "name": "实例"},
        ],
    }

    estimate = await client.post(
        "/usage/gpu-instances/breakdown/export/estimate", json=payload
    )
    exported = await client.post("/usage/gpu-instances/breakdown/export", json=payload)

    assert estimate.status_code == 200, estimate.text
    assert exported.status_code == 200, exported.text
    assert "xl/workbook.xml" in zipfile.ZipFile(io.BytesIO(exported.content)).namelist()


@pytest.mark.asyncio
async def test_split_delivers_an_over_limit_resource_export_as_periods(
    client, monkeypatch
):
    monkeypatch.setattr(resource_routes.envs, "USAGE_EXPORT_MAX_ROWS", 4)
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["date", "instance"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    # 3 instances × 3 days = 9 rows over a 4-row limit → 3 periods.
    assert len(archive.namelist()) == 3
    # Splitting must neither drop nor duplicate a row.
    assert sum(len(_parse_csv(archive.read(n))) for n in archive.namelist()) == 9


@pytest.mark.asyncio
async def test_date_grouped_export_comes_out_in_date_order(client, monkeypatch):
    """A time series has to arrive in time order.

    The default sort was the first metric, so every row with the same GB-Days
    was a tie and the database returned them in whatever order it liked — the
    downloaded file's Date column jumped around. Sorting by a flat metric is
    the same as not sorting at all.
    """
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["date", "instance"]},
    )

    assert response.status_code == 200, response.text
    rows = _parse_csv(response.content)
    dates = [row["Date"] for row in rows]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.asyncio
async def test_resource_split_is_csv_even_when_xlsx_was_requested(client, monkeypatch):
    """Same rule as the token side: splitting needs the format that streams."""
    monkeypatch.setattr(resource_routes.envs, "USAGE_EXPORT_MAX_ROWS", 4)
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={
            **RANGE,
            "format": "xlsx",
            "group_by": ["date", "instance"],
            "split": "auto",
        },
    )

    assert response.status_code == 200, response.text
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert all(name.endswith(".csv") for name in names)


@pytest.mark.asyncio
async def test_enrichment_runs_once_per_entity_not_once_per_batch(client, monkeypatch):
    """Enrichment cost must scale with entities, not with rows.

    A ``["date", "instance"]`` export repeats every entity on every date, and
    enrichment ran per 1000-row batch — so a 145k-row export re-resolved the
    same entities 145 times, including a MAX()-per-resource aggregate over
    metered_usage. That is where two minutes went.

    Forcing a tiny batch size makes the repetition visible: with the cache the
    query count stays flat as batches multiply.
    """
    seen = []
    original = resource_routes._dims_by_representative

    async def counting(*args, **kwargs):
        seen.append(kwargs.get("keys"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(resource_routes, "_dims_by_representative", counting)
    monkeypatch.setattr(resource_routes.envs, "USAGE_EXPORT_STREAM_CHUNK_ROWS", 1)

    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "format": "csv", "group_by": ["date", "instance"]},
    )

    assert response.status_code == 200, response.text
    rows = _parse_csv(response.content)
    instances = {row["Instance ID"] for row in rows}
    # One row per batch, so without the cache this would be one lookup per
    # row. It must instead be bounded by the number of distinct instances.
    assert len(rows) > len(instances)
    assert len(seen) <= len(instances)


@pytest.mark.asyncio
async def test_a_dateless_resource_grouping_can_still_be_split(client, monkeypatch):
    """Same rule as the token side: parts are row slices, so time is not
    required. A per-instance export that used to be a dead end now splits."""
    monkeypatch.setattr(resource_routes.envs, "USAGE_EXPORT_MAX_ROWS", 2)
    response = await client.post(
        "/usage/gpu-instances/breakdown/export",
        json={**RANGE, "group_by": ["instance"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    rows = [row for n in archive.namelist() for row in _parse_csv(archive.read(n))]
    # 3 instances, 2 per file, each exactly once across the archive.
    assert len(archive.namelist()) == 2
    assert len({row["Instance"] for row in rows}) == 3


@pytest.mark.asyncio
async def test_estimate_carries_the_columns_the_file_will_have(client):
    response = await client.post(
        "/usage/gpu-instances/breakdown/export/estimate",
        json={**RANGE, "group_by": ["instance"]},
    )

    columns = response.json()["sheets"][0]["columns"]
    assert [c["key"] for c in columns[:3]] == [
        "instance_id",
        "instance_name",
        "instance_deleted",
    ]
    assert [c["title"] for c in columns[:3]] == [
        "Instance ID",
        "Instance",
        "Instance Deleted",
    ]


@pytest.mark.asyncio
async def test_estimate_columns_match_the_exported_header_row(client):
    payload = {**RANGE, "group_by": ["instance"]}
    estimate = await client.post(
        "/usage/gpu-instances/breakdown/export/estimate", json=payload
    )
    exported = await client.post("/usage/gpu-instances/breakdown/export", json=payload)

    promised = [c["title"] for c in estimate.json()["sheets"][0]["columns"]]
    header = exported.content.decode("utf-8-sig").splitlines()[0]
    assert header == ",".join(promised)
