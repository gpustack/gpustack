"""End-to-end export tests against a real database and a real ASGI app.

The unit tests stub the session, so they cannot answer the two questions that
actually decide whether the export works in production:

1. Is the request-scoped session still open while ``StreamingResponse``
   drains the body? The endpoint returns before a single row is written, so a
   dependency closed too early would produce an empty or half file — and only
   in production, never in a mocked test.
2. Does the SQL the builder emits actually run? ``session.stream``, the
   per-batch entity lookups on a second connection, and the aggregate ordering
   are all things a MagicMock will happily accept and a database will not.

These run on SQLite, which is enough to exercise the statements end to end.
"""

import csv
import io
import zipfile
from contextlib import asynccontextmanager
from datetime import date, datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import usage as usage_routes
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.users import User
from gpustack.server.deps import get_current_user, get_session, get_tenant_context

NOW = datetime(2026, 4, 1, 0, 0, 0)


class _Ctx:
    """Platform-admin tenant context: org-wide scope, no pinned Org."""

    user = None
    is_platform_admin = True
    current_principal_id = None
    org_role = None
    current_is_personal_scope = False


# Which tenant each seeded user's API key belongs to. ``consumer_principal_id``
# is denormalized from the key, so this is what a row's Organization resolves
# to: a live Org, an Org since deleted (snapshot name only), and a keyless row
# whose consumer is NULL and therefore Untracked.
_CONSUMERS = {
    1: (10, "acme", "org"),
    2: (11, "gone-org", "org"),
    3: (10, "acme", "org"),
    4: (None, None, None),
    5: (None, None, None),
}


def _usage_row(*, user_id, user_name, route_id, route_name, day, tokens):
    consumer_id, consumer_name, consumer_kind = _CONSUMERS[user_id]
    # A row is only credited to a tenant through its API key; keyless (cookie)
    # traffic has no consumer at all.
    keyed = consumer_id is not None
    return ModelUsage(
        user_id=user_id,
        user_name=user_name,
        model_id=route_id,
        model_name=route_name,
        model_route_id=route_id,
        model_route_name=route_name,
        api_key_id=200 + user_id if keyed else None,
        api_key_name=f"key{user_id}" if keyed else None,
        access_key=f"ak{user_id}" if keyed else None,
        consumer_principal_id=consumer_id,
        consumer_name=consumer_name,
        consumer_principal_kind=consumer_kind,
        api_key_is_custom=False,
        date=day,
        prompt_token_count=tokens,
        completion_token_count=tokens * 2,
        prompt_cached_token_count=0,
        request_count=1,
        operation="chat",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(ModelUsage.__table__.create)
        await conn.run_sync(Principal.__table__.create)
        # The route dimension checks entity existence against this table.
        await conn.run_sync(ModelRoute.__table__.create)
        # ...and the api_key dimension against this one. Left uncreated while
        # every seeded row was keyless; the tenant column is only carried on
        # api_key-grouped rows, so the fixture now has keys and needs it.
        await conn.run_sync(ApiKey.__table__.create)

    async with AsyncSession(engine) as seed:
        seed.add_all(
            [
                Principal(
                    id=1,
                    kind=PrincipalType.USER,
                    name="alice",
                    display_name="Alice",
                    source="local",
                    is_admin=False,
                    is_active=True,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                # Org 10 is live; org 11 deliberately is not, so a row whose
                # tenant was deleted still has to name it from its snapshot.
                Principal(
                    id=10,
                    kind=PrincipalType.ORG,
                    name="acme",
                    display_name="Acme",
                    source="local",
                    is_admin=False,
                    is_active=True,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                # 5 users × 3 days = 15 buckets grouped by (date, user); enough
                # rows to cross the streaming chunk boundary once it is lowered.
                *[
                    _usage_row(
                        user_id=uid,
                        user_name=f"user{uid}",
                        route_id=100 + uid,
                        route_name=f"route{uid}",
                        day=date(2026, 4, day),
                        tokens=uid * 10 + day,
                    )
                    for uid in range(1, 6)
                    for day in range(1, 4)
                ],
            ]
        )
        await seed.commit()

    app = FastAPI()
    register_handlers(app)
    app.include_router(usage_routes.router, prefix="/usage")

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
        # once the server has initialized the database. The export resolves
        # names on a connection of its own while its cursor is open.
        async with AsyncSession(engine) as session:
            yield session

    with patch.object(usage_routes, "async_session", _enrichment_session):
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
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(data_lines))))


@pytest.mark.asyncio
async def test_export_streams_full_result_set_with_live_session(client):
    """The whole file arrives — the session outlives the response object.

    A dependency torn down when the handler returns would give a header-only
    (or truncated) body here, which is the failure mode no mocked test can see.
    """
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    assert response.status_code == 200, response.text
    rows = _parse_csv(response.content)
    # 5 users × 3 days, every bucket present — not one page of them.
    assert len(rows) == 15
    assert response.headers["content-disposition"].startswith("attachment")


@pytest.mark.asyncio
async def test_export_rows_match_the_breakdown_endpoint(client):
    """The file and the table must agree — that is the point of sharing the
    statement builder rather than writing a second query for the export."""
    listed = await client.post(
        "/usage/breakdown",
        json={**RANGE, "group_by": ["user"], "page": -1},
    )
    exported = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["user"]}
    )

    assert listed.status_code == 200 and exported.status_code == 200
    items = listed.json()["items"]
    rows = _parse_csv(exported.content)

    assert len(rows) == len(items)
    by_user_listed = {i["user"]["label"]: i["total_tokens"] for i in items}
    by_user_exported = {r["User"]: int(r["Total Tokens"]) for r in rows}
    assert by_user_exported == by_user_listed


@pytest.mark.asyncio
async def test_streaming_survives_a_chunk_boundary(client, monkeypatch):
    """Rows keep flowing across batches of the server-side cursor.

    With the chunk size at 2 the prefetch/partition path runs many times; an
    off-by-one there would silently drop or duplicate rows.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_STREAM_CHUNK_ROWS", 2)
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    rows = _parse_csv(response.content)
    assert len(rows) == 15
    # No duplicates: (date, user) is the grouping key.
    assert len({(r["Date"], r["User"]) for r in rows}) == 15


@pytest.mark.asyncio
async def test_deleted_entity_is_flagged_not_suffixed(client):
    """Only user id 1 has a live Principal, so the rest are gone.

    Deletion has to arrive as a boolean column; the UI's ``[Deleted.x]`` name
    suffix would corrupt the name field for downstream consumers.
    """
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["user"]}
    )

    rows = {r["User"]: r for r in _parse_csv(response.content)}
    assert rows["user1"]["User Deleted"] == "False"
    assert rows["user2"]["User Deleted"] == "True"
    assert "[Deleted" not in response.content.decode("utf-8-sig")


@pytest.mark.asyncio
async def test_multi_sheet_export_returns_a_readable_zip(client):
    response = await client.post(
        "/usage/breakdown/export",
        json={
            **RANGE,
            "sheets": [
                {"key": "user", "group_by": ["user"]},
                {"key": "route", "group_by": ["route"]},
            ],
        },
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    assert archive.namelist() == ["by_user.csv", "by_route.csv"]
    # Each member is a complete CSV, not a truncated one.
    for name in archive.namelist():
        assert len(_parse_csv(archive.read(name))) == 5


@pytest.mark.asyncio
async def test_xlsx_export_is_a_valid_workbook(client):
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["user"], "format": "xlsx"},
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert "xl/workbook.xml" in archive.namelist()


@pytest.mark.asyncio
async def test_estimate_matches_the_exported_row_count(client):
    estimate = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": ["date", "user"]},
    )
    exported = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    assert estimate.status_code == 200, estimate.text
    payload = estimate.json()
    # The number shown before the click must be the number that comes out.
    assert payload["total"] == len(_parse_csv(exported.content)) == 15
    assert payload["hard_limit"] > 0


@pytest.mark.asyncio
async def test_oversized_export_is_refused_with_structured_details(client, monkeypatch):
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 3)
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    # InvalidException is 422, not 400 — clients keying on 400 would miss it.
    assert response.status_code == 422
    details = response.json()["details"]
    assert details["kind"] == "export_too_large"
    assert details["total"] == 15 and details["limit"] == 3
    assert {s["action"] for s in details["suggestions"]} >= {
        "shorten_range",
        "split_export",
    }


@pytest.mark.asyncio
async def test_ordinary_errors_keep_their_previous_shape(client):
    """``details`` is opt-in: an unrelated error must not grow a null field,
    because ~60 route declarations share this response model."""
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["nonsense"]}
    )

    assert response.status_code == 422
    assert set(response.json()) == {"code", "reason", "message"}


@pytest.mark.asyncio
async def test_trailer_reports_the_row_count_and_effective_scope(client):
    """The trailer is the only signal distinguishing a complete file from one
    truncated mid-stream, and it records the scope that was actually used."""
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["user"]}
    )

    trailer = response.content.decode("utf-8-sig").strip().splitlines()[-1]
    assert trailer.startswith("# rows=5 ")
    assert "scope=all" in trailer


# --------------------------------------------------------------------------
# Cross-repo contract
#
# The UI builds these payloads; nothing in either repo's type system checks
# that the backend accepts them, so drift here fails only at runtime. These
# pin the exact shapes gpustack-ui sends today.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_the_chart_export_payload_the_ui_sends(client):
    """``export-data.tsx`` → ``buildExportRequest`` (entry 1)."""
    payload = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "scope": "all",
        "filters": {
            "users": [
                {
                    "identity": {
                        "value": {"user_name": "user1"},
                        "current": {"user_id": 1},
                    }
                }
            ],
            "routes": [],
            "api_keys": [],
        },
        "granularity": "day",
        "sort_by": "-date",
        "group_by": ["date", "user", "route", "api_key"],
    }

    estimate = await client.post("/usage/breakdown/export/estimate", json=payload)
    exported = await client.post("/usage/breakdown/export", json=payload)

    assert estimate.status_code == 200, estimate.text
    assert exported.status_code == 200, exported.text
    # The user filter narrowed it to one user over three days.
    assert estimate.json()["total"] == 3
    # The UI sends no ``format``, so this is the default users receive: the
    # same xlsx these exports have always produced.
    assert estimate.json()["effective_format"] == "xlsx"
    assert "xl/workbook.xml" in zipfile.ZipFile(io.BytesIO(exported.content)).namelist()


@pytest.mark.asyncio
async def test_accepts_the_table_export_payload_the_ui_sends(client):
    """``use-export-table.tsx`` → multi-sheet export (entry 2)."""
    payload = {
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "scope": "all",
        "filters": {},
        "sheets": [
            {"key": "route", "group_by": ["route"], "name": "模型"},
            {"key": "api_key", "group_by": ["api_key"], "name": "API Key"},
            {"key": "user", "group_by": ["user"], "name": "用户"},
        ],
    }

    estimate = await client.post("/usage/breakdown/export/estimate", json=payload)
    exported = await client.post("/usage/breakdown/export", json=payload)

    assert estimate.status_code == 200, estimate.text
    assert exported.status_code == 200, exported.text
    assert [s["key"] for s in estimate.json()["sheets"]] == [
        "route",
        "api_key",
        "user",
    ]
    # Default format: one workbook with a worksheet per table.
    assert "xl/workbook.xml" in zipfile.ZipFile(io.BytesIO(exported.content)).namelist()


# --------------------------------------------------------------------------
# Split export
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_turns_an_over_limit_export_into_periods(client, monkeypatch):
    """Over the limit stops being a dead end.

    Without this the only answer to "my range is too big" is "make it
    smaller", and operators end up raising the ceiling until something falls
    over.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["date", "user"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].endswith('_split.zip"')
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    # 15 rows over a 6-row limit → 3 periods across a 3-day range.
    assert len(archive.namelist()) == 3
    # Every row still lands in exactly one member: splitting must not drop or
    # duplicate anything.
    exported = sum(len(_parse_csv(archive.read(n))) for n in archive.namelist())
    assert exported == 15


@pytest.mark.asyncio
async def test_date_grouped_export_defaults_to_date_order(client):
    """Same rule on the token side, and for the same reason.

    It happened to work because the dialog sends ``sort_by: "-date"`` — but a
    file's row order must not depend on what the client remembered to ask for.
    """
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["date", "user"]},
    )

    assert response.status_code == 200, response.text
    rows = _parse_csv(response.content)
    dates = [row["Date"] for row in rows]
    assert dates == sorted(dates, reverse=True)
    # A date is a bucket, not a row: five users share each one. Without a
    # tie-break their order is the database's to pick, which shows up as a
    # paginated /breakdown repeating and skipping rows across pages.
    for date_value in set(dates):
        tokens = [int(row["Total Tokens"]) for row in rows if row["Date"] == date_value]
        assert tokens == sorted(tokens, reverse=True)


@pytest.mark.asyncio
async def test_split_is_csv_even_when_xlsx_was_requested(client, monkeypatch):
    """Splitting is the escape hatch for a too-large export, so it has to be
    the format that streams.

    Every part would fit a worksheet, so this is not a format limit — it is a
    throughput one: a workbook must be assembled in full before any of it is
    valid, and a split multiplies that by the part count. Measured on a
    ~108k-row two-part export the download never finished. The estimate says
    ``effective_format: "csv"`` up front so this is announced, not sprung.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "format": "xlsx", "group_by": ["date", "user"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert all(name.endswith(".csv") for name in names)


@pytest.mark.asyncio
async def test_estimate_announces_csv_when_splitting_is_the_way_out(
    client, monkeypatch
):
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    response = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "format": "xlsx", "group_by": ["date", "user"]},
    )

    payload = response.json()
    assert payload["exceeds_hard_limit"] is True
    assert payload["split_parts"] == 3
    # The remedy the dialog is about to offer produces CSV; say so before the
    # click rather than letting a .zip of .csv arrive unannounced.
    assert payload["effective_format"] == "csv"


@pytest.mark.asyncio
async def test_split_member_names_sort_into_reading_order(client, monkeypatch):
    """Names are zero-padded and carry ``of-N``.

    A part is a row slice, so the name deliberately does not claim a date
    range — the rows inside are date-ordered and the trailer records the row
    range. What it must do is sort correctly and let a consumer see whether
    they have the whole set.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 2)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["date", "user"], "split": "auto"},
    )

    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert len(names) == 8  # 15 rows / 2
    # Plain lexical sort === part order, which needs the zero padding.
    assert names == sorted(names)
    assert names[0].endswith("part-1-of-8.csv")
    assert names[-1].endswith("part-8-of-8.csv")


@pytest.mark.asyncio
async def test_split_keeps_table_and_period_on_separate_axes(client, monkeypatch):
    """N tables × K periods must not be flattened into one name."""
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    response = await client.post(
        "/usage/breakdown/export",
        json={
            **RANGE,
            "split": "auto",
            "sheets": [
                {"key": "user", "group_by": ["date", "user"]},
                {"key": "route", "group_by": ["date", "route"]},
            ],
        },
    )

    assert response.status_code == 200, response.text
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert all(n.startswith(("by_user/", "by_route/")) for n in names)
    assert len([n for n in names if n.startswith("by_user/")]) == 3


@pytest.mark.asyncio
async def test_a_dateless_grouping_can_still_be_split(client, monkeypatch):
    """Splitting is not a time operation.

    Parts are slices of the already-aggregated ROW STREAM, so there is nothing
    to re-aggregate and nothing to double-count — which is what used to make a
    ``group_by: ["user"]`` export unsplittable. Every grouping gets the escape
    hatch now.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 2)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["user"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    rows = [row for n in archive.namelist() for row in _parse_csv(archive.read(n))]
    # Five users, two per file, and each user appears exactly once overall —
    # the property a time-based split could not offer.
    assert len(archive.namelist()) == 3
    assert len(rows) == 5
    assert len({row["User"] for row in rows}) == 5


@pytest.mark.asyncio
async def test_estimate_offers_split_for_every_grouping(client, monkeypatch):
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 2)
    response = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": ["user"]},
    )

    payload = response.json()
    assert payload["split_parts"] == 3
    assert "split_export" in {s["action"] for s in payload["suggestions"]}


@pytest.mark.asyncio
async def test_split_periods_each_fit_the_limit(client, monkeypatch):
    """End to end: no part comes back over the limit.

    The plan is built from the real per-bucket counts, so this is the property
    that matters — not how many parts it took to get there.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["date", "user"], "split": "auto"},
    )

    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    for name in archive.namelist():
        assert len(_parse_csv(archive.read(name))) <= 6
    # And nothing is lost or duplicated across the parts.
    assert sum(len(_parse_csv(archive.read(n))) for n in archive.namelist()) == 15


@pytest.mark.asyncio
async def test_split_delivers_exactly_the_promised_number_of_files(client, monkeypatch):
    """The estimate's ``split_parts`` is the file count, not an approximation.

    Row slices make ``ceil(total / limit)`` exact. The date-range splitter it
    replaced could only give a lower bound — uneven usage needed more slices
    than a perfect division — so the button promised a number the download
    then exceeded.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 4)
    payload = {**RANGE, "group_by": ["date", "user"]}

    estimate = (
        await client.post("/usage/breakdown/export/estimate", json=payload)
    ).json()
    exported = await client.post(
        "/usage/breakdown/export", json={**payload, "split": "auto"}
    )

    assert estimate["split_parts"] == 4  # 15 rows / 4
    names = zipfile.ZipFile(io.BytesIO(exported.content)).namelist()
    assert len(names) == estimate["split_parts"]
    assert names[0].endswith("part-1-of-4.csv")


@pytest.mark.asyncio
async def test_split_parts_counts_every_sheets_files_not_just_the_biggest(
    client, monkeypatch
):
    """A multi-sheet split writes parts for EVERY sheet, so the promise must
    count them all.

    ``split_parts`` was derived from the largest sheet alone, on the same
    reasoning that makes the over-limit VERDICT per-sheet — but the two
    questions have opposite shapes: whether to refuse is about the biggest
    table, while how many files come back is about their sum. A user offered
    "Export in 3 files" received 6.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    payload = {
        **RANGE,
        "sheets": [
            {"key": "user", "group_by": ["date", "user"]},
            {"key": "route", "group_by": ["date", "route"]},
        ],
    }

    estimate = (
        await client.post("/usage/breakdown/export/estimate", json=payload)
    ).json()
    exported = await client.post(
        "/usage/breakdown/export", json={**payload, "split": "auto"}
    )

    names = zipfile.ZipFile(io.BytesIO(exported.content)).namelist()
    assert len(names) == estimate["split_parts"]
    # Both sheets contributed; the count is not one sheet's answer reused.
    assert any(n.startswith("by_user/") for n in names)
    assert any(n.startswith("by_route/") for n in names)
    # The remedy button carries the same number as the estimate field.
    split = next(s for s in estimate["suggestions"] if s["action"] == "split_export")
    assert split["parts"] == estimate["split_parts"]


@pytest.mark.asyncio
async def test_no_split_remedy_is_offered_when_the_split_would_be_refused(
    client, monkeypatch
):
    """An offered remedy must be one the next request accepts.

    The cap on file count lives in the split path, which the estimate never
    consulted — so an export too large to split still advertised "Export in N
    files", and the click came back 422. Withholding the button leaves the one
    remedy that does work.
    """
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 1)
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_SPLIT_MEMBERS", 2)
    payload = {**RANGE, "group_by": ["date", "user"]}

    estimate = (
        await client.post("/usage/breakdown/export/estimate", json=payload)
    ).json()

    assert estimate["exceeds_hard_limit"] is True
    assert estimate["split_parts"] is None
    actions = [s["action"] for s in estimate["suggestions"]]
    assert actions == ["shorten_range"]
    # And the refusal the user would have hit is still there for anyone who
    # asks for a split directly.
    refused = await client.post(
        "/usage/breakdown/export", json={**payload, "split": "auto"}
    )
    assert refused.status_code == 422


@pytest.mark.asyncio
async def test_split_refuses_to_produce_an_unbounded_number_of_files(
    client, monkeypatch
):
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 1)
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_SPLIT_MEMBERS", 2)
    response = await client.post(
        "/usage/breakdown/export",
        json={**RANGE, "group_by": ["date", "user"], "split": "auto"},
    )

    assert response.status_code == 422
    assert response.json()["details"]["kind"] == "export_split_too_many_parts"


@pytest.mark.asyncio
async def test_estimate_offers_the_same_remedies_the_error_would(client, monkeypatch):
    """The advice before the click and after a rejection come from one helper,
    so they cannot disagree."""
    monkeypatch.setattr(usage_routes.envs, "USAGE_EXPORT_MAX_ROWS", 6)
    estimated = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": ["date", "user"]},
    )
    rejected = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    assert estimated.status_code == 200 and rejected.status_code == 422
    payload = estimated.json()
    assert payload["suggestions"] == rejected.json()["details"]["suggestions"]
    assert payload["split_parts"] == 3
    assert payload["suggested_max_days"] >= 1


@pytest.mark.asyncio
async def test_estimate_reports_no_remedies_when_the_export_fits(client):
    response = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": ["date", "user"]},
    )

    payload = response.json()
    assert payload["suggestions"] == []
    assert payload["split_parts"] is None


@pytest.mark.asyncio
async def test_estimate_carries_the_columns_the_file_will_have(client):
    """The preview renders from this list.

    Sending it — instead of letting the client derive its own — is what keeps
    the dialog's table and the downloaded file describing the same thing. The
    keys are how the preview reads a value; the titles are the file's header.
    """
    response = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": ["date", "user", "route", "api_key"]},
    )

    columns = response.json()["sheets"][0]["columns"]
    assert [c["key"] for c in columns[:4]] == [
        "date",
        "route_id",
        "route_name",
        "route_deleted",
    ]
    # Titles are readable and fixed; Model, not the internal "route".
    assert [c["title"] for c in columns[:4]] == [
        "Date",
        "Model ID",
        "Model",
        "Model Deleted",
    ]


@pytest.mark.asyncio
async def test_estimate_columns_match_the_exported_header_row(client):
    """Whatever the estimate promises, the file must deliver."""
    payload = {**RANGE, "group_by": ["date", "user"]}
    estimate = await client.post("/usage/breakdown/export/estimate", json=payload)
    exported = await client.post("/usage/breakdown/export", json=payload)

    promised = [c["title"] for c in estimate.json()["sheets"][0]["columns"]]
    header = exported.content.decode("utf-8-sig").splitlines()[0]
    assert header == ",".join(promised)


@pytest.mark.asyncio
async def test_a_row_with_no_api_key_leaves_its_columns_empty(client):
    """Keyless (cookie-authenticated) usage has no API key to name.

    A ``-`` placeholder is a UI convention, and ``FALSE`` would assert that
    something which never existed has not been deleted. Both mislead whoever
    reads the file, so the cells stay empty — and empty has to survive
    alongside rows that DO carry a key, which is why this asserts on both.
    """
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["user", "api_key"]}
    )

    rows = {row["User"]: row for row in _parse_csv(response.content)}
    keyless = rows["user4"]
    assert keyless["API Key ID"] == ""
    assert keyless["API Key"] == ""
    assert keyless["API Key Deleted"] == ""
    # Keyed rows name theirs (the label is "<owner> / <key>").
    assert rows["user1"]["API Key"] == "user1 / key1"


CHART_GROUP_BY = ["date", "user", "route", "api_key"]


@pytest.mark.asyncio
async def test_all_view_names_the_tenant_each_row_belongs_to(client):
    """Cross-tenant token rows must say whose they are.

    Two organizations can each have an ``ops`` user calling a shared model.
    Without the tenant those rows differ only in their numbers, and attributing
    spend is what these files are for.

    Carried as an attribute, not a grouping: ``consumer_principal_id`` is
    denormalized from the API key, so an api_key-grouped row already has
    exactly one tenant and naming it moves no rows.
    """
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": CHART_GROUP_BY}
    )

    assert response.status_code == 200, response.text
    rows = {row["User"]: row for row in _parse_csv(response.content)}

    assert rows["user1"]["Organization"] == "acme"
    assert rows["user1"]["Organization ID"] == "10"
    assert rows["user1"]["Organization Type"] == "org"
    assert rows["user1"]["Organization Deleted"] == "False"
    # A deleted tenant keeps its snapshot name; dropping the row would lose
    # real, billable usage.
    assert rows["user2"]["Organization"] == "gone-org"
    assert rows["user2"]["Organization Deleted"] == "True"
    # Keyless traffic has no consumer — un-attributed, not missing.
    assert rows["user4"]["Organization"] == "Untracked"


@pytest.mark.asyncio
async def test_a_grouping_without_api_key_omits_the_tenant(client):
    """Only the API key pins a row to one tenant.

    A user can hold keys in several Orgs and a model can be shared across
    them, so for those groupings ``MAX(id)`` and ``MAX(name)`` could come from
    different tenants — pairing one Org's id with another's name. A wrong
    attribution is worse than an absent one.
    """
    response = await client.post(
        "/usage/breakdown/export", json={**RANGE, "group_by": ["date", "user"]}
    )

    header = response.content.decode("utf-8-sig").splitlines()[0]
    assert "Organization" not in header


@pytest.mark.asyncio
async def test_the_tenant_column_follows_the_grouping_columns(client):
    """Grouping columns identify a row; attribute columns describe it.

    Reusing the dimension loop would have put the tenant FIRST (it is the
    broadest dimension, and that is where it belongs when it IS the grouping).
    As an attribute it trails them instead — the same place the resource
    export puts the owner and the tenant.
    """
    response = await client.post(
        "/usage/breakdown/export/estimate",
        json={**RANGE, "group_by": CHART_GROUP_BY},
    )

    keys = [c["key"] for c in response.json()["sheets"][0]["columns"]]
    assert keys[0] == "date"
    assert keys.index("organization_id") > keys.index("api_key_deleted")
    assert keys[keys.index("organization_id") : keys.index("organization_id") + 4] == [
        "organization_id",
        "organization_name",
        "organization_kind",
        "organization_deleted",
    ]


@pytest.mark.asyncio
async def test_the_preview_can_read_the_tenant_off_a_breakdown_item(client):
    """The preview reads values from ``/breakdown``, columns from the estimate.

    The attribute reuses the item's existing ``organization`` field — the same
    shape the Organization grouping produces — so the client reads one
    organization one way in either role and needs no second code path. This
    pins that shape: an attribute delivered under some new field would render
    as empty preview cells while the file itself looked fine.
    """
    response = await client.post(
        "/usage/breakdown", json={**RANGE, "group_by": CHART_GROUP_BY, "page": -1}
    )

    items = {item["user"]["label"]: item for item in response.json()["items"]}
    org = items["user1"]["organization"]
    assert org["label"] == "acme"
    assert org["deleted"] is False
    assert org["identity"]["current"]["organization_id"] == 10
    assert org["identity"]["value"]["organization_kind"] == "org"
    # Untracked carries no identity at all (the field is omitted, not null),
    # which the preview's optional chaining already handles — it reads the
    # label and the flag and asks for nothing else.
    untracked = items["user4"]["organization"]
    assert untracked.get("identity") is None
    assert untracked["label"] == "Untracked"
    assert untracked["deleted"] is False
