import io
import zipfile
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import ForbiddenException, InvalidException
from gpustack.routes import usage as usage_routes
from gpustack.routes.usage import (
    estimate_usage_breakdown_export,
    export_usage_breakdown,
)
from gpustack.schemas.usage import (
    UsageBreakdownDateDimension,
    UsageBreakdownDimension,
    UsageBreakdownItem,
    UsageExportRequest,
    UsageIdentity,
    UsageIdentityCurrent,
    UsageIdentityValue,
)
from gpustack.schemas.users import User
from gpustack.utils.usage_export import (
    UNTRACKED_ORGANIZATION_NAME,
    build_export_columns,
    export_column_keys,
    export_row,
)


def _mock_exec_result(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


def _ctx_for(user, principal_id=None):
    ctx = MagicMock()
    ctx.user = user
    ctx.is_platform_admin = bool(getattr(user, "is_admin", False))
    ctx.current_principal_id = (
        principal_id
        if principal_id is not None
        else (None if ctx.is_platform_admin else 1)
    )
    ctx.org_role = None
    ctx.current_is_personal_scope = False
    return ctx


class _StreamResult:
    """Stand-in for the AsyncResult returned by ``session.stream``."""

    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    async def partitions(self, size):
        for start in range(0, len(self._rows), size):
            yield self._rows[start : start + size]

    async def close(self):
        self.closed = True


def _session(*, counts, stream_rows):
    """Request-session stub.

    ``session.exec`` answers the sizing pass — every sheet is counted before a
    byte is written, since the status code is committed once streaming starts
    — and ``session.stream`` hands out one cursor per sheet. Entity lookups do
    NOT land here: they run on the enrichment session (see ``_enrichment``),
    because this connection is holding the cursor open.
    """
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[_mock_exec_result([count]) for count in counts]
    )
    session.stream = AsyncMock(
        side_effect=[_StreamResult(rows) for rows in stream_rows]
    )
    return session


@pytest.fixture(autouse=True)
def _enrichment():
    """Stand in for the app-wide session factory the enrichment borrows.

    Every entity resolves to "gone", which is what the export does with an id
    it cannot find; no test here asserts on names or deletion flags, and the
    ones that do run against a real database in the integration module.
    """
    enrich_session = MagicMock()
    enrich_session.exec = AsyncMock(return_value=_mock_exec_result([]))

    @asynccontextmanager
    async def factory():
        yield enrich_session

    with patch.object(usage_routes, "async_session", factory):
        yield enrich_session


async def _collect(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    return b"".join(chunks)


def _admin():
    return User(id=1, name="admin", is_admin=True)


def _date_rows():
    return [
        SimpleNamespace(group_date=date(2026, 4, 1), total_tokens=500, api_requests=5),
        SimpleNamespace(group_date=date(2026, 4, 2), total_tokens=300, api_requests=3),
    ]


def _route_rows():
    return [
        SimpleNamespace(
            group_model_route_id=11,
            group_model_route_name="qwen3",
            total_tokens=500,
            api_requests=5,
        ),
    ]


def _user_rows():
    return [
        SimpleNamespace(
            group_user_id=7, group_user_name="alice", total_tokens=300, api_requests=3
        ),
    ]


# --------------------------------------------------------------------------
# Column layout
# --------------------------------------------------------------------------


def test_entity_dimension_expands_to_id_name_deleted_columns():
    # The export must not ship a single pre-formatted label: scripts join on
    # the id, and deletion has to be a boolean rather than a name suffix.
    keys = export_column_keys(["user"])
    assert keys[:3] == ["user_id", "user_name", "user_deleted"]
    assert "[Deleted" not in "".join(keys)


def test_organization_carries_kind_column():
    keys = export_column_keys(["organization"])
    assert keys[:4] == [
        "organization_id",
        "organization_name",
        "organization_kind",
        "organization_deleted",
    ]


def test_headers_are_machine_names_not_translations():
    """The header row is part of the file's contract.

    Localizing it would make the shape depend on the viewer's language, so a
    customer's reconciliation script would break the day someone switches the
    UI to another language — the same failure the zip member names avoid.
    """
    # Keys are the machine contract the preview reads by...
    assert export_column_keys(["user"])[:3] == [
        "user_id",
        "user_name",
        "user_deleted",
    ]
    # ...and the header row is a fixed English title, not a translation.
    assert build_export_columns(["user"])[:3] == ["User ID", "User", "User Deleted"]
    # The product calls these Models; "route" is internal and never shown.
    assert "Model" in build_export_columns(["route"])


def test_single_group_by_adds_derived_metric_columns():
    keys = export_column_keys(["user"])
    assert "models_called" in keys and "api_keys_used" in keys
    # A compound grouping doesn't populate them, so they must not be columns.
    compound = export_column_keys(["date", "user"])
    assert "models_called" not in compound


def test_export_row_matches_column_order():
    item = UsageBreakdownItem(
        input_tokens=10,
        output_tokens=5,
        input_cached_tokens=1,
        total_tokens=15,
        api_requests=2,
        avg_tokens_per_request=7.5,
        user=UsageBreakdownDimension(
            identity=UsageIdentity(
                value=UsageIdentityValue(user_name="alice"),
                current=UsageIdentityCurrent(user_id=7),
            ),
            label="alice",
            deleted=False,
        ),
        date=UsageBreakdownDateDimension(value=date(2026, 4, 1), label="2026-04-01"),
    )
    columns = export_column_keys(["date", "user"])
    row = export_row(item, ["date", "user"])

    assert len(row) == len(columns)
    assert len(build_export_columns(["date", "user"])) == len(columns)
    values = dict(zip(columns, row))
    assert values["date"] == date(2026, 4, 1)
    assert values["user_id"] == 7
    assert values["user_name"] == "alice"
    assert values["user_deleted"] is False
    assert values["total_tokens"] == 15


def test_untracked_organization_row_is_named_not_blank():
    # A NULL consumer principal is un-attributed direct traffic. It is real
    # usage, so it must appear with an explicit name — a blank cell reads as
    # missing data and dropping the row would break the file's totals.
    item = UsageBreakdownItem(
        total_tokens=42,
        organization=UsageBreakdownDimension(
            identity=None, label="Untracked", deleted=False
        ),
    )
    columns = export_column_keys(["organization"])
    values = dict(zip(columns, export_row(item, ["organization"])))

    assert values["organization_id"] is None
    assert values["organization_name"] == UNTRACKED_ORGANIZATION_NAME
    assert values["total_tokens"] == 42


# --------------------------------------------------------------------------
# Export endpoint
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_sheet_streams_csv_with_bom_and_trailer():
    user = _admin()
    session = _session(counts=[2], stream_rows=[_date_rows()])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date"],
        format="csv",
    )

    response = await export_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )
    body = await _collect(response)

    assert response.media_type.startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    # The BOM is what makes Excel read UTF-8 on a double-click.
    assert body.startswith(b"\xef\xbb\xbf")
    text = body.decode("utf-8-sig")
    assert text.splitlines()[0].startswith("Date,")
    # The trailer is the only way a consumer can tell a complete file from one
    # truncated mid-stream, since the status code is already 200 by then.
    assert text.strip().splitlines()[-1].startswith("# rows=2 scope=all")


@pytest.mark.asyncio
async def test_multi_sheet_csv_is_zipped_with_stable_member_names():
    user = _admin()
    session = _session(
        counts=[2, 2],
        stream_rows=[_route_rows(), _user_rows()],
    )
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        format="csv",
        sheets=[
            {"key": "route", "group_by": ["route"], "name": "模型"},
            {"key": "user", "group_by": ["user"], "name": "用户"},
        ],
    )

    response = await export_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )
    body = await _collect(response)

    archive = zipfile.ZipFile(io.BytesIO(body))
    assert archive.testzip() is None
    # Members are named by the stable key, not the localized display name, so
    # a consumer script survives a UI language change.
    assert archive.namelist() == ["by_route.csv", "by_user.csv"]


@pytest.mark.asyncio
async def test_oversized_sheet_is_rejected_with_actionable_details(monkeypatch):
    # 183 rows at 25 per file is 8 parts, inside USAGE_EXPORT_MAX_SPLIT_MEMBERS
    # — this test is about which remedies come back, not about the file cap.
    monkeypatch.setattr("gpustack.routes.usage.envs.USAGE_EXPORT_MAX_ROWS", 25)
    user = _admin()
    session = _session(counts=[183], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        group_by=["date"],
        granularity="day",
    )

    with pytest.raises(InvalidException) as excinfo:
        await export_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )

    details = excinfo.value.details
    assert details["kind"] == "export_too_large"
    assert details["total"] == 183 and details["limit"] == 25
    actions = {item["action"] for item in details["suggestions"]}
    # "Narrow the range" alone is unactionable; the client needs the numbers.
    # Exactly these two: both change how the rows are delivered, never which
    # rows exist. A month-bucket option used to sit alongside them, which put
    # a lossy choice one click away from two lossless ones.
    assert actions == {"shorten_range", "split_export"}
    shorten = next(
        item for item in details["suggestions"] if item["action"] == "shorten_range"
    )
    assert 1 <= shorten["max_days"] <= 90


@pytest.mark.asyncio
async def test_xlsx_falls_back_to_csv_beyond_the_worksheet_row_limit(monkeypatch):
    """A worksheet holds ~1M rows; past that the data still has to come out.

    Refusing would leave the user with no way to get their rows at all, while
    the same data in CSV loses nothing — so the format gives, not the export.
    The extension announces the change and the estimate reports it up front.
    """
    monkeypatch.setattr("gpustack.routes.usage.envs.XLSX_MAX_ROWS_PER_SHEET", 5)
    user = _admin()
    session = _session(counts=[9], stream_rows=[_date_rows()])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date"],
        format="xlsx",
    )

    response = await export_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.media_type.startswith("text/csv")
    assert response.headers["content-disposition"].endswith('.csv"')


@pytest.mark.asyncio
async def test_export_refuses_instead_of_silently_downgrading_scope():
    # The list endpoint downgrades an unauthorized "all" to "self" so the page
    # still renders. An export must not: the file leaves the page, and nothing
    # in it would reveal that "platform-wide usage" is one person's rows.
    user = User(id=2, name="member", is_admin=False)
    session = _session(counts=[1], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date"],
        scope="all",
    )

    with pytest.raises(ForbiddenException):
        await export_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_default_scope_exports_the_callers_own_rows():
    # ``scope`` defaults to "all", so a client that never sets it is not
    # asking for platform-wide data — every regular user's export sends
    # exactly this payload. Refusing it would leave non-admins unable to
    # export their own usage at all; the file records the scope it got.
    user = User(id=2, name="member", is_admin=False)
    session = _session(counts=[2], stream_rows=[_date_rows()])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date"],
        format="csv",
    )

    response = await export_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )
    text = (await _collect(response)).decode("utf-8-sig")

    assert "scope=self" in text.strip().splitlines()[-1]


@pytest.mark.asyncio
async def test_export_rejects_whole_request_when_a_sheet_is_forbidden():
    # Dropping just the forbidden sheet would hand back a file the user
    # believes is complete — the same class of bug as a truncated stream.
    user = User(id=2, name="member", is_admin=False)
    session = _session(counts=[], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="self",
        sheets=[
            {"key": "route", "group_by": ["route"]},
            {"key": "organization", "group_by": ["organization"]},
        ],
    )

    with pytest.raises(ForbiddenException):
        await export_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_too_many_sheets_is_rejected(monkeypatch):
    monkeypatch.setattr("gpustack.routes.usage.envs.USAGE_EXPORT_MAX_SHEETS", 1)
    user = _admin()
    session = _session(counts=[], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        sheets=[
            {"key": "route", "group_by": ["route"]},
            {"key": "user", "group_by": ["user"]},
        ],
    )

    with pytest.raises(InvalidException):
        await export_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


# --------------------------------------------------------------------------
# Estimate endpoint
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_returns_per_sheet_totals_and_limits():
    user = _admin()
    session = _session(counts=[8200, 15400], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        sheets=[
            {"key": "route", "group_by": ["route"]},
            {"key": "api_key", "group_by": ["api_key"]},
        ],
    )

    response = await estimate_usage_breakdown_export(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert [sheet.total for sheet in response.sheets] == [8200, 15400]
    assert response.total == 23600
    # Echoed so the UI doesn't hardcode thresholds and support can read the
    # deployment's values straight off a bug report.
    assert response.hard_limit > 0 and response.soft_limit > 0


@pytest.mark.asyncio
async def test_estimate_judges_the_limit_per_sheet_not_on_the_sum(monkeypatch):
    """Four 30k tables are not one 120k table.

    Each sheet is its own query and its own worksheet, and the export endpoint
    checks them one at a time. The estimate has to agree: a client comparing
    the SUM against the hard limit refuses exports the server would run, which
    is the exact drift these server-side verdicts exist to prevent.
    """
    monkeypatch.setattr("gpustack.routes.usage.envs.USAGE_EXPORT_MAX_ROWS", 100000)
    monkeypatch.setattr("gpustack.routes.usage.envs.USAGE_EXPORT_SOFT_ROWS", 10000)
    user = _admin()
    session = _session(counts=[30000, 30000, 30000, 30000], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        sheets=[
            {"key": key, "group_by": [key]}
            for key in ("route", "api_key", "user", "organization")
        ],
    )

    response = await estimate_usage_breakdown_export(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    # The sum is still reported — it IS the file's row count, and the dialog
    # says "about N rows will be exported".
    assert response.total == 120000
    assert response.exceeds_hard_limit is False
    assert response.suggestions == [] and response.split_parts is None
    # Each table on its own is past the soft threshold, so the "this will take
    # a while" warning is right here — it tracks the same per-sheet number.
    assert response.exceeds_soft_limit is True

    # And the soft verdict follows the same rule in the other direction: four
    # 5k tables sum past 10k without any single one being slow.
    session = _session(counts=[5000, 5000, 5000, 5000], stream_rows=[])
    response = await estimate_usage_breakdown_export(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )
    assert response.total == 20000
    assert response.exceeds_soft_limit is False


@pytest.mark.asyncio
async def test_estimate_marks_forbidden_sheet_unavailable_instead_of_failing():
    # Unlike the export, the estimate degrades per sheet so the UI can grey out
    # just the Organization table rather than blocking the whole dialog.
    user = User(id=2, name="member", is_admin=False)
    session = _session(counts=[12], stream_rows=[])
    request = UsageExportRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        scope="self",
        sheets=[
            {"key": "route", "group_by": ["route"]},
            {"key": "organization", "group_by": ["organization"]},
        ],
    )

    response = await estimate_usage_breakdown_export(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    by_key = {sheet.key: sheet for sheet in response.sheets}
    assert by_key["route"].available is True
    assert by_key["organization"].available is False
    assert by_key["organization"].reason
    assert response.total == 12


# --------------------------------------------------------------------------
# Entity-existence lookups
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existence_lookup_is_chunked_under_the_bind_parameter_cap():
    """``IN (...)`` spends one bind parameter per id.

    PostgreSQL's wire protocol caps a statement at 65535 of them, and a
    breakdown grouped by a single entity dimension has as many distinct ids as
    it has rows — so an unchunked lookup turns a large export into a hard
    failure instead of a slow one.
    """
    from gpustack.routes.usage import _existing_entity_ids, _ID_LOOKUP_CHUNK
    from gpustack.schemas.principals import Principal

    ids = list(range(1, 12_001))
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(list(range(1, 5_001))),
            _mock_exec_result(list(range(5_001, 10_001))),
            _mock_exec_result(list(range(10_001, 12_001))),
        ]
    )

    existing = await _existing_entity_ids(session, Principal, ids)

    assert _ID_LOOKUP_CHUNK <= 65_535
    assert session.exec.await_count == 3
    # Chunking must not change the answer — every id still accounted for.
    assert existing == set(ids)


@pytest.mark.asyncio
async def test_existence_lookup_skips_the_query_when_there_is_nothing_to_check():
    from gpustack.routes.usage import _existing_entity_ids
    from gpustack.schemas.principals import Principal

    session = MagicMock()
    session.exec = AsyncMock()

    assert await _existing_entity_ids(session, Principal, [None, None]) == set()
    session.exec.assert_not_awaited()
