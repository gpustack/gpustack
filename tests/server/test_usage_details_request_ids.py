"""Per-request locating data on the audit row: ``ttft_ms``, ``request_id``
and ``upstream_response_id``.

Run against a real database rather than a stubbed session, because what is
being checked is that reported values survive the whole ingest path onto the
row -- a mocked ``session.add`` would accept them whether or not the column
exists. They are deliberately *not* on ``ModelUsage``: that row aggregates a
day's requests, where an id or a single TTFT would mean nothing.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.schemas.models import Model
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.server.metrics_collector import ModelUsageMetrics, store_usage_metrics

NOW = datetime(2026, 9, 11, 12, 0, 0)
# 1757548800000 ms = 2026-09-11 UTC.
COMPLETED_AT_MS = 1757548800000


@pytest_asyncio.fixture
async def session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for table in (
            Model,
            ModelProvider,
            Principal,
            ApiKey,
            ModelRoute,
            Cluster,
            ModelUsage,
            ModelUsageDetails,
        ):
            await conn.run_sync(table.__table__.create)

    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            Model(
                id=1,
                name="qwen3-0.6b",
                source="huggingface",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        s.add(
            Principal(
                id=7,
                kind=PrincipalType.USER,
                name="alice",
                source="local",
                is_admin=False,
                is_active=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await s.commit()

        @asynccontextmanager
        async def _session():
            yield s

        monkeypatch.setattr("gpustack.server.metrics_collector.async_session", _session)
        yield s
    await engine.dispose()


def _metric(**overrides):
    fields = {
        "model": "qwen3-0.6b",
        "model_id": 1,
        "user_id": 7,
        "input_token": 10,
        "output_token": 20,
        "completed": True,
        "completed_at": COMPLETED_AT_MS,
    }
    fields.update(overrides)
    return ModelUsageMetrics(**fields)


async def _details(session):
    return (await session.exec(select(ModelUsageDetails))).all()


@pytest.mark.asyncio
async def test_reported_ids_and_ttft_land_on_the_audit_row(session):
    metric = _metric(
        ttft_ms=123,
        request_id="3f9c1b2e-77a4-4f21-9a0e-1c5d8e2b4a67",
        upstream_response_id="chatcmpl-Bx1kQZ2v",
    )

    await store_usage_metrics([], [metric])

    (row,) = await _details(session)
    assert row.ttft_ms == 123
    assert row.request_id == "3f9c1b2e-77a4-4f21-9a0e-1c5d8e2b4a67"
    assert row.upstream_response_id == "chatcmpl-Bx1kQZ2v"


@pytest.mark.asyncio
async def test_a_report_carrying_none_of_them_stores_nulls(session):
    """Each is absent in ordinary traffic -- ``ttft_ms`` on every
    non-streaming request, ``upstream_response_id`` whenever the upstream
    mints none -- so absence has to be storable, not a defaulted value that
    reads as reported."""
    await store_usage_metrics([], [_metric()])

    (row,) = await _details(session)
    assert row.ttft_ms is None
    assert row.request_id is None
    assert row.upstream_response_id is None


@pytest.mark.asyncio
async def test_an_oversized_reported_id_is_stored_as_absent(session, caplog):
    """``upstream_response_id`` is copied verbatim out of a third-party
    response body, so nothing bounds it on the way here. The columns are
    VARCHAR(255) on MySQL and OceanBase, which reject an over-long value rather
    than truncating -- and a rejected insert is not one lost row, because
    ``flush_gateway_metrics`` re-buffers the batch, so one such value would
    fail every later flush and stop usage persisting for good."""
    with caplog.at_level(logging.WARNING):
        await store_usage_metrics(
            [],
            [
                _metric(
                    request_id="r" * 300,
                    upstream_response_id="chatcmpl-" + "x" * 300,
                )
            ],
        )

    (row,) = await _details(session)
    assert row.request_id is None
    assert row.upstream_response_id is None
    # The row itself still lands: dropping the id costs the id, not the audit
    # record or every report queued behind it.
    assert row.prompt_token_count == 10
    assert "over the 255 the column holds" in caplog.text


@pytest.mark.asyncio
async def test_an_id_exactly_at_the_limit_is_kept(session):
    """The bound is the column width, not a margin below it."""
    await store_usage_metrics([], [_metric(upstream_response_id="x" * 255)])

    (row,) = await _details(session)
    assert row.upstream_response_id == "x" * 255


@pytest.mark.asyncio
async def test_two_reports_may_share_one_request_id(session):
    """A fallback pass is an internal redirect of one downstream request and
    reports twice, so the column is indexed and never constrained unique. A
    UNIQUE here would drop the second row -- or fail the whole flush."""
    shared = "3f9c1b2e-77a4-4f21-9a0e-1c5d8e2b4a67"

    await store_usage_metrics(
        [],
        [
            _metric(request_id=shared, upstream_response_id="chatcmpl-first"),
            _metric(request_id=shared, upstream_response_id="chatcmpl-second"),
        ],
    )

    rows = await _details(session)
    assert [row.request_id for row in rows] == [shared, shared]
    assert {row.upstream_response_id for row in rows} == {
        "chatcmpl-first",
        "chatcmpl-second",
    }
    assert not any(
        index.unique
        for index in ModelUsageDetails.__table__.indexes
        if "request_id" in index.columns
    )


@pytest.mark.asyncio
async def test_the_daily_rollup_does_not_carry_them(session):
    """``ModelUsage`` aggregates many requests into one row, so a per-request
    id on it could only name an arbitrary one of them."""
    await store_usage_metrics([_metric(request_id="req-1", ttft_ms=5)], [])

    # The report was accepted -- so the absence below is the schema's doing,
    # not a metric that was dropped before it got there.
    assert len((await session.exec(select(ModelUsage))).all()) == 1
    rollup_columns = {c.name for c in ModelUsage.__table__.columns}
    assert not rollup_columns & {"ttft_ms", "request_id", "upstream_response_id"}
