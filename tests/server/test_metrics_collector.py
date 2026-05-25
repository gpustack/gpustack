import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.model_usage import ModelUsage
from gpustack.server.metrics_collector import (
    ModelUsageMetrics,
    _estimate_partial_usage,
    _make_buffer_key,
    _resolve_usage_tokens,
    accumulate_gateway_metrics,
    create_or_update_model_usage,
    gateway_details_buffer,
    gateway_metrics_buffer,
)


@pytest.fixture(autouse=True)
def clear_buffer():
    gateway_metrics_buffer.clear()
    gateway_details_buffer.clear()
    yield
    gateway_metrics_buffer.clear()
    gateway_details_buffer.clear()


def test_model_usage_metrics_defaults():
    m = ModelUsageMetrics(model="qwen3-0.6b")
    assert m.input_token == 0
    assert m.output_token == 0
    assert m.total_token == 0
    assert m.input_cached_token == 0
    assert m.request_count == 1
    assert m.user_id is None
    assert m.model_id is None
    assert m.provider_id is None
    assert m.access_key is None


# 1700000000000 ms = 2023-11-14 UTC; pin completed_at so the date segment
# of the buffer key is deterministic.
_FIXED_COMPLETED_AT_MS = 1700000000000
_FIXED_DATE_ISO = "2023-11-14"


def test_make_buffer_key():
    m = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        access_key="abc",
        model_route_id=9,
        completed_at=_FIXED_COMPLETED_AT_MS,
    )
    # Segment order: model_id, provider_id, model, user_id, access_key,
    # organization_id, model_route_id, operation, date.
    assert _make_buffer_key(m) == f"1..qwen3-0.6b.2.abc..9..{_FIXED_DATE_ISO}"


def test_make_buffer_key_none_fields():
    m = ModelUsageMetrics(model="qwen3-0.6b", completed_at=_FIXED_COMPLETED_AT_MS)
    assert _make_buffer_key(m) == f"..qwen3-0.6b......{_FIXED_DATE_ISO}"


def test_make_buffer_key_route_separates_otherwise_identical_metrics():
    """Different routes must split into different rollup keys even when
    user / model / api_key / date all match — otherwise route-grouped
    breakdowns would merge unrelated traffic."""
    base = dict(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        access_key="abc",
        completed_at=_FIXED_COMPLETED_AT_MS,
    )
    assert _make_buffer_key(
        ModelUsageMetrics(**base, model_route_id=1)
    ) != _make_buffer_key(ModelUsageMetrics(**base, model_route_id=2))


def test_make_buffer_key_organization_separates_otherwise_identical_metrics():
    """Same user / api_key / route called from two Org contexts within
    one flush window must stay separate — the DB upsert key in
    ``create_or_update_model_usage`` includes ``consumer_principal_id``,
    so merging in memory and splitting on write would lose tokens."""
    base = dict(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        access_key="abc",
        model_route_id=9,
        completed_at=_FIXED_COMPLETED_AT_MS,
    )
    assert _make_buffer_key(
        ModelUsageMetrics(**base, organization_id="1")
    ) != _make_buffer_key(ModelUsageMetrics(**base, organization_id="2"))


def test_accumulate_new_entry():
    m = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        input_token=100,
        output_token=200,
        input_cached_token=60,
        request_count=1,
    )
    asyncio.run(accumulate_gateway_metrics([m]))
    assert len(gateway_metrics_buffer) == 1
    entry = list(gateway_metrics_buffer.values())[0]
    assert entry.input_token == 100
    assert entry.output_token == 200
    assert entry.input_cached_token == 60
    assert entry.request_count == 1


def test_accumulate_same_key_sums_values():
    m1 = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        input_token=100,
        output_token=200,
        input_cached_token=60,
        request_count=1,
    )
    m2 = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        input_token=50,
        output_token=80,
        input_cached_token=15,
        request_count=1,
    )
    asyncio.run(accumulate_gateway_metrics([m1]))
    asyncio.run(accumulate_gateway_metrics([m2]))
    assert len(gateway_metrics_buffer) == 1
    entry = list(gateway_metrics_buffer.values())[0]
    assert entry.input_token == 150
    assert entry.output_token == 280
    assert entry.input_cached_token == 75
    assert entry.request_count == 2


def test_accumulate_different_keys():
    m1 = ModelUsageMetrics(model="qwen3-0.6b", model_id=1, user_id=2, input_token=100)
    m2 = ModelUsageMetrics(model="qwen3-0.6b", model_id=1, user_id=3, input_token=50)
    asyncio.run(accumulate_gateway_metrics([m1, m2]))
    assert len(gateway_metrics_buffer) == 2


def test_accumulate_total_token_summed():
    m1 = ModelUsageMetrics(model="m", model_id=1, total_token=300)
    m2 = ModelUsageMetrics(model="m", model_id=1, total_token=100)
    asyncio.run(accumulate_gateway_metrics([m1]))
    asyncio.run(accumulate_gateway_metrics([m2]))
    entry = list(gateway_metrics_buffer.values())[0]
    assert entry.total_token == 400


def test_resolve_usage_tokens_falls_back_total_for_reranker_model():
    metric = ModelUsageMetrics(model="bge-m3", total_token=77)
    model = type("StubModel", (), {"categories": ["reranker"]})()

    prompt_tokens, completion_tokens = _resolve_usage_tokens(metric, model)

    assert prompt_tokens == 77
    assert completion_tokens == 0


# ---------------------------------------------------------------------------
# _estimate_partial_usage — server-side backfill for incomplete reports
# ---------------------------------------------------------------------------


def test_estimate_partial_usage_skips_completed_reports():
    # ``completed=True`` means the canonical usage chunk arrived; no estimation
    # should overwrite the authoritative tokens.
    metric = ModelUsageMetrics(
        model="m",
        completed=True,
        request_content_bytes=4096,
        output_chunk_count=128,
    )
    _estimate_partial_usage(metric)
    assert metric.input_token == 0
    assert metric.output_token == 0
    assert metric.total_token == 0


def test_estimate_partial_usage_backfills_blank_tokens(monkeypatch):
    # Pin divisors so the assertion is independent of the production defaults.
    monkeypatch.setattr("gpustack.envs.USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN", 4)
    monkeypatch.setattr("gpustack.envs.USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK", 2)
    metric = ModelUsageMetrics(
        model="m",
        completed=False,
        request_content_bytes=400,
        output_chunk_count=10,
    )
    _estimate_partial_usage(metric)
    assert metric.input_token == 100  # 400 / 4
    assert metric.output_token == 20  # 10 * 2
    assert metric.total_token == 120


def test_estimate_partial_usage_preserves_existing_partial_values(monkeypatch):
    # Anthropic-style early ``input_token`` from message_start must survive
    # — only blank slots get filled.
    monkeypatch.setattr("gpustack.envs.USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN", 4)
    metric = ModelUsageMetrics(
        model="m",
        completed=False,
        input_token=999,
        request_content_bytes=400,
        output_chunk_count=0,
    )
    _estimate_partial_usage(metric)
    assert metric.input_token == 999  # not overwritten
    assert metric.output_token == 0  # no chunks → no estimate


def test_estimate_partial_usage_clamps_input_token_to_at_least_one(monkeypatch):
    # Tiny payload (< divisor) must still produce a non-zero input token —
    # otherwise the request count rises but token count stays at 0 forever
    # for short prompts on disconnect, masking the request from billing.
    monkeypatch.setattr("gpustack.envs.USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN", 100)
    metric = ModelUsageMetrics(
        model="m",
        completed=False,
        request_content_bytes=10,
    )
    _estimate_partial_usage(metric)
    assert metric.input_token == 1


# ---------------------------------------------------------------------------
# _trim_details_buffer_locked — bounds details buffer under flush failure
# ---------------------------------------------------------------------------


def test_accumulate_caps_details_buffer_and_drops_oldest(monkeypatch, caplog):
    # Cap to 3 so the test is fast; push 5 distinct metrics and assert the
    # oldest two are evicted FIFO with a WARNING log.
    monkeypatch.setattr("gpustack.envs.USAGE_DETAILS_BUFFER_MAX_SIZE", 3)

    metrics = [
        ModelUsageMetrics(model="m", model_id=1, user_id=i, input_token=i)
        for i in range(1, 6)
    ]
    with caplog.at_level("WARNING"):
        asyncio.run(accumulate_gateway_metrics(metrics))

    assert len(gateway_details_buffer) == 3
    # FIFO eviction: the surviving entries are the last three pushed.
    assert [e.user_id for e in gateway_details_buffer] == [3, 4, 5]
    assert any(
        "gateway_details_buffer exceeded cap" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Buffer key splits across midnight so cross-day streams roll up separately
# ---------------------------------------------------------------------------


def test_make_buffer_key_splits_across_midnight():
    # Two metrics identical except for the completion date — must hash to
    # distinct keys so the rollup attributes each request to the day it
    # ended in (proxy contract: completed_at anchors the billing period).
    nov_14_ms = 1700000000000  # 2023-11-14
    nov_15_ms = nov_14_ms + 24 * 3600 * 1000  # 2023-11-15

    m_day1 = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        completed_at=nov_14_ms,
    )
    m_day2 = ModelUsageMetrics(
        model="qwen3-0.6b",
        model_id=1,
        user_id=2,
        completed_at=nov_15_ms,
    )

    key1 = _make_buffer_key(m_day1)
    key2 = _make_buffer_key(m_day2)
    assert key1 != key2
    assert key1.endswith(".2023-11-14")
    assert key2.endswith(".2023-11-15")


def test_accumulate_splits_rollup_across_midnight():
    nov_14_ms = 1700000000000
    nov_15_ms = nov_14_ms + 24 * 3600 * 1000
    asyncio.run(
        accumulate_gateway_metrics(
            [
                ModelUsageMetrics(
                    model="qwen3-0.6b",
                    model_id=1,
                    user_id=2,
                    input_token=10,
                    completed_at=nov_14_ms,
                ),
                ModelUsageMetrics(
                    model="qwen3-0.6b",
                    model_id=1,
                    user_id=2,
                    input_token=20,
                    completed_at=nov_15_ms,
                ),
            ]
        )
    )
    assert len(gateway_metrics_buffer) == 2


@pytest.mark.asyncio
async def test_create_or_update_refreshes_route_name_on_rename(monkeypatch):
    """A mid-day route rename must converge the (route_id, date) cell to
    one row with the latest non-NULL name — not split into two rows or
    keep the morning's stale snapshot."""
    existing = ModelUsage(
        model_id=1,
        model_name="qwen3-0.6b",
        user_id=2,
        access_key="abc",
        model_route_id=21,
        model_route_name="old-name",
        date=date(2023, 11, 14),
        prompt_token_count=100,
        completion_token_count=50,
        prompt_cached_token_count=0,
        request_count=1,
    )

    async def fake_one_by_fields(session, fields):
        return existing

    monkeypatch.setattr(ModelUsage, "one_by_fields", fake_one_by_fields)

    incoming = ModelUsage(
        model_id=1,
        model_name="qwen3-0.6b",
        user_id=2,
        access_key="abc",
        model_route_id=21,
        model_route_name="new-name",
        date=date(2023, 11, 14),
        prompt_token_count=30,
        completion_token_count=20,
        prompt_cached_token_count=0,
        request_count=1,
    )

    save_mock = AsyncMock()
    monkeypatch.setattr(ModelUsage, "save", save_mock)

    await create_or_update_model_usage(MagicMock(), incoming, auto_commit=False)

    assert existing.model_route_name == "new-name"
    assert existing.prompt_token_count == 130
    assert existing.completion_token_count == 70
    assert existing.request_count == 2
    save_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_or_update_preserves_route_name_when_incoming_is_null(
    monkeypatch,
):
    """If the live route was deleted between dispatch and flush, the name
    lookup returns NULL — but the rollup row's existing snapshot must
    survive so that audit/breakdown can still label the bucket."""
    existing = ModelUsage(
        model_id=1,
        model_name="qwen3-0.6b",
        user_id=2,
        access_key="abc",
        model_route_id=21,
        model_route_name="old-name",
        date=date(2023, 11, 14),
        prompt_token_count=100,
        completion_token_count=50,
        prompt_cached_token_count=0,
        request_count=1,
    )

    async def fake_one_by_fields(session, fields):
        return existing

    monkeypatch.setattr(ModelUsage, "one_by_fields", fake_one_by_fields)

    incoming = ModelUsage(
        model_id=1,
        model_name="qwen3-0.6b",
        user_id=2,
        access_key="abc",
        model_route_id=21,
        model_route_name=None,
        date=date(2023, 11, 14),
        prompt_token_count=30,
        completion_token_count=20,
        prompt_cached_token_count=0,
        request_count=1,
    )

    save_mock = AsyncMock()
    monkeypatch.setattr(ModelUsage, "save", save_mock)

    await create_or_update_model_usage(MagicMock(), incoming, auto_commit=False)

    assert existing.model_route_name == "old-name"
    assert existing.prompt_token_count == 130
    save_mock.assert_awaited_once()
