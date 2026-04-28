import asyncio

import pytest

from gpustack.server.metrics_collector import (
    ModelUsageMetrics,
    _make_buffer_key,
    _resolve_usage_tokens,
    accumulate_gateway_metrics,
    gateway_metrics_buffer,
)


@pytest.fixture(autouse=True)
def clear_buffer():
    gateway_metrics_buffer.clear()
    yield
    gateway_metrics_buffer.clear()


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


def test_make_buffer_key():
    m = ModelUsageMetrics(model="qwen3-0.6b", model_id=1, user_id=2, access_key="abc")
    assert _make_buffer_key(m) == "1..qwen3-0.6b.2.abc"


def test_make_buffer_key_none_fields():
    m = ModelUsageMetrics(model="qwen3-0.6b")
    assert _make_buffer_key(m) == "..qwen3-0.6b.."


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
