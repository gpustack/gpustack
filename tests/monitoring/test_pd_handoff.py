"""Connector-specific KV handoff observations shared by deployment and history."""

import pytest

from tests.monitoring.test_models import (
    _collector,
    _entry,
    _instance,
    _model,
    _prometheus,
)


@pytest.mark.parametrize(
    'mode,backend,key,owner,status',
    [
        ('vllm-nixl', 'vllm', 'pd_kv_completion', 'decode', None),
        ('vllm-mooncake', 'vllm', 'pd_kv_completion', 'prefill', None),
        ('sglang-mooncake', 'sglang', 'pd_kv_completion', 'prefill', None),
        ('sglang-nixl', 'sglang', 'pd_kv_throughput', 'prefill', None),
        ('vllm-nixl', 'vllm', 'pd_kv_throughput', 'decode', None),
        ('vllm-mooncake', 'vllm', 'pd_kv_throughput', 'prefill', None),
        ('vllm-nixl', 'vllm', 'pd_kv_transfer_failures', 'decode', None),
        ('vllm-mooncake', 'vllm', 'pd_kv_transfer_failures', 'prefill', None),
        ('vllm-nixl', 'vllm', 'pd_kv_expired', 'prefill', None),
        ('sglang-nixl', 'sglang', 'pd_kv_expired', None, 'unsupported'),
        ('vllm-nixl', 'vllm', 'pd_kv_bootstrap_failures', None, 'unsupported'),
        ('vllm-ascend-mooncake', 'vllm', 'pd_kv_completion', None, 'unsupported'),
        ('vllm-ascend-mooncake', 'vllm', 'pd_kv_throughput', None, 'unsupported'),
    ],
)
def test_handoff_metrics_select_the_native_connector_owner(
    mode, backend, key, owner, status
):
    module = _collector()
    model = _model(backend, pd=True)
    model.disaggregation.mode.value = mode
    definition = next(
        d for d in module.model_metric_definitions(backend) if d.key == key
    )
    roles, actual_status, _ = module._roles(definition, model, backend, 'all')
    assert roles == (() if owner is None else (owner,))
    assert actual_status == status
    if owner:
        other = 'decode' if owner == 'prefill' else 'prefill'
        assert module._roles(definition, model, backend, other)[1:] == (
            'not_applicable',
            'role_not_applicable',
        )


def test_sglang_completion_is_not_a_bootstrap_handshake_or_speed_percentile():
    module = _collector()
    definitions = {d.key: d for d in module.model_metric_definitions('sglang')}
    assert (
        definitions['pd_kv_completion'].family
        == 'gpustack:pd_kv_transfer_latency_seconds'
    )
    assert definitions['pd_kv_completion'].measurement_scope == 'kv_transfer_duration'
    assert definitions['pd_kv_throughput'].family == 'gpustack:pd_kv_transfer_bytes_sum'
    assert definitions['pd_kv_throughput'].kind == 'counter'
    assert definitions['pd_kv_throughput'].unit == 'bytes_per_second'


@pytest.mark.asyncio
@pytest.mark.parametrize('view', ['aggregate', 'instances'])
async def test_handoff_byte_rate_and_event_counts_preserve_zero_and_native_scope(
    monkeypatch, view
):
    module = _collector()
    model = _model(pd=True)

    def samples(query):
        if 'gpustack:pd_kv_transfer_bytes_sum{' in query:
            return [_entry(1024, 13, 'decode')]
        if 'gpustack:pd_kv_transfer_failed_total{' in query:
            # The numeric query returns increase/window_seconds, including resets.
            return [_entry(2 / 3600, 13, 'decode'), _entry(0, 15, 'decode')]
        if 'gpustack:pd_kv_lease_expired_total{' in query:
            return [_entry(0, 12, 'prefill')]
        return []

    queries = _prometheus(monkeypatch, module, samples)
    response = await module.collect_model_metrics(
        model,
        [_instance(12, 'prefill'), _instance(13, 'decode'), _instance(15, 'decode')],
        window_seconds=3600,
        view=view,
        client=object(),
        section='pd',
    )
    throughput = response.metrics['pd_kv_throughput']
    assert throughput.status == 'available'
    assert throughput.summaries[0].value == 1024
    for _, query in queries:
        if 'gpustack:pd_kv_transfer_bytes_sum{' in query:
            assert 'role=~"decode"' in query
            assert 'rate(' in query or 'increase(' in query
            assert '_bucket' not in query and 'histogram_quantile' not in query
    failures = response.metrics['pd_kv_transfer_failures']
    assert sum(s.sample_count for s in failures.summaries) == pytest.approx(2)
    assert all(s.statistic == 'rate' for s in failures.summaries)
    if view == 'instances':
        assert [s.dimensions.model_instance_id for s in failures.summaries] == [13, 15]
        assert [s.sample_count for s in failures.summaries] == [2, 0]
    expired = response.metrics['pd_kv_expired']
    assert expired.status == 'available'
    assert expired.summaries[0].sample_count == 0
    assert expired.summaries[0].dimensions.role == 'prefill'


@pytest.mark.asyncio
async def test_missing_optional_sglang_transfer_samples_do_not_borrow_handshake(
    monkeypatch,
):
    module = _collector()
    model = _model('sglang', pd=True)
    model.disaggregation.mode.value = 'sglang-mooncake'
    queries = _prometheus(monkeypatch, module, lambda _: [])
    response = await module.collect_model_metrics(
        model,
        [_instance(12, 'prefill'), _instance(13, 'decode')],
        window_seconds=3600,
        client=object(),
        section='pd',
    )
    for key in ('pd_kv_completion', 'pd_kv_throughput'):
        assert response.metrics[key].status == 'no_data'
        assert all(s.value is None for s in response.metrics[key].summaries)
    completion = [q for _, q in queries if 'pd_kv_transfer_latency_seconds' in q]
    assert completion and all('role=~"prefill"' in q for q in completion)
    from gpustack.schemas.benchmark_monitoring import BenchmarkModelMetrics

    historical = BenchmarkModelMetrics.model_validate(response.model_dump())
    assert (
        historical.metrics['pd_kv_completion'].measurement_scope
        == 'kv_transfer_duration'
    )
