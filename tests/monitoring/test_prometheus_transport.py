from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, MagicMock
from urllib.parse import urlencode

import pytest

from gpustack.monitoring import prometheus as prometheus_query


@pytest.mark.asyncio
@pytest.mark.parametrize('ranged', [False, True])
@pytest.mark.parametrize('long', [False, True])
async def test_prometheus_queries_use_form_post_for_large_exact_identity_populations(
    monkeypatch, ranged, long
):
    response = SimpleNamespace(
        status=200,
        text=AsyncMock(return_value='{"status":"success","data":{"result":[]}}'),
    )
    context = MagicMock()
    context.__aenter__.return_value = response
    client = SimpleNamespace(
        get=Mock(return_value=context), post=Mock(return_value=context)
    )
    monkeypatch.setattr(prometheus_query, 'api_base', lambda base: base + '/api/v1')
    query = ' or '.join(
        f'gpustack:worker_gpu_memory_used_bytes{{worker_id="{index}",worker_uuid="trusted-worker-{index}",owner_principal_id="17",gpu_uuid="GPU-trusted-{index}",gpu_chip_index="0"}}'
        for index in range(128 if long else 1)
    )
    if ranged:
        result = await prometheus_query.query_range(
            client, 'http://prom', query, 100, 200, 15
        )
    else:
        result = await prometheus_query.query_instant(client, 'http://prom', query, 200)
    assert result == []
    used, unused = (client.post, client.get) if long else (client.get, client.post)
    unused.assert_not_called()
    used.assert_called_once()
    args, kwargs = used.call_args
    assert args == ('http://prom/api/v1/' + ('query_range' if ranged else 'query'),)
    assert kwargs['timeout'].total == prometheus_query.QUERY_TIMEOUT_SECONDS
    params = kwargs['data' if long else 'params']
    assert params['query'] == query
    assert (
        len(urlencode(params).encode()) > prometheus_query.MAX_GET_QUERY_BYTES
    ) == long


@pytest.mark.asyncio
@pytest.mark.parametrize('long', [False, True])
async def test_get_and_post_keep_prometheus_error_semantics(monkeypatch, long):
    response = SimpleNamespace(status=503, text=AsyncMock(return_value='unavailable'))
    context = MagicMock()
    context.__aenter__.return_value = response
    client = SimpleNamespace(
        get=Mock(return_value=context), post=Mock(return_value=context)
    )
    monkeypatch.setattr(prometheus_query, 'api_base', lambda base: base + '/api/v1')
    with pytest.raises(ValueError, match='Prometheus returned 503'):
        await prometheus_query.query_instant(
            client, 'http://prom', 'x' * (5000 if long else 1), 200
        )
