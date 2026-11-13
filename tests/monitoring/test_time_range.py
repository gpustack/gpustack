from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gpustack.monitoring import time_range as monitoring


@pytest.mark.parametrize('seconds', [300, 3600, 604800])
def test_relative_intervals_keep_the_service_span_limit(seconds):
    value = monitoring.monitoring_interval(seconds, now=700000)
    assert (value.start, value.end, value.seconds) == (
        700000 - seconds,
        700000,
        seconds,
    )


def test_absolute_interval_is_frozen_in_the_past_instead_of_ending_now():
    value = monitoring.monitoring_interval(3600, 1000.0, 2000.0, now=900000)
    assert (value.start, value.end, value.seconds) == (1000, 2000, 1000)


@pytest.mark.parametrize(
    'start,end',
    [
        (None, 2000),
        (1000, None),
        (1000, 1000),
        (2000, 1000),
        (1000, 1299),
        (0, 604801),
        (-1, 1000),
        (float('nan'), 2000),
        (1000, float('inf')),
        (1000.5, 2000),
        (True, 1000),
        (1000, 900001),
    ],
)
def test_invalid_absolute_intervals_are_rejected_before_queries(start, end):
    with pytest.raises(ValueError):
        monitoring.monitoring_interval(3600, start, end, now=900000)


@pytest.fixture
def status_environment(monkeypatch):
    monitoring._POLICY_CACHE.clear()
    monkeypatch.setattr(
        monitoring.prometheus_query, 'prometheus_url', lambda: 'http://prom'
    )
    monkeypatch.setattr(
        monitoring.prometheus_query, 'api_base', lambda _: 'http://prom/api/v1'
    )

    def make(responses):
        def request(url, **kwargs):
            endpoint = url.rsplit('/', 1)[-1]
            value = responses[endpoint]
            if isinstance(value, Exception):
                raise value
            context = MagicMock()
            context.__aenter__.return_value = SimpleNamespace(
                status=value[0], json=AsyncMock(return_value=value[1])
            )
            return context

        return SimpleNamespace(get=Mock(side_effect=request))

    yield make
    monitoring._POLICY_CACHE.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'value,seconds,size,known',
    [
        ('15d', 1296000, False, True),
        ('1w2d3h4m5s', 788645, False, True),
        ('15d or 10GiB', 1296000, True, True),
        ('15d or 10GiB or 80%', 1296000, True, True),
        ('80%', None, True, True),
        ('10GB', None, True, True),
        ('0s', None, False, False),
        ('', None, False, False),
        (None, None, False, False),
        ('future format', None, False, False),
    ],
)
async def test_effective_runtime_policy_is_sanitized_and_not_overridden_by_flags(
    status_environment, value, seconds, size, known
):
    client = status_environment(
        {
            'runtimeinfo': (
                200,
                {
                    'status': 'success',
                    'data': {
                        'storageRetention': value,
                        'hostname': 'private-host',
                        'timeSeriesCount': 99999,
                    },
                },
            ),
        }
    )
    result = await monitoring.collect_monitoring_capabilities(7, client)
    assert result.retention_seconds == seconds
    assert result.retention_size_based is size
    assert result.retention_known is known
    assert result.retention_source == 'prometheus_runtimeinfo'
    assert result.max_query_seconds == 604800
    assert result.min_query_seconds == 300
    assert result.supports_custom_range
    assert 'private-host' not in result.model_dump_json()
    assert '99999' not in result.model_dump_json()
    assert client.get.call_count == 1
    assert client.get.call_args.kwargs['timeout'].total == 2


@pytest.mark.asyncio
async def test_missing_runtime_property_falls_back_to_explicit_flags(
    status_environment,
):
    client = status_environment(
        {
            'runtimeinfo': (
                200,
                {'status': 'success', 'data': {'hostname': 'private'}},
            ),
            'flags': (
                200,
                {
                    'status': 'success',
                    'data': {
                        'storage.tsdb.retention.time': '3d',
                        'storage.tsdb.retention.size': '8GB',
                        'web.config.file': '/private/secrets.yml',
                    },
                },
            ),
        }
    )
    result = await monitoring.collect_monitoring_capabilities(9, client)
    assert result.retention_source == 'prometheus_flags'
    assert result.retention_seconds == 259200
    assert result.retention_size_based
    assert '/private/' not in result.model_dump_json()
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_unreadable_policy_does_not_guess_default_retention(status_environment):
    client = status_environment(
        {
            'runtimeinfo': (403, {'status': 'error', 'data': {}}),
            'flags': (404, {'status': 'error', 'data': {}}),
        }
    )
    result = await monitoring.collect_monitoring_capabilities(7, client)
    assert result.retention_seconds is None and not result.retention_known
    assert result.retention_source == 'unknown'
    assert result.max_query_seconds == 604800 and result.supports_custom_range


@pytest.mark.asyncio
@pytest.mark.parametrize('size,known', [('0', False), ('1073741824', True)])
async def test_flags_can_report_byte_size_without_claiming_a_time_bound(
    status_environment, size, known
):
    client = status_environment(
        {
            'runtimeinfo': (404, {'status': 'error', 'data': {}}),
            'flags': (
                200,
                {
                    'status': 'success',
                    'data': {
                        'storage.tsdb.retention.time': '0s',
                        'storage.tsdb.retention.size': size,
                    },
                },
            ),
        }
    )
    result = await monitoring.collect_monitoring_capabilities(7, client)
    assert result.retention_seconds is None
    assert result.retention_size_based is known
    assert result.retention_known is known


@pytest.mark.asyncio
async def test_only_sanitized_policy_is_cached_and_model_identity_does_not_cross_requests(
    status_environment,
):
    client = status_environment(
        {
            'runtimeinfo': (
                200,
                {'status': 'success', 'data': {'storageRetention': '15d'}},
            ),
        }
    )
    first = await monitoring.collect_monitoring_capabilities(7, client)
    second = await monitoring.collect_monitoring_capabilities(8, client)
    assert first.model_id == 7 and second.model_id == 8
    assert first.retention_seconds == second.retention_seconds
    assert client.get.call_count == 1


@pytest.mark.asyncio
async def test_unconfigured_prometheus_keeps_service_limits_without_network(
    monkeypatch,
):
    monkeypatch.setattr(monitoring.prometheus_query, 'prometheus_url', lambda: None)
    client = SimpleNamespace(get=Mock())
    result = await monitoring.collect_monitoring_capabilities(7, client)
    assert not result.retention_known and result.retention_seconds is None
    client.get.assert_not_called()
