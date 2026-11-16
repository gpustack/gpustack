"""Censored quantiles retain evidence without inventing an overflow value."""

import math

import pytest

from gpustack.monitoring import queries
from gpustack.schemas.model_metrics import MetricDimensions, MetricSeries, MetricSummary


def _bounds():
    from gpustack.monitoring import histogram_bounds

    return histogram_bounds


@pytest.mark.parametrize(
    'finite_count,expected',
    [(2, {'p50': 5, 'p95': 5, 'p99': 5}), (50, {'p95': 5, 'p99': 5}), (100, {})],
)
def test_only_quantiles_strictly_above_the_last_finite_bucket_get_lower_bounds(
    finite_count, expected
):
    assert (
        _bounds().histogram_lower_bounds(
            {'decode': {1.0: 0, 5.0: finite_count, math.inf: 100}}, count=100
        )
        == expected
    )


@pytest.mark.parametrize(
    'profiles,count',
    [
        ({}, 100),
        ({'a': {5.0: 1, math.inf: 1}}, 0),
        ({'a': {5.0: 1, math.inf: 2}, 'b': {10.0: 1, math.inf: 2}}, 4),
        ({'a': {5.0: 1, math.inf: 2}}, 3),
        ({'a': {-math.inf: 0, math.inf: 100}}, 100),
    ],
)
def test_missing_zero_or_incompatible_histograms_cannot_prove_a_lower_bound(
    profiles, count
):
    assert _bounds().histogram_lower_bounds(profiles, count=count) == {}


def test_cdf_query_keeps_cohort_pairing_population_guards_and_bucket_labels():
    scope = queries.MetricQueryScope(1, 9, (12,), ('decode',), 'instances')
    query = queries.build_histogram_cdf_query(
        'gpustack:pd_kv_transfer_seconds', [(scope, 5.0, ('5.0',))], 75
    )
    assert 'le=~"5\\\\.0"' in query
    assert 'sum by (role, model_instance_id)' in query
    assert 'max without (le)' in query
    assert 'le="+Inf"' in query
    assert 'histogram_upper_bound' in query
    assert 'model_instance_id=~"12"' in query
    assert 'gpustack:model_instance_source_info' in query
    assert 'status="success"' in query
    assert '[75s]' in query
    assert ' == ' in query


def test_cdf_normalizes_bucket_aliases_and_checks_all_three_populations():
    scope = queries.MetricQueryScope(1, 9, (12,), ('decode',), 'instances')
    query = queries.build_histogram_cdf_query(
        'gpustack:pd_kv_transfer_seconds', [(scope, 5.0, ('5', '5.0'))], 75
    )
    assert 'max without (le)' in query
    assert 'min without (le)' in query
    assert query.count('count by (role, model_instance_id)') >= 6
    assert 'without (le, engine)' not in query


def test_censoring_uses_matching_timestamp_population_and_cdf_not_value_equality():
    bounds = _bounds()
    line = MetricSeries(
        name='Decode',
        statistic='p50',
        dimensions=MetricDimensions(role='decode', model_instance_id=12),
        points=[(100, 0), (130, 5), (160, 5), (190, 5)],
    )
    avg = MetricSeries(name='Decode', statistic='avg', points=[(130, 14)])
    rows = [
        {
            'metric': {
                'role': 'decode',
                'model_instance_id': '12',
                'histogram_upper_bound': '5.0',
            },
            'values': [[130, '0.02'], [160, '0.5']],
        }
    ]
    censored, missing = bounds.apply_lower_bounds(
        [line, avg], {('decode', 12): 5.0}, rows, view='instances'
    )
    assert censored and missing
    assert line.points == [(100, 0), (130, None), (160, 5), (190, None)]
    assert line.lower_bounds == [(130, 5)]
    assert avg.points == [(130, 14)] and avg.lower_bounds is None


def test_foreign_role_or_instance_cdf_cannot_classify_an_observation():
    line = MetricSeries(
        name='D',
        statistic='p99',
        dimensions=MetricDimensions(role='decode', model_instance_id=12),
        points=[(100, 5)],
    )
    rows = [
        {
            'metric': {
                'role': 'prefill',
                'model_instance_id': '13',
                'histogram_upper_bound': '5',
            },
            'values': [[100, '0']],
        }
    ]
    censored, missing = _bounds().apply_lower_bounds(
        [line], {('decode', 12): 5}, rows, view='instances'
    )
    assert not censored and missing
    assert line.points == [(100, None)] and line.lower_bounds is None


def test_optional_metadata_preserves_old_payloads_and_finite_censored_observations():
    old = MetricSeries(name='P', statistic='p99', points=[(100, 1)])
    assert 'lower_bounds' not in old.model_dump(mode='json')
    summary = MetricSummary(statistic='p99', value=None, lower_bound=5)
    assert summary.model_dump(mode='json')['lower_bound'] == 5
    series = MetricSeries(
        name='P', statistic='p99', points=[(100, None)], lower_bounds=[(100, 5)]
    )
    assert series.model_dump(mode='json')['lower_bounds'] == [[100, 5]]
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MetricSummary(statistic='p99', value=5, lower_bound=5)
    with pytest.raises(ValidationError):
        MetricSeries(
            name='P',
            statistic='p99',
            points=[(100, None)],
            lower_bounds=[(100, float('inf'))],
        )
