"""Server-selected PromQL populations and compatible histogram statistics."""

import importlib
import importlib.util
import math

import pytest


def _queries():
    name = 'gpustack.monitoring.queries'
    assert importlib.util.find_spec(name) is not None, 'Semantic query builders missing'
    return importlib.import_module(name)


def _scope(view='aggregate'):
    return _queries().MetricQueryScope(1, 9, (12, 13), ('prefill', 'decode'), view)


def test_range_histogram_keeps_pd_roles_and_current_source_guards():
    module = _queries()
    query = module.build_histogram_query(
        'gpustack:time_to_first_token_seconds', _scope(), 'p95', 90
    )
    assert 'histogram_quantile(0.95,' in query
    assert 'sum by (role, le)' in query
    assert 'rate(gpustack:time_to_first_token_seconds_bucket' in query
    assert '[90s]' in query
    assert 'cluster_id="1"' in query and 'model_id="9"' in query
    assert 'model_instance_id=~"12|13"' in query
    assert 'restart_count!=""' in query and 'source_kind!=""' in query
    assert 'gpustack:runtime_metrics_scrape_status' in query
    assert 'status="success"' in query
    assert 'gpustack:model_instance_source_info' in query
    for label in module.SOURCE_LABELS:
        assert label in query
    assert 'max_over_time' not in query


def test_whole_window_summary_preserves_old_epoch_presence():
    module = _queries()
    queries = module.build_histogram_components(
        'gpustack:time_to_first_token_seconds', _scope(), 3600
    )
    assert set(queries) == {'buckets', 'sum', 'count'}
    query = queries['buckets']
    assert 'increase(gpustack:time_to_first_token_seconds_bucket' in query
    assert '[3600s]' in query
    assert 'max_over_time(gpustack:model_instance_source_info' in query
    assert 'max_over_time(gpustack:runtime_metrics_scrape_status' in query
    assert 'restart_count="' not in query
    assert 'source_name="' not in query
    # Native dimensions survive the raw probe, rather than a fixed group-by
    # list erasing streaming/finish/engine cohorts before compatibility checks.
    assert 'sum by' not in query
    assert 'engine' not in query.split('max without (', 1)[1].split(')', 1)[0]


def test_counter_queries_use_reset_aware_rates_and_full_window_summary():
    module = _queries()
    scope = _scope('instances')
    trend = module.build_counter_query('gpustack:generation_tokens_total', scope, 90)
    summary = module.build_counter_query(
        'gpustack:generation_tokens_total', scope, 3600, summary=True
    )
    assert 'sum by (role, model_instance_id)' in trend
    assert 'rate(gpustack:generation_tokens_total' in trend
    assert 'increase(gpustack:generation_tokens_total' in summary
    assert '/ 3600' in summary
    assert 'max_over_time' in summary


def test_average_divides_aggregated_sums_and_counts_not_source_averages():
    query = _queries().build_histogram_query(
        'gpustack:time_per_output_token_seconds', _scope(), 'avg', 90
    )
    assert 'avg(' not in query
    assert query.count('sum by (role)') == 2
    assert '_sum{' in query and '_count{' in query


def test_source_quantiles_are_merged_from_compatible_buckets():
    stats, compatible = _queries().histogram_statistics(
        {'a': {1.0: 1, 2.0: 3, math.inf: 4}, 'b': {1.0: 1, 2.0: 1, math.inf: 1}},
        count=5,
        total=6,
    )
    assert compatible
    assert stats == {'avg': 1.2, 'p50': 1.25, 'p95': 2.0, 'p99': 2.0}


@pytest.mark.parametrize(
    'other',
    [
        {1.0: 1, 3.0: 1, math.inf: 1},
        {1.0: 1, 2.0: 1},
    ],
)
def test_incompatible_or_missing_buckets_keep_average_without_global_p95(other):
    stats, compatible = _queries().histogram_statistics(
        {'a': {1.0: 1, 2.0: 3, math.inf: 4}, 'b': other}, count=5, total=6
    )
    assert not compatible
    assert stats == {'avg': 1.2, 'p50': None, 'p95': None, 'p99': None}


def test_zero_histogram_observations_do_not_become_zero_latency():
    stats, _ = _queries().histogram_statistics(
        {'a': {1.0: 0, math.inf: 0}}, count=0, total=0
    )
    assert all(value is None for value in stats.values())


def test_query_scope_rejects_foreign_selector_syntax_and_empty_population():
    module = _queries()
    with pytest.raises(ValueError):
        module.MetricQueryScope(1, 9, (12,), ('decode"}',), 'aggregate')
    with pytest.raises(ValueError):
        module.MetricQueryScope(1, 9, (), ('decode',), 'aggregate')
    with pytest.raises(ValueError):
        module.build_counter_query('arbitrary{leak="yes"}', _scope(), 90)


def test_lhs_scrape_plumbing_is_deduplicated_before_business_sum():
    query = _queries().build_counter_query(
        'gpustack:generation_tokens_total', _scope(), 90
    )
    assert 'max without (job, instance,' in query
    assert 'min without (job, instance,' in query
    assert ' == min without (' in query
    dedup_labels = query.split('max without (', 1)[1].split(')', 1)[0]
    assert 'is_streaming' not in dedup_labels
    assert 'finished_reason' not in dedup_labels
    assert 'engine' not in dedup_labels and 'dp_rank' not in dedup_labels
    assert 'exported_instance' not in dedup_labels


def test_duplicate_jobs_do_not_double_histogram_tokens_or_erase_native_cohorts():
    module = _queries()
    rows = []
    for job in ('primary', 'additional-scrape'):
        for streaming, value in [('true', 3), ('false', 2)]:
            rows.append(
                {
                    'metric': {
                        'model_instance_id': '12',
                        'source_name': 'head',
                        'engine': '0',
                        'is_streaming': streaming,
                        'job': job,
                    },
                    'value': [100, str(value)],
                }
            )
    assert hasattr(
        module, 'deduplicate_instant'
    ), 'Instant cohorts are not deduplicated'
    distinct = module.deduplicate_instant(rows)
    assert len(distinct) == 2
    assert sum(float(row['value'][1]) for row in distinct) == 5


@pytest.mark.parametrize('values', [(2, 3), (3, 2), (2, 3, 2)])
def test_conflicting_scrape_copies_are_not_resolved_by_the_largest_value(values):
    rows = [
        {
            'metric': {
                'model_instance_id': '12',
                'source_name': 'head',
                'job': str(index),
            },
            'value': [100, str(value)],
        }
        for index, value in enumerate(values)
    ]
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        _queries().deduplicate_instant(rows)


def test_incomplete_bucket_population_cannot_supply_percentiles_for_full_count():
    stats, compatible = _queries().histogram_statistics(
        {'source': {0.1: 2, 0.2: 5, math.inf: 5}}, count=10, total=2
    )
    assert not compatible
    assert stats == {'avg': 0.2, 'p50': None, 'p95': None, 'p99': None}


def test_exported_native_instance_is_a_real_cohort_not_target_plumbing():
    rows = [
        {
            'metric': {
                'job': job,
                'instance': 'scrape:1',
                'exported_instance': native,
                'model_instance_id': '12',
            },
            'value': [100, str(value)],
        }
        for job in ('one', 'two')
        for native, value in [('a', 3), ('b', 4)]
    ]
    distinct = _queries().deduplicate_instant(rows)
    assert len(distinct) == 2
    assert sum(float(row['value'][1]) for row in distinct) == 7


def test_ratio_trend_pairs_native_cohorts_before_role_aggregation():
    module = _queries()
    assert hasattr(module, 'build_ratio_query'), 'Ratio population matching missing'
    query = module.build_ratio_query(
        'gpustack:prefix_cache_hits_total',
        'gpustack:prefix_cache_queries_total',
        _scope(),
        90,
    )
    assert query.count('count by (role)') == 4
    assert query.count('and on (role)') == 2
    assert 'gpustack:prefix_cache_hits_total' in query
    assert 'gpustack:prefix_cache_queries_total' in query
