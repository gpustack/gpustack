from types import SimpleNamespace

import pytest

from gpustack.schemas.benchmark import BenchmarkMetrics
from gpustack.worker.benchmark import analysis
from gpustack.worker.benchmark_manager import BenchmarkManager


def point(rate, output, total=None, ttft=100.0, sequence=0, **extra):
    return {
        "rate": rate,
        "input_tokens": 128,
        "sequence": sequence,
        "output_tokens_per_second_mean": output,
        "tokens_per_second_mean": output * 2 if total is None else total,
        "time_to_first_token_mean": ttft,
        "inter_token_latency_mean": 5.0,
        "request_total": 100,
        "request_successful": 100,
        "request_errored": 0,
        "request_incomplete": 0,
        **extra,
    }


def coordinates(row):
    return {key: row[key] for key in ("input_tokens", "sequence", "rate")}


def slo_benchmark(**extra):
    return SimpleNamespace(
        slo_avg_ttft_ms=500.0,
        auto_tune=True,
        max_points=20,
        upper_bound=256,
        **extra,
    )


def test_highest_passing_load_is_not_capped_by_total_peak():
    rows = [point(128, 15000), point(256, 14000, ttft=450)]

    result = analysis.compute_best_points(slo_benchmark(), rows)

    assert result == {
        "peak_rate": 128.0,
        "slo_met_rate": 256.0,
        "recommended_rate": 256.0,
    }


def test_non_monotonic_throughput_does_not_change_slo_capacity():
    rows = [
        point(128, 14004.82, ttft=415.75),
        point(142, 14060.40, ttft=449.03, sequence=1),
        point(143, 13413.13, ttft=499.09, sequence=2),
        point(144, 12498.39, ttft=608.79, sequence=3),
        point(256, 14724.31, ttft=1351.30, sequence=4),
    ]
    benchmark = slo_benchmark()
    result = analysis.compute_best_points(benchmark, rows)

    assert result["recommended_rate"] == result["slo_met_rate"] == 143.0
    assert result["peak_rate"] == 256.0
    validity = analysis.compute_validity(benchmark, rows, result)
    assert validity["primary_point"] == coordinates(rows[2])
    assert validity["peak_point"] == coordinates(rows[4])


@pytest.mark.parametrize("profile", ["Max Throughput", "Custom"])
def test_total_throughput_selects_peak_when_output_has_a_different_maximum(profile):
    rows = [point(16, 3000, total=20000), point(32, 4000, total=15000)]

    result = analysis.compute_best_points(SimpleNamespace(profile=profile), rows)

    assert result == {"peak_rate": 16.0, "recommended_rate": 16.0}


def test_a_measured_zero_total_does_not_fall_back_to_output_throughput():
    rows = [point(16, 100000, total=0), point(32, 10, total=1)]

    assert analysis.compute_best_points(SimpleNamespace(), rows)["peak_rate"] == 32


def test_missing_total_does_not_fall_back_to_output_throughput():
    rows = [point(16, 100000), point(32, 10, total=1)]
    rows[0]["tokens_per_second_mean"] = None

    assert analysis.compute_best_points(SimpleNamespace(), rows)["peak_rate"] == 32

    rows[1]["tokens_per_second_mean"] = None
    assert analysis.compute_best_points(SimpleNamespace(), rows) == {}


def test_total_throughput_does_not_require_output_metrics():
    rows = [point(16, 3000), point(32, 4000)]
    for row in rows:
        row.pop("output_tokens_per_second_mean")

    assert analysis.compute_best_points(SimpleNamespace(), rows)["peak_rate"] == 32


def test_peak_ties_select_the_lower_load_and_then_the_earlier_stage():
    rows = [
        point(32, 4000, sequence=2),
        point(16, 4000, sequence=1),
        point(16, 4000, sequence=0),
    ]
    benchmark = SimpleNamespace(auto_tune=False)
    result = analysis.compute_best_points(benchmark, rows)
    validity = analysis.compute_validity(benchmark, rows, result)

    assert result["peak_rate"] == 16
    assert validity["primary_point"] == coordinates(rows[2])
    assert validity["peak_point"] == coordinates(rows[2])
    assert validity["coverage_applicable"] is False


def test_duplicate_slo_load_selects_the_qualifying_total_peak_stage():
    rows = [
        point(32, 5000, ttft=600, sequence=0),
        point(32, 4000, ttft=400, sequence=1),
        point(32, 3500, ttft=350, sequence=2),
    ]
    benchmark = slo_benchmark()
    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows)
    )

    assert validity["primary_point"] == coordinates(rows[1])
    assert validity["peak_point"] == coordinates(rows[0])


@pytest.mark.parametrize("with_facts", [False, True])
def test_slo_range_end_is_not_a_boundary_even_after_throughput_declines(with_facts):
    rows = [point(128, 15000), point(256, 14000, ttft=450, sequence=1)]
    benchmark = slo_benchmark()
    facts = (
        {
            "stop_reason": "upper_bound",
            "bracket_reason": "upper_bound",
            "slo_bracket": [256.0, None],
        }
        if with_facts
        else None
    )

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["slo_boundary_located"] is False
    assert validity["sufficient"] is False
    assert [item["code"] for item in validity["warnings"]] == ["not_saturated"]


def test_old_plateau_stop_reports_an_unlocated_slo_boundary():
    rows = [point(128, 15000), point(256, 14000, ttft=450, sequence=1)]
    benchmark = slo_benchmark()
    validity = analysis.compute_validity(
        benchmark,
        rows,
        analysis.compute_best_points(benchmark, rows),
        {"stop_reason": "capacity_plateau", "bracket_reason": "capacity_plateau"},
    )

    assert validity["slo_boundary_located"] is False
    assert [item["code"] for item in validity["warnings"]] == ["slo_not_binding"]
    assert validity["primary_point"] == coordinates(rows[1])


@pytest.mark.parametrize("with_bracket", [False, True])
def test_plateau_stop_does_not_override_a_measured_slo_boundary(with_bracket):
    rows = [
        point(64, 5000, ttft=200),
        point(128, 4900, ttft=450, sequence=1),
        point(256, 4800, ttft=600, sequence=2),
    ]
    benchmark = slo_benchmark()
    facts = {"stop_reason": "converged", "bracket_reason": "capacity_plateau"}
    if with_bracket:
        facts["slo_bracket"] = [128, 256]

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["slo_boundary_located"] is True
    assert validity["warnings"] == []
    assert validity["sufficient"] is True


def test_fixed_rate_shortfall_does_not_invalidate_a_located_slo_boundary():
    rows = [
        point(64, 5000, ttft=200, requests_per_second_mean=23.6),
        point(128, 4900, ttft=450, sequence=1, requests_per_second_mean=23.6),
        point(256, 4800, ttft=600, sequence=2, requests_per_second_mean=23.6),
    ]
    benchmark = slo_benchmark(load_type="fixed_rate")
    best = analysis.compute_best_points(benchmark, rows)
    validity = analysis.compute_validity(
        benchmark,
        rows,
        best,
        {"stop_reason": "converged", "bracket_reason": "slo_failed"},
    )

    assert best["recommended_rate"] == 128
    assert validity["primary_point"] == coordinates(rows[1])
    assert validity["slo_boundary_located"] is True
    assert validity["warnings"] == []


def test_no_passing_point_clears_primary_selection_but_preserves_peak():
    rows = [point(16, 5000, ttft=600)]
    benchmark = slo_benchmark()
    best = analysis.compute_best_points(benchmark, rows)
    validity = analysis.compute_validity(benchmark, rows, best)

    assert best == {"peak_rate": 16.0}
    assert validity["primary_point"] is None
    assert validity["peak_point"] == coordinates(rows[0])


class PointReport:
    def __init__(self, row):
        self.row = row

    def to_results(self, input_tokens, sequence_start):
        return [{**self.row, "input_tokens": input_tokens, "sequence": sequence_start}]

    def to_metrics(self):
        return BenchmarkMetrics.model_validate(self.row)


@pytest.mark.parametrize("auto_tune", [False, True])
def test_parent_metrics_belong_to_highest_passing_load(monkeypatch, auto_tune):
    rows = [point(128, 15000, ttft=300), point(256, 14000, ttft=450)]
    reports = {str(i): PointReport(row) for i, row in enumerate(rows)}
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        reports.__getitem__,
    )
    benchmark = slo_benchmark()
    benchmark.dataset_input_tokens = 128
    benchmark.auto_tune = auto_tune

    collected = object.__new__(BenchmarkManager)._aggregate_points(
        benchmark, list(reports)
    )

    assert collected.metrics.output_tokens_per_second_mean == 14000
    assert collected.metrics.time_to_first_token_mean == 450
    assert len(collected.results) == 2


def test_unmet_slo_keeps_results_and_empty_parent_metrics(monkeypatch):
    row = point(16, 5000, ttft=600)
    report = PointReport(row)
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        lambda _path: report,
    )
    benchmark = slo_benchmark()
    benchmark.dataset_input_tokens = 128

    collected = object.__new__(BenchmarkManager)._aggregate_points(benchmark, ["x"])

    assert collected.metrics is not None
    assert collected.metrics.output_tokens_per_second_mean is None
    assert collected.metrics.time_to_first_token_mean is None
    assert collected.metrics.request_total is None
    assert collected.loaded == 1
    assert len(collected.results) == 1
    assert collected.report is report


def test_single_run_with_unmet_slo_keeps_empty_parent_metrics(monkeypatch, tmp_path):
    report = PointReport(point(16, 5000, ttft=600))
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        lambda _path: report,
    )
    benchmark = slo_benchmark()
    benchmark.id = 7
    benchmark.name = "single"
    benchmark.dataset_input_tokens = 128
    benchmark.auto_tune = False
    benchmark.stages = None
    manager = object.__new__(BenchmarkManager)
    manager._benchmark_dir = str(tmp_path)

    collected = manager._collect_results(benchmark)

    assert collected.metrics is not None
    assert all(value is None for value in collected.metrics.model_dump().values())
    assert len(collected.results) == collected.loaded == 1


def test_slo_rate_answer_is_not_cleared_by_an_offered_rate_shortfall():
    row = point(64, 5000, requests_per_second_mean=23.6)
    benchmark = slo_benchmark(load_type="fixed_rate")
    best = analysis.compute_best_points(benchmark, [row])
    validity = analysis.compute_validity(benchmark, [row], best)

    assert best["recommended_rate"] == 64
    assert validity["primary_point"] == coordinates(row)
    assert "saturated_at_lower_bound" not in [w["code"] for w in validity["warnings"]]


def test_unmet_slo_result_sync_preserves_grid_without_failing(monkeypatch):
    manager = object.__new__(BenchmarkManager)
    benchmark = SimpleNamespace(id=7, name="unmet")
    row = point(16, 5000, ttft=600)
    from gpustack.worker.benchmark_manager import CollectedResults

    collected = CollectedResults([row], BenchmarkMetrics(), None, 1, 0)
    monkeypatch.setattr(manager, "_collect_results", lambda _benchmark: collected)
    monkeypatch.setattr(manager, "_load_request_samples", lambda *a, **k: ([], []))
    monkeypatch.setattr(manager, "_log_request_failures_if_any", lambda **k: None)
    monkeypatch.setattr(manager, "_write_best_points_and_validity", lambda *a: None)
    posted = []
    monkeypatch.setattr(
        manager, "_post_metrics_and_results", lambda *a: posted.append(a)
    )

    manager._sync_benchmark_metrics(benchmark)

    assert posted[0][2] == [row]
    assert all(value is None for value in posted[0][1].model_dump().values())


def test_parent_metrics_follow_total_peak_even_when_output_is_lower(monkeypatch):
    rows = [point(16, 3000, total=20000), point(32, 4000, total=15000)]
    reports = {str(i): PointReport(row) for i, row in enumerate(rows)}
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        reports.__getitem__,
    )
    benchmark = SimpleNamespace(dataset_input_tokens=128)

    collected = object.__new__(BenchmarkManager)._aggregate_points(
        benchmark, list(reports)
    )

    assert collected.metrics.output_tokens_per_second_mean == 3000
    assert collected.metrics.tokens_per_second_mean == 20000


def test_duplicate_load_metrics_and_descriptor_select_the_same_stage(monkeypatch):
    rows = [
        point(32, 3000, total=20000, ttft=100),
        point(32, 4000, total=15000, ttft=200),
    ]
    reports = {str(i): PointReport(row) for i, row in enumerate(rows)}
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        reports.__getitem__,
    )
    benchmark = slo_benchmark()
    benchmark.dataset_input_tokens = 128
    collected = object.__new__(BenchmarkManager)._aggregate_points(
        benchmark, list(reports)
    )

    validity = analysis.compute_validity(
        benchmark,
        collected.results,
        analysis.compute_best_points(benchmark, collected.results),
    )

    assert collected.metrics.output_tokens_per_second_mean == 3000
    assert collected.metrics.tokens_per_second_mean == 20000
    assert collected.metrics.time_to_first_token_mean == 100
    assert validity["primary_point"] == {"input_tokens": 128, "sequence": 0, "rate": 32}


def test_invalid_throughput_search_floor_has_no_parent_result_metrics(monkeypatch):
    report = PointReport(point(64, 5000, requests_per_second_mean=23.6))
    monkeypatch.setattr(
        "gpustack.worker.benchmark_manager.GenerativeBenchmarksReport.load_file",
        lambda _path: report,
    )
    benchmark = SimpleNamespace(
        dataset_input_tokens=128,
        auto_tune=True,
        load_type="fixed_rate",
        upper_bound=256,
    )

    collected = object.__new__(BenchmarkManager)._aggregate_points(benchmark, ["x"])

    assert collected.metrics is not None
    assert all(value is None for value in collected.metrics.model_dump().values())
    assert len(collected.results) == 1


def declining_slo_points():
    return [
        point(16, 10000, ttft=100),
        point(32, 8400, ttft=200, sequence=1),
        point(64, 8000, ttft=400, sequence=2),
    ]


@pytest.mark.parametrize("with_final_reason", [False, True])
def test_explicit_throughput_decline_stop_leaves_slo_boundary_unlocated(
    with_final_reason,
):
    rows = declining_slo_points()
    benchmark = slo_benchmark()
    best = analysis.compute_best_points(benchmark, rows)
    facts = {
        "bracket_reason": "throughput_decline",
        "slo_bracket": [64.0, None],
        "stopped_at": 64.0,
    }
    if with_final_reason:
        facts["stop_reason"] = "throughput_decline"

    validity = analysis.compute_validity(benchmark, rows, best, facts)

    assert validity["warnings"] == [{"code": "throughput_decline", "params": {}}]
    assert validity["sufficient"] is False
    assert validity["slo_boundary_located"] is False
    assert validity["stop_reason"] == "throughput_decline"
    assert validity["slo_bracket"] == [64.0, None]
    assert best["recommended_rate"] == best["slo_met_rate"] == 64.0
    assert best["peak_rate"] == 16.0
    assert validity["primary_point"] == coordinates(rows[2])
    assert validity["peak_point"] == coordinates(rows[0])


def test_a_declining_grid_does_not_infer_a_protective_stop():
    rows = declining_slo_points()
    benchmark = slo_benchmark()

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows)
    )

    assert [warning["code"] for warning in validity["warnings"]] == ["not_saturated"]
    assert validity["slo_boundary_located"] is False


@pytest.mark.parametrize(
    "stop_reason", ["budget_points", "budget_seconds", "point_failed"]
)
def test_final_budget_or_error_reason_overrides_an_earlier_decline_bracket(stop_reason):
    rows = declining_slo_points()
    benchmark = slo_benchmark()
    facts = {
        "stop_reason": stop_reason,
        "bracket_reason": "throughput_decline",
        "slo_bracket": [64.0, None],
    }

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["stop_reason"] == stop_reason
    assert "throughput_decline" not in [
        warning["code"] for warning in validity["warnings"]
    ]
    if stop_reason.startswith("budget_"):
        assert validity["warnings"][0]["code"] == "budget_exhausted"


@pytest.mark.parametrize("stop_reason", ["converged", "slo_failed"])
def test_a_measured_slo_boundary_is_not_overwritten_by_a_decline_bracket(stop_reason):
    rows = declining_slo_points() + [point(128, 7500, ttft=700, sequence=3)]
    benchmark = slo_benchmark()
    facts = {
        "stop_reason": stop_reason,
        "bracket_reason": "throughput_decline",
        "slo_bracket": [64.0, 128.0],
    }

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["warnings"] == []
    assert validity["sufficient"] is True
    assert validity["slo_boundary_located"] is True


def decline_evidence():
    return {
        "threshold_percent": 15,
        "consecutive_points": 2,
        "peak_rate": 64,
        "peak_total_tps": 10000,
        "points": [
            {"rate": 128, "total_tps": 8200, "drop_percent": 18},
            {"rate": 256, "total_tps": 8000, "drop_percent": 20},
        ],
    }


def evidence_rows():
    return [
        point(64, 3000, total=10000, ttft=100),
        point(128, 4000, total=8200, ttft=200, sequence=1),
        point(256, 5000, total=8000, ttft=400, sequence=2),
    ]


def evidence_facts(**changes):
    return {
        "version": 6,
        "stop_reason": "throughput_decline",
        "bracket_reason": "throughput_decline",
        "slo_bracket": [256, None],
        "stopped_at": 256,
        "throughput_decline": decline_evidence(),
        **changes,
    }


def test_decline_evidence_and_policy_parameters_are_forwarded_verbatim():
    rows = evidence_rows()
    benchmark = slo_benchmark()
    best = analysis.compute_best_points(benchmark, rows)
    facts = evidence_facts()
    facts["throughput_decline"]["points"][0]["drop_percent"] = 18.123456

    validity = analysis.compute_validity(benchmark, rows, best, facts)

    assert validity["throughput_decline"] == facts["throughput_decline"]
    assert validity["warnings"] == [
        {"code": "throughput_decline", "params": {"points": 2, "threshold": 15}}
    ]
    assert validity["sufficient"] is False
    assert validity["slo_boundary_located"] is False
    assert best == {"peak_rate": 64.0, "recommended_rate": 256.0, "slo_met_rate": 256.0}
    assert validity["primary_point"] == coordinates(rows[2])
    assert validity["peak_point"] == coordinates(rows[0])


def test_decline_evidence_is_serialized_from_the_validated_model():
    facts = evidence_facts()
    recorded = analysis._throughput_decline_evidence(facts)

    assert recorded == facts["throughput_decline"]
    assert type(recorded["peak_total_tps"]) is float
    assert type(recorded["points"][0]["rate"]) is float
    assert type(facts["throughput_decline"]["points"][0]["rate"]) is int


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {},
        {**decline_evidence(), "threshold_percent": "15"},
        {**decline_evidence(), "consecutive_points": True},
        {**decline_evidence(), "consecutive_points": 3},
        {**decline_evidence(), "peak_total_tps": float("nan")},
        {**decline_evidence(), "points": []},
        {**decline_evidence(), "points": [{"rate": 128, "total_tps": 8200}]},
    ],
)
def test_missing_or_malformed_evidence_retains_the_generic_guard_warning(evidence):
    rows = evidence_rows()
    benchmark = slo_benchmark()
    facts = evidence_facts(throughput_decline=evidence)

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert "throughput_decline" not in validity
    assert validity["warnings"] == [{"code": "throughput_decline", "params": {}}]
    assert validity["sufficient"] is False


@pytest.mark.parametrize(
    "stop_reason", ["budget_points", "budget_seconds", "point_failed", "converged"]
)
def test_other_final_reasons_do_not_publish_decline_evidence(stop_reason):
    rows = evidence_rows()
    benchmark = slo_benchmark()
    facts = evidence_facts(stop_reason=stop_reason)

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["stop_reason"] == stop_reason
    assert "throughput_decline" not in validity
    assert "throughput_decline" not in [
        warning["code"] for warning in validity["warnings"]
    ]


@pytest.mark.parametrize("stop_reason", ["converged", "slo_failed"])
def test_a_completed_slo_boundary_does_not_publish_stale_decline_evidence(stop_reason):
    rows = evidence_rows() + [point(512, 5000, total=7500, ttft=700, sequence=3)]
    benchmark = slo_benchmark()
    facts = evidence_facts(stop_reason=stop_reason, slo_bracket=[256, 512])

    validity = analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )

    assert validity["warnings"] == []
    assert validity["slo_boundary_located"] is True
    assert "throughput_decline" not in validity


def gain_rows():
    return [
        point(rate, 1000, total=tps, sequence=i)
        for i, (rate, tps) in enumerate(
            [
                (16, 1600),
                (32, 3200),
                (64, 3300),
                (48, 3320),
                (40, 3310),
                (56, 3315),
            ]
        )
    ]


def gain_facts(**changes):
    return {
        "version": 7,
        "stop_reason": "throughput_gain_converged",
        "bracket_reason": "capacity_plateau",
        "throughput_gain_evidence": {
            "threshold_percent": 1,
            "window_points": 3,
            "baseline_rate": 64,
            "baseline_total_tps": 3300,
            "best_rate": 48,
            "best_total_tps": 3320,
            "gain_percent": 0.606061,
            "lower_rate": 40,
            "upper_rate": 56,
            "points": [
                {"rate": 48, "total_tps": 3320, "best_total_tps": 3320},
                {"rate": 40, "total_tps": 3310, "best_total_tps": 3320},
                {"rate": 56, "total_tps": 3315, "best_total_tps": 3320},
            ],
        },
        **changes,
    }


def gain_validity(rows, facts, **configuration):
    benchmark = SimpleNamespace(
        auto_tune=True, max_points=len(rows), upper_bound=64, **configuration
    )
    return analysis.compute_validity(
        benchmark, rows, analysis.compute_best_points(benchmark, rows), facts
    )


@pytest.mark.parametrize(
    "bracket", ["capacity_plateau", "budget_points", "upper_bound"]
)
def test_max_gain_convergence_is_a_normal_final_reason_with_verified_evidence(bracket):
    rows = gain_rows()
    facts = gain_facts(bracket_reason=bracket)

    validity = gain_validity(rows, facts)

    assert validity["warnings"] == []
    assert validity["sufficient"] is True
    assert validity["throughput_gain_evidence"] == facts["throughput_gain_evidence"]
    assert validity["primary_point"] == coordinates(rows[3])
    assert validity["peak_point"] == coordinates(rows[3])


@pytest.mark.parametrize("evidence", [None, {}])
def test_gain_reason_without_usable_evidence_does_not_invent_numbers(evidence):
    facts = gain_facts(
        throughput_gain_evidence=evidence, bracket_reason="budget_points"
    )

    validity = gain_validity(gain_rows(), facts)

    assert validity["warnings"] == []
    assert validity["sufficient"] is True
    assert validity["stop_reason"] == "throughput_gain_converged"
    assert "throughput_gain_evidence" not in validity


@pytest.mark.parametrize(
    "load_type,warning",
    [
        ("fixed_rate", "saturated_at_lower_bound"),
        ("concurrency", "peak_at_floor"),
    ],
)
def test_gain_reason_preserves_lower_edge_coverage_findings(load_type, warning):
    rows = [
        point(64, 5000, total=10000, requests_per_second_mean=23.6),
        point(128, 4500, total=9000, sequence=1),
        point(256, 4000, total=8000, sequence=2),
    ]
    validity = gain_validity(
        rows,
        gain_facts(throughput_gain_evidence=None, bracket_reason="budget_points"),
        load_type=load_type,
    )

    assert [item["code"] for item in validity["warnings"]] == [warning]
    assert validity["sufficient"] is False
    assert validity["stop_reason"] == "throughput_gain_converged"
    if load_type == "fixed_rate":
        assert validity["warnings"][0]["params"]["ceiling"] == 23.6


def test_gain_evidence_is_serialized_from_the_validated_model():
    facts = gain_facts()
    validity = gain_validity(gain_rows(), facts)
    recorded = validity["throughput_gain_evidence"]

    assert recorded == facts["throughput_gain_evidence"]
    assert type(recorded["baseline_total_tps"]) is float
    assert type(recorded["points"][0]["rate"]) is float
    assert type(facts["throughput_gain_evidence"]["points"][0]["rate"]) is int


@pytest.mark.parametrize(
    "stop", ["budget_points", "budget_seconds", "point_failed", "converged"]
)
def test_other_final_reasons_do_not_publish_stale_gain_evidence(stop):
    validity = gain_validity(gain_rows(), gain_facts(stop_reason=stop))

    assert validity["stop_reason"] == stop
    assert "throughput_gain_evidence" not in validity
    if stop.startswith("budget_"):
        assert validity["warnings"][0]["code"] == "budget_exhausted"


def test_gain_reason_cannot_complete_an_unlocated_slo_boundary():
    validity = gain_validity(gain_rows(), gain_facts(), slo_avg_ttft_ms=500)

    assert validity["slo_boundary_located"] is False
    assert validity["sufficient"] is False
    assert [warning["code"] for warning in validity["warnings"]] == ["not_saturated"]
    assert "throughput_gain_evidence" not in validity


@pytest.mark.parametrize(
    "case",
    [
        "threshold",
        "count",
        "nonfinite",
        "raw_total",
        "order",
        "baseline",
        "best",
        "running_best",
        "gain",
        "at_threshold",
        "lower",
        "upper",
        "sequence",
        "window_success",
        "baseline_success",
        "best_success",
    ],
)
def test_bad_gain_evidence_is_omitted_without_fabricating_a_budget_warning(case):
    rows = gain_rows()
    facts = gain_facts()
    evidence = facts["throughput_gain_evidence"]
    mutations = {
        "threshold": lambda rows, e: e.update(threshold_percent=0),
        "count": lambda rows, e: e.update(window_points=2),
        "nonfinite": lambda rows, e: e.update(gain_percent=float("nan")),
        "raw_total": lambda rows, e: rows[-1].update(tokens_per_second_mean=3333),
        "order": lambda rows, e: e["points"].reverse(),
        "baseline": lambda rows, e: e.update(baseline_rate=32, baseline_total_tps=3200),
        "best": lambda rows, e: e.update(best_rate=40, best_total_tps=3310),
        "running_best": lambda rows, e: e["points"][1].update(best_total_tps=3310),
        "gain": lambda rows, e: e.update(gain_percent=0.5),
        "at_threshold": lambda rows, e: e.update(threshold_percent=e["gain_percent"]),
        "lower": lambda rows, e: e.update(lower_rate=32),
        "upper": lambda rows, e: e.update(upper_rate=64),
        "sequence": lambda rows, e: rows[-1].update(sequence=rows[-2]["sequence"]),
        "window_success": lambda rows, e: rows[-1].update(request_successful=90),
        "baseline_success": lambda rows, e: rows[2].update(request_successful=90),
        "best_success": lambda rows, e: rows[3].update(request_successful=90),
    }
    mutations[case](rows, evidence)

    validity = gain_validity(rows, facts)

    assert validity["stop_reason"] == "throughput_gain_converged"
    assert "throughput_gain_evidence" not in validity
    assert not any(
        warning["code"] == "budget_exhausted" for warning in validity["warnings"]
    )


def test_gain_window_order_comes_from_sequence_not_result_list_or_load_order():
    rows = gain_rows()
    facts = gain_facts()

    validity = gain_validity(list(reversed(rows)), facts)

    assert validity["throughput_gain_evidence"] == facts["throughput_gain_evidence"]


def test_gain_baseline_ties_use_the_lower_load():
    rows = gain_rows()
    rows[1]["tokens_per_second_mean"] = 3300
    facts = gain_facts()
    facts["throughput_gain_evidence"]["baseline_rate"] = 32

    validity = gain_validity(rows, facts)

    assert validity["throughput_gain_evidence"]["baseline_rate"] == 32
    facts["throughput_gain_evidence"]["baseline_rate"] = 64
    assert "throughput_gain_evidence" not in gain_validity(rows, facts)


def test_gain_best_ties_match_the_primary_point_lower_load():
    rows = gain_rows()
    rows[3]["tokens_per_second_mean"] = 3300
    rows[4]["tokens_per_second_mean"] = 3290
    rows[5]["tokens_per_second_mean"] = 3295
    facts = gain_facts()
    evidence = facts["throughput_gain_evidence"]
    evidence.update(best_total_tps=3300, gain_percent=0)
    evidence["points"] = [
        {
            "rate": row["rate"],
            "total_tps": row["tokens_per_second_mean"],
            "best_total_tps": 3300,
        }
        for row in rows[-3:]
    ]

    validity = gain_validity(rows, facts)

    assert validity["throughput_gain_evidence"]["best_rate"] == 48
    assert validity["primary_point"] == coordinates(rows[3])
    evidence["best_rate"] = 64
    assert "throughput_gain_evidence" not in gain_validity(rows, facts)


def test_gain_completion_preserves_an_earlier_measurement_failure():
    rows = gain_rows()
    rows.insert(3, point(52, 100, total=500, sequence=3, request_successful=90))
    for i, row in enumerate(rows):
        row["sequence"] = i
    facts = gain_facts()

    validity = gain_validity(rows, facts)

    assert validity["throughput_gain_evidence"] == facts["throughput_gain_evidence"]
    assert [warning["code"] for warning in validity["warnings"]] == ["point_high_error"]
    assert validity["sufficient"] is False
