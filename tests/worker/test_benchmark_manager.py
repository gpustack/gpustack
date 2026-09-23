import time
from collections import deque
from types import SimpleNamespace

import pytest

import gpustack.worker.benchmark_manager as bm
from gpustack.schemas import benchmark as bm_schemas
from gpustack.worker.benchmark import analysis, artifacts
from gpustack.worker.benchmark.runner import BenchmarkRunner
from gpustack.worker.benchmark_manager import BenchmarkManager, CollectedResults


def _bare_manager(benchmark_dir="/tmp/does-not-matter"):
    """A BenchmarkManager with only the attributes the sync helpers touch, built
    without running __init__ (which needs a real Config + clientset). Collaborator
    methods are monkeypatched per-test."""
    mgr = object.__new__(BenchmarkManager)
    mgr._benchmark_dir = str(benchmark_dir)
    mgr._partial_synced_count = {}
    mgr._last_partial_sync_at = {}
    mgr._last_target_check_at = {}
    mgr._target_missing_since = {}
    return mgr


def _point(rate, tps, tpot, *, total=100, ok=None, **extra):
    """One row of the per-point result grid, as uploaded to benchmark_results.

    `tpot` lands on `inter_token_latency_mean`, which is where the decode-only
    per-token time lives (guidellm's naming) and what `slo_avg_tpot_ms` is
    evaluated against — see SLO_THRESHOLDS.
    """
    return {
        "rate": rate,
        "tokens_per_second_mean": tps,
        "inter_token_latency_mean": tpot,
        "request_total": total,
        "request_successful": total if ok is None else ok,
        **extra,
    }


# A real max-throughput sweep (shape taken from a 15-point run on qwen3-0.6b):
# throughput climbs to a peak around rate 31 and then collapses while TPOT
# explodes.
_SWEEP = [
    _point(1, 1199, 0.39),
    _point(2, 2350, 0.27),
    _point(4, 4600, 0.16),
    _point(8, 9100, 0.35),
    _point(16, 17800, 0.62),
    _point(24, 33000, 0.94),
    _point(31, 35961, 1.35),
    _point(40, 30000, 12.0),
    _point(55, 21000, 177.0),
]


class TestComputeBestPoints:
    def test_peak_and_recommendation_without_slo(self):
        benchmark = SimpleNamespace()
        out = analysis.compute_best_points(benchmark, _SWEEP)
        assert out["peak_rate"] == 31
        # No SLO => recommend the throughput peak, and report no SLO capacity.
        assert out["recommended_rate"] == 31
        assert "slo_met_rate" not in out

    def test_p95_threshold_is_evaluated(self):
        points = [
            _point(4, 4600, 0.16, time_to_first_token_p95=120.0),
            _point(8, 9100, 0.35, time_to_first_token_p95=480.0),
            _point(16, 17800, 0.62, time_to_first_token_p95=900.0),
        ]
        benchmark = SimpleNamespace(slo_p95_ttft_ms=500.0)
        out = analysis.compute_best_points(benchmark, points)
        # rate 16 breaches the p95 budget, so capacity tops out at 8.
        assert out["slo_met_rate"] == 8
        assert out["recommended_rate"] == 8

    def test_point_missing_the_p95_metric_never_meets_a_p95_threshold(self):
        benchmark = SimpleNamespace(slo_p95_tpot_ms=50.0)
        # Rows written before p95 was collected have no value to compare.
        assert not analysis.meets_slo(benchmark, _point(4, 4600, 0.16))


# Benchmark 74 (qwen3-0.6b, concurrency axis, SLO avg TTFT <= 10s / TPOT <= 1s):
# throughput peaks at 256 streams and then declines, while a threshold that loose
# is never violated anywhere in [4, 1024].
_BM74 = [
    _point(4, 1738.7, 0.15, time_to_first_token_mean=19.1),
    _point(8, 3155.4, 0.25, time_to_first_token_mean=32.1),
    _point(16, 4838.7, 0.28, time_to_first_token_mean=35.5),
    _point(32, 8742.4, 0.32, time_to_first_token_mean=41.3),
    _point(64, 13146.0, 0.49, time_to_first_token_mean=62.0),
    _point(128, 16343.2, 0.72, time_to_first_token_mean=92.4),
    _point(256, 17552.3, 1.11, time_to_first_token_mean=142.4),
    _point(512, 17447.6, 14.95, time_to_first_token_mean=1913.3),
    _point(1024, 16471.5, 45.38, time_to_first_token_mean=5809.1),
]

_BM74_BENCHMARK = dict(
    load_type="concurrency",
    lower_bound=4,
    upper_bound=1024,
    max_points=12,
    slo_avg_ttft_ms=10000.0,
    slo_avg_tpot_ms=1000.0,
)


class TestLooseSloFallsBackToThePeak:
    """A threshold too loose to ever break must not hand back the top of the range.

    Benchmark 74 reported concurrency 1024 as the SLO capacity: 6% LESS throughput
    than 256 at 5809ms TTFT instead of 142ms — a load nobody would choose to run at.
    Past saturation more load buys only queueing, so the operating point is capped
    at the throughput peak and the reason is stated instead of being implied.
    """

    def test_recommendation_is_capped_at_the_throughput_peak(self):
        benchmark = SimpleNamespace(**_BM74_BENCHMARK)
        out = analysis.compute_best_points(benchmark, _BM74)
        assert out["peak_rate"] == 256
        # The literal SLO answer is kept as measured...
        assert out["slo_met_rate"] == 1024
        # ...but the point to actually run at is the peak.
        assert out["recommended_rate"] == 256

    def test_a_binding_slo_below_the_peak_is_left_alone(self):
        # Tight TTFT budget: the boundary is at 128, well under the 256 peak, so the
        # cap must not move it.
        benchmark = SimpleNamespace(**{**_BM74_BENCHMARK, "slo_avg_ttft_ms": 100.0})
        out = analysis.compute_best_points(benchmark, _BM74)
        assert out["slo_met_rate"] == 128
        assert out["recommended_rate"] == 128

    def test_verdict_says_tighten_the_thresholds_not_raise_the_bound(self):
        benchmark = SimpleNamespace(**_BM74_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM74)
        v = analysis.compute_validity(benchmark, _BM74, best)
        codes = [w["code"] for w in v["warnings"]]
        assert codes == ["slo_not_binding"]
        # "Raise the upper bound" would only buy more points past the peak.
        assert "not_saturated" not in codes
        # The advice carries the peak, so the message can name a load.
        w = v["warnings"][0]
        assert w["params"]["rate"] == 256

    def test_slo_holding_at_the_top_of_a_still_climbing_curve_is_not_flagged(self):
        # Same loose SLO, but throughput never turned over: here "raise the upper
        # bound" IS the right advice, so the top-edge code must still win.
        points = [
            _point(r, r * 1000.0, 0.5, time_to_first_token_mean=50.0)
            for r in (4, 8, 16, 32)
        ]
        benchmark = SimpleNamespace(**{**_BM74_BENCHMARK, "upper_bound": 32})
        best = analysis.compute_best_points(benchmark, points)
        v = analysis.compute_validity(benchmark, points, best)
        codes = [w["code"] for w in v["warnings"]]
        assert codes == ["not_saturated"]


class TestComputeValidity:
    """`not_saturated` vs `budget_exhausted` vs `peak_at_floor`.

    All three describe "the best point sits at an edge of what we measured", but
    they need opposite advice, so the code must reflect WHICH limit ended the sweep.
    """

    def _climbing(self, rates):
        # Throughput still rising at every point: the curve never turned over.
        return [_point(r, r * 1000.0, 0.5) for r in rates]

    def test_hitting_the_search_range_says_raise_the_range(self):
        points = self._climbing([4, 8, 16, 32])
        benchmark = SimpleNamespace(upper_bound=32, max_points=12)
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 32})
        assert [w["code"] for w in v["warnings"]] == ["not_saturated"]
        assert v["sufficient"] is False

    def test_running_out_of_points_says_raise_the_budget(self):
        # Same shape, but the point budget — not the range — is what ran out.
        # Advising "raise the bound" here points at the one number that was never
        # the constraint: the sweep never got to use the range it already had.
        points = self._climbing([4, 8, 16, 32])
        benchmark = SimpleNamespace(upper_bound=1024, max_points=4)
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 32})
        assert [w["code"] for w in v["warnings"]] == ["budget_exhausted"]

    def test_best_point_at_the_bottom_says_lower_the_range(self):
        # Saturated from the first point: every knob reports the same throughput,
        # so the cheapest one wins and the optimum may be below the range.
        points = [
            _point(4, 2000.0, 0.5),
            _point(8, 2000.0, 0.9),
            _point(16, 2000.0, 2.0),
        ]
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 4})
        assert "peak_at_floor" in [w["code"] for w in v["warnings"]]

    def test_search_range_starting_above_saturation_says_lower_the_range(self):
        # lower_bound=64 against a server sustaining ~23.6 rps: the one measured
        # point outran it, so nothing inside [64, 128] can be the optimum.
        points = [
            _point(64, 28249.0, 0.5, requests_per_second_mean=23.6),
        ]
        benchmark = SimpleNamespace(
            load_type="fixed_rate", lower_bound=64, upper_bound=128, max_points=6
        )
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 64})
        codes = [w["code"] for w in v["warnings"]]
        assert "saturated_at_lower_bound" in codes
        # Must NOT also say "raise the upper bound" — the optimum is below the
        # LOWER bound, so that advice points the wrong way.
        assert "not_saturated" not in codes
        assert "budget_exhausted" not in codes
        # The advice carries a number, not just a direction.
        w = next(w for w in v["warnings"] if w["code"] == "saturated_at_lower_bound")
        assert w["params"]["ceiling"] == 23.6

    def test_server_keeping_up_at_the_floor_is_not_flagged(self):
        # achieved ~= offered at the lowest point => the range floor is fine, and
        # the top-edge verdict applies as usual.
        points = [
            _point(4, 4000.0, 0.2, requests_per_second_mean=3.9),
            _point(8, 8000.0, 0.3, requests_per_second_mean=7.8),
        ]
        benchmark = SimpleNamespace(
            load_type="fixed_rate", lower_bound=4, upper_bound=8, max_points=12
        )
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 8})
        codes = [w["code"] for w in v["warnings"]]
        assert "saturated_at_lower_bound" not in codes
        assert "not_saturated" in codes

    def test_concurrency_axis_is_not_judged_by_achieved_vs_offered(self):
        # On the concurrency axis the knob counts streams, so rps is not comparable
        # to it — 4 streams sustaining 2 rps is perfectly normal, not saturation.
        points = [
            _point(4, 2000.0, 0.5, requests_per_second_mean=2.0),
            _point(8, 2100.0, 0.9, requests_per_second_mean=2.1),
        ]
        benchmark = SimpleNamespace(
            load_type="concurrency", lower_bound=4, upper_bound=64, max_points=12
        )
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 4})
        codes = [w["code"] for w in v["warnings"]]
        assert "saturated_at_lower_bound" not in codes
        # peak_at_floor is the signal that survives on this axis.
        assert "peak_at_floor" in codes

    def test_a_turned_over_curve_is_clean(self):
        # Peak strictly inside the measured range = the sweep did its job.
        points = self._climbing([4, 8, 16]) + [
            _point(32, 30000.0, 3.0),
            _point(64, 20000.0, 12.0),
        ]
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        v = analysis.compute_validity(benchmark, points, {"recommended_rate": 32})
        assert v == {"sufficient": True, "warnings": []}


class TestBuildCommandArgs:
    def _runner(self, **benchmark_fields):
        benchmark = SimpleNamespace(
            id=1,
            # The name the target is asked for, rather than one discovered from
            # it — see `--model` in the runner.
            model_name="qwen3-0.6b",
            auto_tune=False,
            stages=None,
            load_type="fixed_rate",
            request_rate=10,
            total_requests=None,
            max_seconds=None,
            dataset_name="Random",
            dataset_input_tokens=128,
            dataset_output_tokens=128,
            dataset_input_stdev=None,
            dataset_input_min=None,
            dataset_input_max=None,
            dataset_output_stdev=None,
            dataset_output_min=None,
            dataset_output_max=None,
            dataset_seed=42,
            dataset_seed_increment=True,
            prefix_buckets=None,
            turns=None,
            warmup=None,
            cooldown=None,
            max_errors=None,
            max_error_rate=None,
            stop_on_saturation=None,
        )
        for key, value in benchmark_fields.items():
            setattr(benchmark, key, value)
        return SimpleNamespace(
            _benchmark=benchmark,
            _model_endpoint="http://127.0.0.1:8000",
            _model_path="/models/qwen3-0.6b",
            _model_backend_parameters=None,
            _benchmark_dir="/var/lib/gpustack/benchmarks",
            _api_url="http://127.0.0.1:80/v2/benchmarks/1/state",
            _api_key="token",
            # Plain-HTTP progress endpoint, so the TLS gating stays out of the way
            # of what these cover. The gating itself is exercised in
            # test_benchmark_tls.py.
            _progress_insecure_skip_tls_verify=False,
            _progress_is_https=False,
        )

    def test_manual_stages_carry_the_load_axis(self):
        stages = [{"rate": 2}, {"rate": 4}]
        args = BenchmarkRunner._build_command_args(
            self._runner(stages=stages, load_type="fixed_rate")
        )
        # Without --axis every stage would run as `concurrent`, turning a
        # fixed-rate stage list into a concurrency sweep.
        assert "--stages" in args
        assert args[args.index("--axis") + 1] == "rate"

    def test_manual_stages_on_the_concurrency_axis(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(stages=[{"rate": 8}], load_type="concurrency")
        )
        assert args[args.index("--axis") + 1] == "concurrency"

    def test_non_stage_run_gets_the_global_max_requests(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(load_type="fixed_rate", total_requests=500)
        )
        assert args[args.index("--max-requests") + 1] == "500"

    def test_stages_do_not_inherit_the_global_max_requests(self):
        # Each stage carries its own request cap in the --stages payload; a global
        # --max-requests would silently pin every stage that omitted its own to
        # total_requests (asymmetric with --max-seconds, which already guards).
        args = BenchmarkRunner._build_command_args(
            self._runner(
                load_type="fixed_rate",
                stages=[{"rate": 2}, {"rate": 4}],
                total_requests=500,
            )
        )
        assert "--max-requests" not in args

    def test_auto_tune_passes_p95_slo_thresholds(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(
                auto_tune=True,
                load_type="concurrency",
                lower_bound=4,
                upper_bound=64,
                max_points=6,
                max_total_seconds=300,
                slo_p95_ttft_ms=500.0,
                slo_p95_tpot_ms=None,
                slo_avg_ttft_ms=None,
                slo_avg_tpot_ms=None,
                slo_p99_ttft_ms=None,
                slo_p99_tpot_ms=None,
                slo_avg_latency_ms=None,
                slo_p95_latency_ms=None,
                slo_p99_latency_ms=None,
            )
        )
        assert args[args.index("--slo-p95-ttft-ms") + 1] == "500.0"

    def test_max_error_rate_is_forwarded_only_when_guidellm_accepts_it(self):
        # guidellm's MaxErrorRateConstraint takes a fraction in (0, 1) and rejects
        # the endpoints. 1.0 ("tolerate everything") used to be forwarded verbatim
        # and killed the run at scenario construction, before any request.
        def rate_arg(value):
            args = BenchmarkRunner._build_command_args(
                self._runner(max_error_rate=value)
            )
            return (
                args[args.index("--max-error-rate") + 1]
                if "--max-error-rate" in args
                else None
            )

        assert rate_arg(0.5) == "0.5"
        assert rate_arg(1.0) is None  # "no ceiling" == omit the constraint
        assert rate_arg(0) is None  # not expressible; max_errors covers it
        assert rate_arg(None) is None

    def test_backend_uses_the_native_path_keyed_request_handlers(self):
        args = BenchmarkRunner._build_command_args(self._runner())
        backend_kwargs = args[args.index("--backend-kwargs") + 1]
        assert '"/v1/chat/completions"' in backend_kwargs
        assert "response_handlers" not in backend_kwargs


class TestCountReadyPointFiles:
    """The count that gates a partial sync: how many finished point/stage files
    are on disk. Must ignore the throughput probe and the `.full` companions, or
    it would fire on artifacts that aren't measured points."""

    def test_auto_tune_counts_only_point_summaries(self, tmp_path):
        for name in (
            "1__p0.json",
            "1__p1.json",
            "1__p2.json",
            "1__p0.full.json",  # companion, not a point
            "1__satprobe.json",  # saturation probe, not a point
            "1__satprobe.full.json",
            "2__p0.json",  # a different benchmark
        ):
            (tmp_path / name).write_text("{}")
        mgr = _bare_manager(tmp_path)
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None)
        assert mgr._count_ready_point_files(benchmark) == 3

    def test_stages_count_only_the_files_that_exist(self, tmp_path):
        # Stage 1 hasn't finished yet, so only 0 and 2 are on disk.
        (tmp_path / "5__stage0.json").write_text("{}")
        (tmp_path / "5__stage2.json").write_text("{}")
        mgr = _bare_manager(tmp_path)
        benchmark = SimpleNamespace(id=5, auto_tune=False, stages=[{}, {}, {}])
        assert mgr._count_ready_point_files(benchmark) == 2


class TestMaybeSyncPartialMetrics:
    """Gating around streaming finished points mid-run. The collaborators
    (_collect_results / _post_metrics_and_results / _write_best_points_and_validity
    / _count_ready_point_files) are stubbed; these tests pin the DECISION logic:
    who gets synced, when, and how the counters advance."""

    @staticmethod
    def _rows(n):
        """Point rows shaped like `to_results` returns them: columns + the bulky
        per-point dump the partial sync is expected to leave behind."""
        return [
            {"sequence": i, "rate": float(4 * 2**i), "raw_metrics": {"dump": i}}
            for i in range(n)
        ]

    def _wire(self, mgr, *, on_disk, loaded, results=None, metrics=None):
        calls = SimpleNamespace(posted=None, wrote=None, collected=False)

        def collect(_benchmark):
            calls.collected = True
            return CollectedResults(
                results=results if results is not None else self._rows(loaded),
                metrics=metrics if metrics is not None else SimpleNamespace(),
                report=None,
                loaded=loaded,
                skipped=0,
            )

        def post(_benchmark, _metrics, res, attempts=3):
            calls.posted = res

        def write(_benchmark, _res, in_progress=False, attempts=3):
            calls.wrote = (in_progress, attempts)

        mgr._count_ready_point_files = lambda _b: on_disk
        mgr._collect_results = collect
        mgr._post_metrics_and_results = post
        mgr._write_best_points_and_validity = write
        return calls

    def test_single_fixed_run_is_never_streamed(self):
        mgr = _bare_manager()
        touched = []
        mgr._count_ready_point_files = lambda _b: touched.append(1) or 5
        benchmark = SimpleNamespace(id=1, auto_tune=False, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        # Bailed on the axis check, before even counting files.
        assert touched == []

    def test_time_throttle_blocks_a_too_soon_sync(self, monkeypatch):
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        mgr._last_partial_sync_at[1] = 995.0  # 5s ago < 10s interval
        touched = []
        mgr._count_ready_point_files = lambda _b: touched.append(1) or 3
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        assert touched == []  # returned on the time gate, before counting

    def test_no_new_points_means_no_upload(self, monkeypatch):
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        mgr._partial_synced_count[1] = 3
        calls = self._wire(mgr, on_disk=3, loaded=3)  # equal => nothing new
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        assert calls.collected is False
        assert calls.posted is None

    def test_growth_uploads_and_flags_in_progress(self, monkeypatch):
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        calls = self._wire(mgr, on_disk=2, loaded=2)
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        assert [r["sequence"] for r in calls.posted] == [0, 1]
        assert [r["rate"] for r in calls.posted] == [4.0, 8.0]
        # The in-progress analysis is written with a single attempt (a blip just
        # retries next poll) and tagged so the UI can show it's still firming up.
        assert calls.wrote == (True, 1)
        assert mgr._partial_synced_count[1] == 2
        assert mgr._last_partial_sync_at[1] == 1000.0

    def test_the_bulky_dumps_stay_home_until_the_run_is_over(self, monkeypatch):
        # The whole grid is re-posted every 10s for the life of the run, and each
        # POST is a full replace, so shipping every point's raw_metrics again on
        # every tick is the one part of this that grows with max_points. The
        # columns still go (they are what the running page draws); the dumps are
        # the terminal sync's job.
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        calls = self._wire(mgr, on_disk=2, loaded=2)
        mgr._maybe_sync_partial_metrics(
            SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        )
        assert all("raw_metrics" not in row for row in calls.posted)
        assert all("rate" in row for row in calls.posted)

    def test_stripping_the_dumps_does_not_mutate_the_analysis_input(self, monkeypatch):
        # The same rows go on to _write_best_points_and_validity; the strip has to
        # be a copy, or the analysis would be reading a grid we quietly edited.
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        rows = self._rows(2)
        calls = self._wire(mgr, on_disk=2, loaded=2, results=rows)
        analysed = []
        mgr._write_best_points_and_validity = (
            lambda _b, res, in_progress=False, attempts=3: analysed.append(res)
        )
        mgr._maybe_sync_partial_metrics(
            SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        )
        assert all("raw_metrics" in row for row in rows)
        assert analysed == [rows]
        assert all("raw_metrics" not in row for row in calls.posted)

    def test_a_point_still_mid_write_is_retried_next_tick(self, monkeypatch):
        # 3 files on disk but the newest failed to parse, so only 2 loaded. The
        # counter must advance to the LOADED count, not the on-disk count, so the
        # next poll (on_disk 3 > synced 2) picks the point up once it's whole.
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        self._wire(mgr, on_disk=3, loaded=2)
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        assert mgr._partial_synced_count[1] == 2

    def test_a_sync_failure_neither_raises_nor_advances_the_progress(self, monkeypatch):
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        mgr._partial_synced_count[1] = 1
        mgr._count_ready_point_files = lambda _b: 2

        def boom(_benchmark):
            raise RuntimeError("api down")

        mgr._collect_results = boom
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)  # must not propagate
        assert mgr._partial_synced_count[1] == 1  # nothing was synced
        # The THROTTLE, on the other hand, is stamped on the attempt: it bounds how
        # often the work is retried, and a failure that left it unstamped would
        # re-parse every point already synced on each 3s poll. The progress counter
        # above is what decides whether there is anything new to do.
        assert mgr._last_partial_sync_at[1] == 1000.0

    def test_a_mid_write_point_does_not_busy_loop_the_reparse(self, monkeypatch):
        # 3 on disk, 2 loadable: the on-disk count stays ahead of the synced count
        # for as long as the newest file is being written, which is exactly the
        # condition that would re-trigger a full re-parse every poll.
        monkeypatch.setattr(bm.time, "time", lambda: 1000.0)
        mgr = _bare_manager()
        self._wire(mgr, on_disk=3, loaded=2)
        benchmark = SimpleNamespace(id=1, auto_tune=True, stages=None, name="x")
        mgr._maybe_sync_partial_metrics(benchmark)
        assert mgr._last_partial_sync_at[1] == 1000.0

        # A poll 3s later is inside the throttle window and must not collect again.
        monkeypatch.setattr(bm.time, "time", lambda: 1003.0)
        calls = self._wire(mgr, on_disk=3, loaded=3)
        mgr._maybe_sync_partial_metrics(benchmark)
        assert calls.collected is False


class TestWriteBestPointsAndValidity:
    """The in_progress tag is a transient UI hint, present only on partial
    writes and absent from the terminal one."""

    def _mgr(self, captured, monkeypatch, warnings=()):
        # The analysis itself is a pure module function, stubbed here so these tests
        # pin only what the manager does with its output.
        monkeypatch.setattr(
            analysis, "compute_best_points", lambda _b, _r: {"peak_rate": 5}
        )
        # The 4th argument is the ramp facts read off disk (None here: no sidecar).
        monkeypatch.setattr(
            analysis,
            "compute_validity",
            lambda _b, _r, _bp, _ramp=None: {
                "sufficient": not warnings,
                "warnings": list(warnings),
            },
        )
        mgr = object.__new__(BenchmarkManager)
        mgr._read_ramp_facts = lambda _b: None
        mgr._update_benchmark_state_sync = lambda _id, **kw: captured.update(kw)
        return mgr

    def test_partial_write_tags_in_progress(self, monkeypatch):
        captured = {}
        mgr = self._mgr(captured, monkeypatch)
        benchmark = SimpleNamespace(id=1, name="x")
        msg = mgr._write_best_points_and_validity(benchmark, [], in_progress=True)
        assert msg is None
        assert captured["peak_rate"] == 5
        assert captured["validity"]["in_progress"] is True

    def test_terminal_write_leaves_no_in_progress_flag(self, monkeypatch):
        captured = {}
        mgr = self._mgr(captured, monkeypatch)
        benchmark = SimpleNamespace(id=1, name="x")
        mgr._write_best_points_and_validity(benchmark, [])
        assert "in_progress" not in captured["validity"]

    def test_every_conclusion_field_is_sent_so_none_goes_stale(self, monkeypatch):
        # A partial sync may have written an slo_met_rate that the final read no
        # longer supports. Omitting the key would leave that number on the row next
        # to a validity that contradicts it, so all three are always patched.
        captured = {}
        mgr = self._mgr(captured, monkeypatch)
        mgr._write_best_points_and_validity(SimpleNamespace(id=1, name="x"), [])
        assert captured["peak_rate"] == 5
        assert captured["slo_met_rate"] is None
        assert captured["recommended_rate"] is None

    def _mgr_with_warnings(self, captured, monkeypatch, warnings):
        return self._mgr(captured, monkeypatch, warnings=warnings)

    def test_partial_write_publishes_the_full_analysis(self, monkeypatch):
        # The partial analysis is a SNAPSHOT, not a verdict: "as of N points the
        # curve has not turned over" is a true statement about what was measured,
        # so it is persisted in full and only LABELLED provisional. Hiding it is the
        # UI's job (banner hidden + Coverage column "-" while in_progress), which
        # also fails safe: a code added later is hidden by default instead of having
        # to be added to a backend suppress-list.
        captured = {}
        mgr = self._mgr_with_warnings(
            captured,
            monkeypatch,
            [
                {"code": "not_saturated", "params": {}},
                {"code": "few_points", "params": {}},
            ],
        )
        mgr._write_best_points_and_validity(
            SimpleNamespace(id=1, name="x"), [], in_progress=True
        )
        codes = [w["code"] for w in captured["validity"]["warnings"]]
        assert codes == ["not_saturated", "few_points"]
        assert captured["validity"]["sufficient"] is False
        assert captured["validity"]["in_progress"] is True


class TestQueueCancelGuard:
    """A benchmark stopped/deleted while it sits QUEUED must never be started
    (regression for the worker picking up a canceled run once the GPU frees up).
    """

    def _manager(self):
        mgr = object.__new__(BenchmarkManager)
        mgr._benchmark_queue = deque()
        mgr._canceled_ids = set()
        mgr._torn_down_ids = bm.OrderedDict()
        mgr._provisioning_processes = {}
        mgr._benchmark_by_id = {}
        mgr._container_log_offset = {}
        mgr._last_log_snapshot_at = {}
        mgr._partial_synced_count = {}
        mgr._last_partial_sync_at = {}
        mgr._last_target_check_at = {}
        mgr._target_missing_since = {}
        mgr._active_benchmark_id = None
        mgr._is_provisioning = lambda _b: False
        mgr._clear_active_benchmark = lambda _i: None
        return mgr

    def test_stop_does_not_mutate_the_queue(self, monkeypatch):
        # _stop_benchmark runs on the sync thread (completion/failure/timeout) as
        # well as the event loop, so it must NOT rebuild/reassign _benchmark_queue
        # (that would race the loop's append and could drop a just-queued entry).
        # Cancellation of a queued benchmark is handled by the _canceled_ids guard
        # in the queue worker, not by mutating the deque here.
        monkeypatch.setattr(bm, "delete_workload", lambda _name: None)
        mgr = self._manager()
        b1 = SimpleNamespace(id=1, name="a")
        b2 = SimpleNamespace(id=2, name="b")
        mgr._benchmark_queue.extend([b1, b2])
        queue_before = mgr._benchmark_queue

        mgr._stop_benchmark(b1)

        # Same object, same contents — untouched.
        assert mgr._benchmark_queue is queue_before
        assert [b.id for b in mgr._benchmark_queue] == [1, 2]

    def test_a_failed_workload_delete_still_frees_the_queue(self, monkeypatch):
        # The wedge this guards against: delete_workload raises, so the cleanup below
        # it never runs, _active_benchmark_id stays set and the queue worker stops
        # popping ANYTHING. On the completion path the row is already COMPLETED and
        # the state poll only scans RUNNING, so nothing revisits it — a worker
        # restart would be the only way out. A leaked workload is the lesser evil.
        monkeypatch.setattr(
            bm, "delete_workload", lambda _name: (_ for _ in ()).throw(RuntimeError())
        )
        mgr = self._manager()
        cleared = []
        mgr._clear_active_benchmark = lambda i: cleared.append(i)
        mgr._active_benchmark_id = 1
        mgr._benchmark_by_id[1] = SimpleNamespace(id=1, name="a")
        mgr._partial_synced_count[1] = 3

        mgr._stop_benchmark(SimpleNamespace(id=1, name="a"))

        assert cleared == [1]
        assert 1 not in mgr._benchmark_by_id
        assert 1 not in mgr._partial_synced_count

    def test_a_failed_process_teardown_still_frees_the_queue(self, monkeypatch):
        # Same argument one step earlier: the provisioning kill is also on the path
        # to the cleanup, so it cannot be the thing that skips it.
        monkeypatch.setattr(bm, "delete_workload", lambda _name: None)
        monkeypatch.setattr(
            bm,
            "terminate_process_tree",
            lambda _pid: (_ for _ in ()).throw(RuntimeError()),
        )
        mgr = self._manager()
        cleared = []
        mgr._clear_active_benchmark = lambda i: cleared.append(i)
        mgr._is_provisioning = lambda _b: True
        mgr._provisioning_processes[1] = SimpleNamespace(pid=4242)

        mgr._stop_benchmark(SimpleNamespace(id=1, name="a"))

        assert cleared == [1]
        assert 1 not in mgr._provisioning_processes

    def test_queue_worker_skips_a_canceled_benchmark(self, monkeypatch):
        # The real cancel path: a benchmark stopped/deleted while QUEUED stays in
        # the deque, and the worker drops it (never starts it) via _canceled_ids.
        import asyncio

        class _Stop(Exception):
            pass

        async def _stop_sleep(_seconds):
            # Break the worker's `while True` once the queue has drained.
            raise _Stop

        monkeypatch.setattr(asyncio, "sleep", _stop_sleep)

        mgr = self._manager()
        mgr._queue_lock = asyncio.Lock()
        started = []

        async def fake_start(b):
            started.append(b.id)

        mgr._start_benchmark = fake_start
        canceled = SimpleNamespace(id=1, name="canceled")
        normal = SimpleNamespace(id=2, name="normal")
        mgr._benchmark_queue.extend([canceled, normal])
        mgr._canceled_ids.add(1)

        with pytest.raises(_Stop):
            asyncio.run(mgr._benchmark_queue_worker())

        # Canceled id 1 was dropped, only the normal one started.
        assert started == [2]
        assert 1 not in mgr._canceled_ids


class _Rep:
    """A stand-in for GenerativeBenchmarksReport: one measured point that can be
    told to blow up during conversion, or to carry a given error count."""

    def __init__(self, tps, errored=0, incomplete=0, fail_convert=False):
        self._tps = tps
        self._errored = errored
        self._incomplete = incomplete
        self._fail_convert = fail_convert

    def to_results(self, input_tokens, sequence_start):
        if self._fail_convert:
            raise ValueError("conversion boom")
        return [{"rate": 1, "tokens_per_second_mean": self._tps}]

    def to_metrics(self):
        return SimpleNamespace(
            tokens_per_second_mean=self._tps,
            request_errored=self._errored,
            request_incomplete=self._incomplete,
        )


class _ProbeRep:
    """Stand-in for the saturation-probe report: a throughput run whose row has no
    rate and an inflated throughput (unbounded concurrency)."""

    def to_results(self, input_tokens, sequence_start):
        return [
            {
                "rate": None,
                "strategy_type": "throughput",
                "tokens_per_second_mean": 99999.0,
                "requests_per_second_mean": 19.68,
                "sequence": sequence_start,
            }
        ]


class TestCollectResultsResilience:
    """A single bad point must not sink the whole aggregation (#2), and the
    report used for failure samples must be the point with the most failures,
    not the first (#4)."""

    def _benchmark(self, tmp_path, n):
        for i in range(n):
            (tmp_path / f"5__p{i}.json").write_text("{}")
        return SimpleNamespace(
            id=5, name="x", auto_tune=True, stages=None, dataset_input_tokens=100
        )

    def _patch_loader(self, monkeypatch, reps):
        monkeypatch.setattr(
            bm.GenerativeBenchmarksReport,
            "load_file",
            lambda path: reps[path.rsplit("/", 1)[1]],
        )

    def test_a_point_that_fails_to_convert_is_skipped_not_fatal(
        self, tmp_path, monkeypatch
    ):
        b = self._benchmark(tmp_path, 3)
        self._patch_loader(
            monkeypatch,
            {
                "5__p0.json": _Rep(10),
                "5__p1.json": _Rep(999, fail_convert=True),  # loads, fails convert
                "5__p2.json": _Rep(20),
            },
        )
        mgr = _bare_manager(tmp_path)
        collected = mgr._collect_results(b)
        assert collected.loaded == 2  # the failing point is skipped, not fatal
        assert collected.skipped == 1  # ...and counted, so the terminal sync can say so
        assert len(collected.results) == 2
        # best of the two good points
        assert collected.metrics.tokens_per_second_mean == 20

    def test_report_for_samples_is_the_point_with_most_failures(
        self, tmp_path, monkeypatch
    ):
        b = self._benchmark(tmp_path, 3)
        worst = _Rep(15, errored=8)  # high-load point where failures land
        self._patch_loader(
            monkeypatch,
            {
                "5__p0.json": _Rep(10, errored=0),
                "5__p1.json": _Rep(20, errored=0),
                "5__p2.json": worst,
            },
        )
        mgr = _bare_manager(tmp_path)
        assert mgr._collect_results(b).report is worst

    def test_saturation_probe_is_a_trailing_row_not_a_ramp_point(
        self, tmp_path, monkeypatch
    ):
        b = self._benchmark(tmp_path, 2)  # 5__p0, 5__p1
        (tmp_path / "5__satprobe.json").write_text("{}")
        self._patch_loader(
            monkeypatch,
            {
                "5__p0.json": _Rep(10),
                "5__p1.json": _Rep(20),
                "5__satprobe.json": _ProbeRep(),
            },
        )
        mgr = _bare_manager(tmp_path)
        collected = mgr._collect_results(b)
        results, metrics = collected.results, collected.metrics
        assert collected.loaded == 2  # the probe is not counted as a ramp point
        assert len(results) == 3  # ...but it IS surfaced as a trailing row
        probe = results[-1]
        assert probe["rate"] is None and probe["strategy_type"] == "throughput"
        assert probe["requests_per_second_mean"] == 19.68  # the measured ceiling
        # its inflated throughput must NOT become the representative peak
        assert metrics.tokens_per_second_mean == 20


class TestTerminalSyncRobustness:
    """A finished run must always be torn down, and its representative metrics
    POST must be retried, so an API blip can't strand it as COMPLETED-but-empty
    with a leaked workload (#1)."""

    def test_completion_stops_even_when_metrics_sync_fails(self):
        mgr = object.__new__(BenchmarkManager)
        stopped = []
        messages = []
        mgr._update_benchmark_state_sync = lambda _id, **k: messages.append(k)
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._truncate_state_message = lambda m: m

        def boom(_b):
            raise RuntimeError("api down")

        mgr._sync_benchmark_metrics = boom
        mgr._stop_benchmark = lambda b: stopped.append(b.id)

        mgr._handle_benchmark_completion(SimpleNamespace(id=7, name="x"))

        assert stopped == [7]  # teardown ran despite the sync failure
        # Still marked COMPLETED, and in the SAME patch as the failure message
        # (one round-trip instead of two). "sync" rather than "upload": the same
        # handler now also reports result files that could not be READ.
        assert messages == [
            {
                "state": bm.BenchmarkStateEnum.COMPLETED,
                "state_message": (
                    "Result sync failed: api down. See worker logs for details."
                ),
            }
        ]

    def test_the_final_analysis_lands_before_the_row_reads_completed(self):
        # Ordering regression. With the state patch first, there was a window (and
        # a permanent state on upload failure) where state said "completed" while
        # `validity` was still the last PARTIAL snapshot — an `in_progress` verdict
        # computed from a subset of points, e.g. "raise the upper bound and re-run".
        order = []
        mgr = object.__new__(BenchmarkManager)
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._truncate_state_message = lambda m: m
        mgr._sync_benchmark_metrics = lambda _b: order.append("analysis")
        mgr._update_benchmark_state_sync = lambda _id, **k: order.append(
            f"state:{k.get('state')}"
        )
        mgr._stop_benchmark = lambda _b: order.append("teardown")

        mgr._handle_benchmark_completion(SimpleNamespace(id=7, name="x"))

        assert order == [
            "analysis",
            f"state:{bm.BenchmarkStateEnum.COMPLETED}",
            "teardown",
        ]

    def test_teardown_survives_a_failed_state_patch(self):
        # If even the COMPLETED patch fails the workload must still be deleted:
        # sync_benchmark_state only polls RUNNING rows, so an un-torn-down run
        # leaks its container.
        stopped = []
        mgr = object.__new__(BenchmarkManager)
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._truncate_state_message = lambda m: m
        mgr._sync_benchmark_metrics = lambda _b: None
        mgr._stop_benchmark = lambda b: stopped.append(b.id)

        def boom(_id, **_k):
            raise RuntimeError("api down")

        mgr._update_benchmark_state_sync = boom

        mgr._handle_benchmark_completion(SimpleNamespace(id=9, name="x"))
        assert stopped == [9]

    def test_metrics_post_is_retried(self, monkeypatch):
        monkeypatch.setattr(bm.time, "sleep", lambda _s: None)  # no backoff wait
        monkeypatch.setattr(bm, "raise_if_response_error", lambda _r: None)
        calls = {"metrics": 0}

        class _Httpx:
            def post(self, url, json=None):
                if "/metrics" in url:
                    calls["metrics"] += 1
                    if calls["metrics"] < 3:
                        raise RuntimeError("blip")
                return object()

        mgr = object.__new__(BenchmarkManager)
        # _clientset is a read-only property backed by _clientset_getter.
        mgr._clientset_getter = lambda: SimpleNamespace(
            http_client=SimpleNamespace(get_httpx_client=lambda: _Httpx())
        )
        metrics = SimpleNamespace(model_dump=lambda: {})

        mgr._post_metrics_and_results(SimpleNamespace(id=7, name="x"), metrics, [])

        assert calls["metrics"] == 3  # retried twice, then succeeded


class TestValidityWarningNoise:
    """`few_points` is the weakest signal and must not dilute a primary cause
    (round 5/6 UX: it kept tagging along after saturated_at_lower_bound /
    peak_at_floor)."""

    def test_few_points_suppressed_when_a_primary_cause_exists(self):
        # fixed-rate single point where the server plainly can't keep up ->
        # saturated_at_lower_bound is the cause to act on; no few_points noise.
        pts = [_point(64, 100.0, 0.5, requests_per_second_mean=23.6)]
        benchmark = SimpleNamespace(
            load_type="fixed_rate", upper_bound=128, max_points=12
        )
        v = analysis.compute_validity(
            benchmark, pts, {"peak_rate": 64, "recommended_rate": 64}
        )
        codes = [w["code"] for w in v["warnings"]]
        assert codes == ["saturated_at_lower_bound"]
        assert "few_points" not in codes

    def test_few_points_still_reported_when_it_is_the_only_signal(self):
        # A short run with nothing stronger to say still gets the hint.
        pts = [_point(8, 9000.0, 0.3)]
        benchmark = SimpleNamespace(
            load_type="concurrency", upper_bound=1024, max_points=12
        )
        v = analysis.compute_validity(benchmark, pts, {})
        assert [w["code"] for w in v["warnings"]] == ["few_points"]


class TestSaturatedAtLowerBoundClearsBestPoints:
    """When the whole search range sits above saturation, the single measured
    'peak' is the offered floor knob the banner calls NOT the optimum; the cards
    must be cleared so the number can't contradict the advice (round 6 UX-2)."""

    def test_best_point_cards_are_cleared(self, tmp_path):
        # A real analysis (not stubbed) over one saturated point, so the write path
        # and the verdict are exercised together. tmp_path holds no ramp sidecar, so
        # the grid fallback is what decides.
        mgr = _bare_manager(tmp_path)
        captured = {}
        mgr._update_benchmark_state_sync = lambda _id, **k: captured.update(k)
        benchmark = SimpleNamespace(
            id=9,
            name="x",
            load_type="fixed_rate",
            auto_tune=True,
            upper_bound=128,
            max_points=12,
        )
        results = [_point(64, 100.0, 0.5, requests_per_second_mean=23.6)]

        msg = mgr._write_best_points_and_validity(benchmark, results)

        assert msg is None
        # peak/recommended nulled so the number can't contradict the banner...
        assert captured["peak_rate"] is None
        assert captured["recommended_rate"] is None
        # ...while the validity still carries the ceiling to act on.
        codes = [w["code"] for w in captured["validity"]["warnings"]]
        assert "saturated_at_lower_bound" in codes


# Benchmark 86 (round 7): loose SLO (10s TTFT / 1s TPOT), concurrency axis. The
# ramp stops at 512 on the <5% plateau; 512 is the throughput argmax and meets the
# loose SLO, so it is both the peak and the recommendation. The <5% top step keeps
# the run from being mislabeled range-limited.
_BM86 = [
    _point(4, 1738.7, 0.15, time_to_first_token_mean=19.1),
    _point(8, 3155.4, 0.25, time_to_first_token_mean=32.1),
    _point(16, 4838.7, 0.28, time_to_first_token_mean=35.5),
    _point(32, 8742.4, 0.32, time_to_first_token_mean=41.3),
    _point(64, 13146.0, 0.49, time_to_first_token_mean=62.0),
    _point(128, 16144.1, 0.72, time_to_first_token_mean=92.4),
    _point(256, 17184.6, 1.27, time_to_first_token_mean=160.7),
    _point(512, 17427.6, 14.98, time_to_first_token_mean=1917.2),  # +1.4% plateau
]

# Benchmark 93 (round 7 follow-up): a REAL latency SLO (500ms TTFT / 50ms TPOT),
# concurrency axis. Every point meets it and throughput is still climbing at the
# top (256 = argmax, TTFT 155ms << 500ms). The answer to "max load within my SLO"
# is 256 — the knee cap wrongly returned 128.
_BM93 = [
    _point(4, 3536.0, 0.17, time_to_first_token_mean=22.1),
    _point(8, 6482.8, 0.25, time_to_first_token_mean=32.2),
    _point(16, 10106.1, 0.26, time_to_first_token_mean=33.0),
    _point(32, 18574.1, 0.28, time_to_first_token_mean=36.0),
    _point(64, 27649.6, 0.48, time_to_first_token_mean=61.4),
    _point(128, 34382.6, 0.84, time_to_first_token_mean=107.4),
    _point(256, 35731.2, 1.21, time_to_first_token_mean=155.0),
]


_BM93_BENCHMARK = dict(
    load_type="concurrency",
    upper_bound=1024,
    max_points=12,
    slo_avg_ttft_ms=500.0,
    slo_avg_tpot_ms=50.0,
)


class TestRampFactsBeatInference:
    """The ramp reports WHY it stopped; the grid can only be asked to guess.

    Several terminations leave identical grids, so these are the cases where the
    inference path is not merely less precise but cannot be right:

    * `budget_seconds` vs a self-directed stop — both leave "fewer points than the
      range allows". The grid path reported `not_saturated` ("raise the upper
      bound") for a run the clock ended.
    * `capacity_plateau` vs an SLO binding at the peak — both leave "the highest
      knob measured met the SLO".
    """

    def _facts(self, bracket, **kw):
        return {"version": 1, "bracket_reason": bracket, "stop_reason": bracket, **kw}

    def test_the_time_cap_is_no_longer_reported_as_a_range_problem(self):
        # Still climbing at the top, points unspent (4 of 12): the grid concludes
        # "raise the upper bound", which was never the constraint.
        points = [_point(r, r * 1000.0, 0.5) for r in (4, 8, 16, 32)]
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        best = {"recommended_rate": 32}

        inferred = analysis.compute_validity(benchmark, points, best)
        assert [w["code"] for w in inferred["warnings"]] == ["not_saturated"]

        told = analysis.compute_validity(
            benchmark, points, best, self._facts("budget_seconds", stopped_at=32.0)
        )
        assert [w["code"] for w in told["warnings"]] == ["budget_exhausted"]
        assert told["warnings"][0]["params"]["which"] == "seconds"

    def test_the_point_cap_names_itself(self):
        points = [_point(r, r * 1000.0, 0.5) for r in (4, 8, 16, 32)]
        v = analysis.compute_validity(
            SimpleNamespace(upper_bound=1024, max_points=4),
            points,
            {"recommended_rate": 32},
            self._facts("budget_points"),
        )
        assert [w["code"] for w in v["warnings"]] == ["budget_exhausted"]
        assert v["warnings"][0]["params"]["which"] == "points"

    def test_reaching_the_range_ceiling_still_says_raise_the_range(self):
        points = [_point(r, r * 1000.0, 0.5) for r in (4, 8, 16, 32)]
        v = analysis.compute_validity(
            SimpleNamespace(upper_bound=32, max_points=12),
            points,
            {"recommended_rate": 32},
            self._facts("upper_bound"),
        )
        assert [w["code"] for w in v["warnings"]] == ["not_saturated"]

    def test_capacity_plateau_is_taken_as_fact_not_re_derived(self):
        # bm102: the grid path needs the 0.7 headroom heuristic to reach this
        # verdict; with the facts present the threshold is not consulted at all.
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(
            benchmark, _BM93, best, self._facts("capacity_plateau", stopped_at=256.0)
        )
        assert [w["code"] for w in v["warnings"]] == ["slo_not_binding"]

    def test_an_slo_boundary_reported_by_the_ramp_is_not_flagged(self):
        # Same grid, same loose-looking rates — but the ramp says a THRESHOLD ended
        # the bracket, so this is a real latency boundary and there is nothing to
        # report. The grid alone would still call it capacity-bound (31% headroom
        # + a flat top step), which is exactly the ambiguity the facts remove.
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(
            benchmark, _BM93, best, self._facts("slo_failed", stopped_at=256.0)
        )
        assert v["warnings"] == []
        assert v["sufficient"] is True

    def test_the_stop_reason_rides_along_for_the_ui(self):
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(
            benchmark, _BM93, best, self._facts("capacity_plateau", stopped_at=256.0)
        )
        # Present whether or not it earned a warning: the detail page states where
        # the search ended and why.
        assert v["stop_reason"] == "capacity_plateau"
        assert v["stopped_at"] == 256.0

    def test_the_two_reasons_are_not_collapsed_into_one(self):
        # The sidecar carries both, and they answer different questions: the bracket
        # ended on capacity, then Phase 2 bisected and CONVERGED. Forwarding
        # bracket_reason under the `stop_reason` key made the detail page report
        # "the search stopped: throughput plateau" for a search that finished
        # normally — and left the `converged` label it already ships unreachable.
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(
            benchmark,
            _BM93,
            best,
            self._facts("capacity_plateau", stop_reason="converged", stopped_at=256.0),
        )
        assert v["stop_reason"] == "converged"
        assert v["bracket_reason"] == "capacity_plateau"
        # The verdict still keys off the BRACKET reason — capacity is what bounded
        # the answer, whatever Phase 2 went on to do.
        assert [w["code"] for w in v["warnings"]] == ["slo_not_binding"]

    def test_a_legacy_sidecar_without_a_stop_reason_falls_back_to_the_bracket(self):
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        points = [_point(r, r * 1000.0, 0.5) for r in (4, 8, 16, 32)]
        v = analysis.compute_validity(
            benchmark,
            points,
            {"recommended_rate": 32},
            {"version": 1, "bracket_reason": "upper_bound"},
        )
        assert v["stop_reason"] == "upper_bound"

    def test_no_sidecar_leaves_no_stop_reason_key(self):
        v = analysis.compute_validity(
            SimpleNamespace(upper_bound=1024, max_points=12),
            [_point(4, 1000.0, 0.5), _point(8, 2000.0, 0.5), _point(16, 3000.0, 0.5)],
            {"recommended_rate": 16},
        )
        # Absent, not None: a stage / legacy / pre-sidecar run has no such fact, and
        # the UI must not render "stopped because: unknown".
        assert "stop_reason" not in v


class TestTheProbeIsNotAStage:
    """A row without a load value is a measurement, not a point on the curve.

    The auto-tune saturation probe (and, on legacy sweep records, the synchronous /
    throughput bound passes) carry no `rate`: their profile has none by
    construction. They are kept in the results table because they ARE measured
    data, but they must stay out of every aggregate — otherwise "N stages", the
    request total and the success rate all describe a population the analysis
    itself excludes, and the page and the API disagree about how many requests a
    run made (observed: 11 stages / 7,340 requests for a 10-stage / 7,290 run).

    Their numbers are also not comparable: a probe measured 1.71s against a 1.65s
    mean latency — one batch — so its prompt-token rate came back 3.4x the true
    value and its TTFT 6x the steady-state value at the same concurrency.
    """

    # A probe row as the worker appends it: no rate, throughput strategy, and — the
    # part that matters here — requests that must not reach the tally.
    PROBE = _point(None, 35641.0, 12.86, total=50, ok=50, strategy_type="throughput")

    def test_measured_stages_drops_rows_without_a_load(self):
        rows = [*_BM93, self.PROBE]
        kept = analysis.measured_stages(rows)
        assert len(kept) == len(_BM93)
        assert all(r.get("rate") is not None for r in kept)

    def test_the_probe_cannot_become_the_peak_or_the_recommendation(self):
        # Its throughput (35,641) beats every measured stage in _BM93 (max 35,731 —
        # close enough that a burst reading could plausibly win on another run).
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        with_probe = analysis.compute_best_points(benchmark, [*_BM93, self.PROBE])
        without = analysis.compute_best_points(benchmark, _BM93)
        assert with_probe == without
        assert with_probe["peak_rate"] == 256

    def test_the_probe_does_not_enter_the_validity_population(self):
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        with_probe = analysis.compute_validity(benchmark, [*_BM93, self.PROBE], best)
        without = analysis.compute_validity(benchmark, _BM93, best)
        assert with_probe == without

    def test_a_failing_probe_does_not_dent_the_success_rate(self):
        # The sharpest case: the probe is time-boxed now, so requests still in
        # flight when the window closes are recorded as incomplete. A probe row can
        # therefore read 290/802 on a perfectly healthy run — and that must not
        # show up as a 36%-success benchmark.
        bad_probe = _point(
            None, 35641.0, 12.86, total=802, ok=290, strategy_type="throughput"
        )
        rows = [*_BM93, bad_probe]
        measured = analysis.measured_stages(rows)
        total = sum(r["request_total"] for r in measured)
        ok = sum(r["request_successful"] for r in measured)
        assert total == sum(r["request_total"] for r in _BM93)
        assert ok == total  # every measured stage was clean

    def test_a_low_sample_probe_does_not_trigger_a_point_warning(self):
        # `point_high_error` fires on any measured stage under 95% success. Fed the
        # probe above it would flag a healthy run as overloaded.
        bad_probe = _point(
            None, 35641.0, 12.86, total=802, ok=290, strategy_type="throughput"
        )
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(benchmark, [*_BM93, bad_probe], best)
        assert "point_high_error" not in [w["code"] for w in v["warnings"]]


class TestProbeCapFactsAreCarriedThrough:
    """`probe_bound` / `probe_relaxed` reach the UI unchanged.

    They exist so the page can say what the probe's cap DID without recomputing
    `ceil(ceiling * 1.2)`, the Phase-1/2 split and the clamp rule — all of which
    live in benchmark-runner. Recomputing them here is how the two drift apart.
    """

    def _validity(self, **ramp):
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        return analysis.compute_validity(benchmark, _BM93, best, ramp or None)

    def test_the_facts_ride_along(self):
        v = self._validity(
            bracket_reason="capacity_plateau",
            stopped_at=36.0,
            probe_ceiling=29.22,
            probe_bound=36.0,
            probe_relaxed=0,
        )
        assert v["probe_ceiling"] == 29.22
        assert v["probe_bound"] == 36.0
        assert v["probe_relaxed"] == 0

    def test_a_run_without_a_probe_carries_no_probe_keys(self):
        # The concurrency axis never probes. Absent, not zero — "0 relaxes" would
        # read as "the cap held", which is a claim about a cap that never existed.
        v = self._validity(bracket_reason="capacity_plateau", stopped_at=256.0)
        assert "probe_ceiling" not in v
        assert "probe_bound" not in v
        assert "probe_relaxed" not in v


class TestSloBoundaryLocated:
    """`slo_met_rate` is an EDGE only when something above it was measured failing.

    Same number, two meanings: "257 breaks the SLO" vs ">= 256, we stopped looking".
    Reporting the second as the first invents a ceiling nobody measured — which is
    what capacity planning would then be done against.
    """

    def test_a_measured_breach_above_the_answer_is_an_edge(self):
        # 512 breaches the loose budget, so 256 is where it actually breaks.
        benchmark = SimpleNamespace(**{**_BM74_BENCHMARK, "slo_avg_ttft_ms": 200.0})
        best = analysis.compute_best_points(benchmark, _BM74)
        assert best["slo_met_rate"] == 256
        v = analysis.compute_validity(benchmark, _BM74, best)
        assert v["slo_boundary_located"] is True

    def test_stopping_before_anything_failed_is_a_floor(self):
        # bm102: the ramp stopped on the plateau at 256 with the budget 31% used.
        # Nothing above 256 was ever measured, so 256 is ">=", not the boundary.
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(benchmark, _BM93, best)
        assert v["slo_boundary_located"] is False

    def test_reaching_the_range_ceiling_is_not_a_located_boundary(self):
        # Everything passed all the way to upper_bound: the RANGE ran out, so the
        # edge is somewhere above it — unmeasured either way.
        points = [
            _point(r, r * 1000.0, 0.5, time_to_first_token_mean=50.0)
            for r in (4, 8, 16, 32)
        ]
        benchmark = SimpleNamespace(**{**_BM74_BENCHMARK, "upper_bound": 32})
        best = analysis.compute_best_points(benchmark, points)
        v = analysis.compute_validity(benchmark, points, best)
        assert v["slo_boundary_located"] is False

    def test_the_ramp_bracket_answers_it_directly(self):
        # With facts it is a lookup, not a scan: first_fail is the knob that
        # breached, and None is exactly "no boundary located".
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        best = analysis.compute_best_points(benchmark, _BM93)

        floor = analysis.compute_validity(
            benchmark, _BM93, best, {"slo_bracket": [256.0, None]}
        )
        assert floor["slo_boundary_located"] is False

        edge = analysis.compute_validity(
            benchmark, _BM93, best, {"slo_bracket": [256.0, 257.0]}
        )
        assert edge["slo_boundary_located"] is True

    def test_the_grid_fallback_uses_the_same_pass_rule_as_the_answer(self):
        # A point above the answer that fails only the SUCCESS floor (its latency is
        # fine) still counts as a breach — because that is the rule
        # compute_best_points used to exclude it from `met` in the first place.
        # Reading the two differently would report a boundary the answer denies.
        points = [
            _point(4, 4000.0, 0.5, time_to_first_token_mean=50.0),
            _point(8, 8000.0, 0.5, time_to_first_token_mean=50.0),
            _point(16, 9000.0, 0.5, time_to_first_token_mean=50.0, total=100, ok=50),
        ]
        benchmark = SimpleNamespace(**{**_BM74_BENCHMARK, "upper_bound": 1024})
        best = analysis.compute_best_points(benchmark, points)
        assert best["slo_met_rate"] == 8  # 16 excluded by the success floor
        v = analysis.compute_validity(benchmark, points, best)
        assert v["slo_boundary_located"] is True

    def test_a_run_without_an_slo_has_no_such_key(self):
        # Meaningless without thresholds — absent, not False.
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        best = analysis.compute_best_points(benchmark, _SWEEP)
        v = analysis.compute_validity(benchmark, _SWEEP, best)
        assert "slo_boundary_located" not in v


class TestReadRampFacts:
    """Reading the sidecar is best-effort: absence and corruption both mean
    "no facts", because every consumer already has to handle runs without one."""

    def test_reads_the_sidecar_for_an_auto_tune_run(self, tmp_path):
        (tmp_path / "7__ramp.json").write_text(
            '{"version": 1, "bracket_reason": "capacity_plateau", "stopped_at": 256}'
        )
        mgr = _bare_manager(tmp_path)
        facts = mgr._read_ramp_facts(SimpleNamespace(id=7, name="x", auto_tune=True))
        assert facts["bracket_reason"] == "capacity_plateau"

    def test_a_non_auto_tune_run_is_never_looked_up(self, tmp_path):
        # Stage / legacy runs have no ramp, so there is nothing to read even if a
        # stale file with the same id happened to sit there.
        (tmp_path / "7__ramp.json").write_text('{"bracket_reason": "upper_bound"}')
        mgr = _bare_manager(tmp_path)
        assert (
            mgr._read_ramp_facts(SimpleNamespace(id=7, name="x", auto_tune=False))
            is None
        )

    def test_a_missing_sidecar_is_not_an_error(self, tmp_path):
        mgr = _bare_manager(tmp_path)
        assert (
            mgr._read_ramp_facts(SimpleNamespace(id=9, name="x", auto_tune=True))
            is None
        )

    def test_corrupt_json_degrades_to_no_facts(self, tmp_path):
        # A half-written file during a partial sync must fall back to inference,
        # not take the analysis down.
        (tmp_path / "9__ramp.json").write_text('{"bracket_reason": "capacity_pl')
        mgr = _bare_manager(tmp_path)
        assert (
            mgr._read_ramp_facts(SimpleNamespace(id=9, name="x", auto_tune=True))
            is None
        )


class TestPeakAndRecommendation:
    """peak_rate is the true throughput argmax; recommended_rate is the peak, or
    (with an SLO) the SLO boundary capped at the peak — never a lower "knee". The
    plateau guard keeps a saturated top from being mislabeled range-limited."""

    def test_bm92_peak_is_the_argmax(self):
        # rate 31 has the highest throughput (0 errors) but was returned as
        # peak_rate=30. peak_rate and the no-SLO recommendation are both the argmax.
        bm92 = [
            _point(4, 4679.8, 0.14),
            _point(8, 9322.6, 0.18),
            _point(16, 18593.1, 0.20),
            _point(24, 27867.1, 0.24),
            _point(28, 32497.5, 0.33),
            _point(30, 34811.5, 0.35),
            _point(31, 35964.5, 0.40),  # true throughput peak
            _point(32, 35849.1, 1.16),
            _point(34, 31006.6, 16.27),
            _point(36, 30419.0, 28.99),
        ]
        benchmark = SimpleNamespace(
            load_type="fixed_rate", upper_bound=1024, max_points=12
        )
        out = analysis.compute_best_points(benchmark, bm92)
        assert out["peak_rate"] == 31
        assert out["recommended_rate"] == 31
        v = analysis.compute_validity(benchmark, bm92, out)
        assert v["warnings"] == []  # curve turned over cleanly; nothing to warn

    def test_bm93_real_slo_recommends_the_max_within_the_slo(self):
        benchmark = SimpleNamespace(**_BM93_BENCHMARK)
        out = analysis.compute_best_points(benchmark, _BM93)
        # 256 is the argmax AND meets the SLO => it is peak, slo_met, and the answer.
        assert out["peak_rate"] == 256
        assert out["slo_met_rate"] == 256
        assert out["recommended_rate"] == 256  # NOT a lower knee (was 128)
        v = analysis.compute_validity(benchmark, _BM93, out)
        codes = [w["code"] for w in v["warnings"]]
        # The recommendation is right, but the RUN has something to report: the
        # ramp stopped on the throughput plateau at 256 of a 4..1024 range, and the
        # 500ms budget was only 31% used there (155ms) — so 256 is a capacity
        # ceiling, not the latency boundary the profile set out to find. Re-observed
        # live as bm102, which reported "sufficient, no warnings" for a 7-point run.
        assert codes == ["slo_not_binding"]
        assert v["warnings"][0]["params"] == {"rate": 256, "used": 31}
        # Still not the reversed advice: raising the bound only buys plateau points.
        assert "not_saturated" not in codes

    def test_bm86_loose_slo_recommends_the_peak_without_a_reversed_warning(self):
        benchmark = SimpleNamespace(
            load_type="concurrency",
            upper_bound=1024,
            max_points=12,
            slo_avg_ttft_ms=10000.0,
            slo_avg_tpot_ms=1000.0,
        )
        out = analysis.compute_best_points(benchmark, _BM86)
        assert out["peak_rate"] == 512
        assert out["slo_met_rate"] == 512
        assert out["recommended_rate"] == 512  # meets the loose SLO, max throughput
        v = analysis.compute_validity(benchmark, _BM86, out)
        codes = [w["code"] for w in v["warnings"]]
        # Plateaued at the top => never "raise the bound". But a 10s TTFT budget
        # that the top point used 19% of (1917ms) did not shape this answer, and
        # saying so is the whole point of the code.
        assert codes == ["slo_not_binding"]
        assert v["warnings"][0]["params"]["used"] == 19
        assert "not_saturated" not in codes

    def test_an_slo_pressed_against_at_the_peak_stays_silent(self):
        # Same curve and range as bm93, but a budget the top point nearly spends
        # (155ms of 160ms = 97%): here the SLO genuinely bound the answer, so there
        # is nothing to report. This is the case _SLO_HEADROOM_RATIO protects — the
        # headroom test, not the rates, is what separates it from bm93.
        benchmark = SimpleNamespace(
            load_type="concurrency",
            upper_bound=1024,
            max_points=12,
            slo_avg_ttft_ms=160.0,
        )
        out = analysis.compute_best_points(benchmark, _BM93)
        assert out["slo_met_rate"] == 256
        assert out["recommended_rate"] == 256
        v = analysis.compute_validity(benchmark, _BM93, out)
        assert v["warnings"] == []
        assert v["sufficient"] is True

    def test_the_tightest_threshold_decides_the_headroom(self):
        # TTFT is loose (31%) but TPOT is nearly spent (1.21 of 1.3ms = 93%): the
        # run WAS shaped by a latency budget, just not by the one with slack. Taking
        # the loosest (or an average) would flag a binding SLO as irrelevant.
        benchmark = SimpleNamespace(
            load_type="concurrency",
            upper_bound=1024,
            max_points=12,
            slo_avg_ttft_ms=500.0,
            slo_avg_tpot_ms=1.3,
        )
        out = analysis.compute_best_points(benchmark, _BM93)
        v = analysis.compute_validity(benchmark, _BM93, out)
        assert v["warnings"] == []

    def test_no_slo_plateau_does_not_say_raise_the_bound(self):
        benchmark = SimpleNamespace(upper_bound=1024, max_points=12)
        out = analysis.compute_best_points(benchmark, _BM86)
        assert out["peak_rate"] == 512
        assert out["recommended_rate"] == 512
        v = analysis.compute_validity(benchmark, _BM86, out)
        assert [w["code"] for w in v["warnings"]] == []


class TestArtifactNaming:
    """The result-file naming is an interface with benchmark-runner, and a mismatch
    does not raise — the collection pass just skips what it cannot find, so the
    symptom would be a curve that silently loses points."""

    def test_a_point_file_is_recognized_and_its_index_parsed(self):
        assert artifacts.is_point_file("7__p11.json", 7)
        assert artifacts.point_file_index("7__p11.json") == 11

    def test_the_full_companion_is_not_a_point(self):
        # Every dual_json output writes a pair; only the trimmed summary is read.
        assert not artifacts.is_point_file("7__p0.full.json", 7)

    def test_the_saturation_probe_is_not_a_point(self):
        # Shares the id prefix but not the `__p` marker.
        assert not artifacts.is_point_file("7__satprobe.json", 7)

    def test_a_sibling_benchmark_is_not_counted(self):
        # The id is part of the prefix, so 71's files never look like 7's.
        assert not artifacts.is_point_file("71__p0.json", 7)
        assert not artifacts.is_point_file("7__p0.json", 71)

    def test_points_are_listed_in_probe_order_not_lexicographic(self, tmp_path):
        # The ramp doubles then bisects, so the index is the only record of how the
        # curve was walked — and "10" sorts before "2" as a string.
        for name in ("5__p0.json", "5__p2.json", "5__p10.json", "5__p1.json"):
            (tmp_path / name).write_text("{}")
        assert artifacts.list_point_files(str(tmp_path), 5) == [
            "5__p0.json",
            "5__p1.json",
            "5__p2.json",
            "5__p10.json",
        ]

    def test_a_missing_directory_is_not_an_error(self):
        # "No points ready yet" is a normal state, not a failure.
        assert artifacts.list_point_files("/nonexistent-dir", 1) == []


class TestFinalizePartialAnalysis:
    """A run that ends any way OTHER than completion still has to drop the
    `in_progress` tag: the row will never change again, so a snapshot labelled
    provisional would hide the coverage banner (and show Coverage "-") forever while
    the peak/recommended cards from that same snapshot stayed on screen."""

    def _mgr(self, monkeypatch, captured, *, results=None, synced=True, posted=None):
        monkeypatch.setattr(
            analysis, "compute_best_points", lambda _b, _r: {"peak_rate": 7}
        )
        monkeypatch.setattr(
            analysis,
            "compute_validity",
            lambda _b, _r, _bp, _ramp=None: {"sufficient": True, "warnings": []},
        )
        mgr = _bare_manager()
        mgr._read_ramp_facts = lambda _b: None
        mgr._update_benchmark_state_sync = lambda _id, **k: captured.update(k)
        mgr._collect_results = lambda _b: CollectedResults(
            results=results if results is not None else [{"rate": 1}],
            metrics=SimpleNamespace(),
            report=None,
            loaded=1,
            skipped=0,
        )
        mgr._post_metrics_and_results = lambda _b, _m, r, **k: (
            posted if posted is not None else []
        ).append(r)
        if synced:
            mgr._partial_synced_count[9] = 1
        return mgr

    def _benchmark(self):
        return SimpleNamespace(id=9, name="x", auto_tune=True, stages=None)

    def test_the_flag_is_dropped_and_the_points_stay(self, monkeypatch):
        captured = {}
        mgr = self._mgr(monkeypatch, captured)
        mgr._finalize_partial_analysis(self._benchmark())
        assert "in_progress" not in captured["validity"]
        assert captured["peak_rate"] == 7  # the measured points are still published

    def test_the_grid_is_republished_alongside_the_conclusion(self, monkeypatch):
        # A point that finished after the last partial sync feeds the analysis, so
        # it has to reach the results table too — otherwise the peak / recommended
        # cards name a load the stage table cannot show.
        posted = []
        mgr = self._mgr(
            monkeypatch, {}, results=[{"rate": 1}, {"rate": 2}], posted=posted
        )
        mgr._finalize_partial_analysis(self._benchmark())
        assert posted == [[{"rate": 1}, {"rate": 2}]]

    def test_a_failed_grid_upload_still_drops_the_provisional_flag(self, monkeypatch):
        # Ordering guard: the flag is what this method exists to clear, so an upload
        # blip must not cost us that (it only returns us to the pre-fix behaviour).
        captured = {}
        mgr = self._mgr(monkeypatch, captured)

        def boom(_b, _m, _r, **_k):
            raise RuntimeError("api blip")

        mgr._post_metrics_and_results = boom
        mgr._finalize_partial_analysis(self._benchmark())
        assert "in_progress" not in captured["validity"]
        assert captured["peak_rate"] == 7

    def test_points_are_published_even_if_no_partial_sync_ever_landed(
        self, monkeypatch
    ):
        # `_partial_synced_count` records SUCCESSFUL partial syncs, and those run
        # with attempts=1. A run that measured its points against a briefly flaky
        # API therefore has an empty counter and full point files on disk — keying
        # off the counter threw those measurements away.
        captured = {}
        posted = []
        mgr = self._mgr(
            monkeypatch,
            captured,
            results=[{"rate": 1}, {"rate": 2}],
            posted=posted,
            synced=False,
        )
        mgr._finalize_partial_analysis(self._benchmark())
        assert captured["peak_rate"] == 7
        assert "in_progress" not in captured["validity"]
        assert posted == [[{"rate": 1}, {"rate": 2}]]

    def test_a_single_point_run_is_still_left_alone(self, monkeypatch):
        # The one gate that stays: a single-point run never partial-syncs by design,
        # so there is no provisional flag, and its terminal path is the completion
        # handler's full sync.
        captured = {}
        mgr = self._mgr(monkeypatch, captured, synced=False)
        mgr._finalize_partial_analysis(
            SimpleNamespace(id=9, name="x", auto_tune=False, stages=None)
        )
        assert captured == {}

    def test_a_single_run_is_never_touched(self, monkeypatch):
        captured = {}
        mgr = self._mgr(monkeypatch, captured)
        mgr._partial_synced_count[9] = 1
        single = SimpleNamespace(id=9, name="x", auto_tune=False, stages=None)
        mgr._finalize_partial_analysis(single)
        assert captured == {}

    def test_nothing_collected_means_nothing_written(self, monkeypatch):
        captured = {}
        mgr = self._mgr(monkeypatch, captured, results=[])
        mgr._finalize_partial_analysis(self._benchmark())
        assert captured == {}

    def test_it_never_raises_so_teardown_still_runs(self, monkeypatch):
        mgr = _bare_manager()
        mgr._partial_synced_count[9] = 1

        def boom(_b):
            raise RuntimeError("disk gone")

        mgr._collect_results = boom
        mgr._finalize_partial_analysis(self._benchmark())  # must not propagate

    @pytest.mark.parametrize(
        "handler, patch_state",
        [
            ("_handle_benchmark_failure", bm.BenchmarkStateEnum.ERROR),
            ("_handle_benchmark_timeout", bm.BenchmarkStateEnum.ERROR),
        ],
    )
    def test_each_terminal_handler_finalizes_before_teardown(
        self, handler, patch_state
    ):
        # Ordering matters: _stop_benchmark drops the partial-sync bookkeeping that
        # _finalize_partial_analysis keys off, so it has to run first.
        order = []
        mgr = object.__new__(BenchmarkManager)
        mgr._update_benchmark_state_sync = lambda _id, **k: order.append(
            f"state:{k.get('state')}"
        )
        mgr._finalize_partial_analysis = lambda _b: order.append("finalize")
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._stop_benchmark = lambda _b: order.append("teardown")

        getattr(mgr, handler)(SimpleNamespace(id=9, name="x"))

        assert order == [f"state:{patch_state}", "finalize", "teardown"]

    def test_a_user_stop_also_finalizes_before_teardown(self):
        import asyncio

        order = []
        mgr = object.__new__(BenchmarkManager)
        mgr._finalize_partial_analysis = lambda _b: order.append("finalize")
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._stop_benchmark = lambda _b: order.append("teardown")
        mgr._clear_active_benchmark = lambda _i: None

        asyncio.run(mgr._handle_stop_benchmark_event(SimpleNamespace(id=9, name="x")))

        assert order == ["finalize", "teardown"]


class TestTerminalSyncReportsLostPoints:
    """An unreadable point file means something different once the run is over: in a
    partial sync the next tick retries it, but here the point is gone for good and
    the curve the user reads is missing it."""

    def _mgr(self, monkeypatch, collected):
        monkeypatch.setattr(
            analysis, "compute_best_points", lambda _b, _r: {"peak_rate": 1}
        )
        monkeypatch.setattr(
            analysis,
            "compute_validity",
            lambda _b, _r, _bp, _ramp=None: {"sufficient": True, "warnings": []},
        )
        mgr = _bare_manager()
        mgr._read_ramp_facts = lambda _b: None
        mgr._collect_results = lambda _b: collected
        mgr._load_request_samples = lambda _r, limit=None: ([], [])
        mgr._log_request_failures_if_any = lambda **_k: None
        mgr._build_partial_failure_state_message = lambda **_k: None
        mgr._post_metrics_and_results = lambda *a, **k: None
        mgr._truncate_state_message = lambda m: m
        mgr._retry_sync = lambda fn, what, attempts=3: fn()
        return mgr

    def _collected(self, skipped):
        return CollectedResults(
            results=[{"rate": 1, "request_total": 10, "request_successful": 10}],
            metrics=SimpleNamespace(
                request_total=10,
                request_successful=10,
                request_errored=0,
                request_incomplete=0,
                model_dump=lambda: {},
            ),
            report=None,
            loaded=2,
            skipped=skipped,
        )

    def test_lost_points_are_surfaced_on_the_row(self, monkeypatch):
        messages = []
        mgr = self._mgr(monkeypatch, self._collected(skipped=1))
        mgr._update_benchmark_state_sync = lambda _id, **k: messages.append(k)

        mgr._sync_benchmark_metrics(SimpleNamespace(id=9, name="x", auto_tune=True))

        said = [m.get("state_message", "") for m in messages]
        assert any("1 of 3 measured point(s) could not be read" in m for m in said)

    def test_a_clean_read_says_nothing(self, monkeypatch):
        messages = []
        mgr = self._mgr(monkeypatch, self._collected(skipped=0))
        mgr._update_benchmark_state_sync = lambda _id, **k: messages.append(k)

        mgr._sync_benchmark_metrics(SimpleNamespace(id=9, name="x", auto_tune=True))

        assert not any(
            "could not be read" in m.get("state_message", "") for m in messages
        )


class TestAllSloThresholdsReachTheRunner:
    """SLO_THRESHOLDS is the single source of truth. The runner used to keep its own
    hand-written copy of the nine flags, so a threshold added to the model but not to
    that list was accepted by the API and silently never forwarded."""

    def test_every_threshold_has_a_flag_the_runner_forwards(self):
        runner = TestBuildCommandArgs()._runner(
            auto_tune=True,
            load_type="concurrency",
            **{t.attr: float(i + 1) for i, t in enumerate(bm_schemas.SLO_THRESHOLDS)},
        )
        args = BenchmarkRunner._build_command_args(runner)
        for i, t in enumerate(bm_schemas.SLO_THRESHOLDS):
            assert t.flag in args, f"{t.attr} was not forwarded"
            assert args[args.index(t.flag) + 1] == str(float(i + 1))

    def test_the_flags_are_distinct(self):
        flags = [t.flag for t in bm_schemas.SLO_THRESHOLDS]
        assert len(set(flags)) == len(flags)

    def test_each_row_points_at_a_real_column_and_a_real_metric(self):
        from gpustack.schemas.benchmark import BenchmarkBase, BenchmarkMetricsLite

        for t in bm_schemas.SLO_THRESHOLDS:
            assert t.attr in BenchmarkBase.model_fields
            assert t.metric in BenchmarkMetricsLite.model_fields


class TestResultsUploadIsTreatedLikeMetrics:
    """The per-point grid used to be posted once, with any failure logged and
    swallowed. That is the more deceptive of the two losses: the parent row still
    carries a representative point so the list page looks fine, while the detail
    page's curve is simply empty and nothing says why."""

    def _mgr(self, monkeypatch, fail_on, fail_times=99):
        monkeypatch.setattr(bm.time, "sleep", lambda _s: None)  # no backoff wait
        monkeypatch.setattr(bm, "raise_if_response_error", lambda _r: None)
        calls = {"metrics": 0, "results": 0}

        class _Httpx:
            def post(self, url, json=None):
                key = "metrics" if url.endswith("/metrics") else "results"
                calls[key] += 1
                if key == fail_on and calls[key] <= fail_times:
                    raise RuntimeError("blip")
                return object()

        mgr = object.__new__(BenchmarkManager)
        mgr._clientset_getter = lambda: SimpleNamespace(
            http_client=SimpleNamespace(get_httpx_client=lambda: _Httpx())
        )
        return mgr, calls

    def _post(self, mgr, **kw):
        mgr._post_metrics_and_results(
            SimpleNamespace(id=7, name="x"),
            SimpleNamespace(model_dump=lambda: {}),
            [{"rate": 1}],
            **kw,
        )

    def test_a_transient_results_failure_is_retried(self, monkeypatch):
        mgr, calls = self._mgr(monkeypatch, fail_on="results", fail_times=2)
        self._post(mgr)
        assert calls["results"] == 3  # retried twice, then succeeded

    def test_a_persistent_results_failure_propagates(self, monkeypatch):
        # It has to reach _handle_benchmark_completion, which turns it into a
        # state_message; swallowing it left "completed" with an empty curve.
        mgr, calls = self._mgr(monkeypatch, fail_on="results")
        with pytest.raises(RuntimeError):
            self._post(mgr)
        assert calls["results"] == 3

    def test_a_partial_sync_gets_a_single_attempt(self, monkeypatch):
        # attempts=1: the next poll retries, so a blip must not block this one.
        mgr, calls = self._mgr(monkeypatch, fail_on="results")
        with pytest.raises(RuntimeError):
            self._post(mgr, attempts=1)
        assert calls["results"] == 1

    def test_the_grid_is_not_posted_when_the_metrics_post_never_lands(
        self, monkeypatch
    ):
        mgr, calls = self._mgr(monkeypatch, fail_on="metrics")
        with pytest.raises(RuntimeError):
            self._post(mgr)
        assert calls["metrics"] == 3
        assert calls["results"] == 0


class TestUnreadableResultsAreReported:
    """A terminal sync that finds no usable metrics used to return silently, so the
    row flipped to COMPLETED with every metric null and nothing saying why."""

    def _mgr(self, collected):
        mgr = _bare_manager()
        mgr._collect_results = lambda _b: collected
        return mgr

    def test_no_result_file_at_all_is_reported(self):
        mgr = self._mgr(CollectedResults([], None, None, 0, 0))
        with pytest.raises(RuntimeError, match="no result file"):
            mgr._sync_benchmark_metrics(SimpleNamespace(id=1, name="x", auto_tune=True))

    def test_files_that_were_all_unreadable_say_so_instead(self):
        # Different cause, different advice: the run DID produce files, they just
        # could not be parsed — so the message must not claim none were written.
        mgr = self._mgr(CollectedResults([], None, None, 0, 3))
        with pytest.raises(RuntimeError, match="3 result file"):
            mgr._sync_benchmark_metrics(SimpleNamespace(id=1, name="x", auto_tune=True))

    def test_the_completion_handler_turns_it_into_a_state_message(self):
        # And the wording is about the SYNC, not an upload: nothing was uploaded.
        messages = []
        mgr = object.__new__(BenchmarkManager)
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._truncate_state_message = lambda m: m
        mgr._stop_benchmark = lambda _b: None
        mgr._update_benchmark_state_sync = lambda _id, **k: messages.append(k)

        def boom(_b):
            raise RuntimeError("the run produced no result file")

        mgr._sync_benchmark_metrics = boom
        mgr._handle_benchmark_completion(SimpleNamespace(id=7, name="x"))

        assert messages == [
            {
                "state": bm.BenchmarkStateEnum.COMPLETED,
                "state_message": (
                    "Result sync failed: the run produced no result file. "
                    "See worker logs for details."
                ),
            }
        ]


class TestTpotSloIsDecodeOnly:
    """A `slo_*_tpot_ms` threshold bounds the DECODE-ONLY per-token time.

    guidellm reports two per-output-token latencies under names that are the
    reverse of the industry's: `inter_token_latency_ms` is
    (last_token - first_token) / (tokens - 1) — what vLLM and genai-perf call TPOT
    — while `time_per_output_token_ms` starts at request_start and folds TTFT into
    the average. The thresholds used to be judged on the second one, which billed
    prefill and queue wait to the decode loop: the error is TTFT / (n * TPOT), so
    ~5% at 128 output tokens and ~40% at 16, and it grew with load exactly where
    the SLO decides capacity.
    """

    def test_the_thresholds_read_the_decode_only_columns(self):
        by_attr = {t.attr: t.metric for t in bm_schemas.SLO_THRESHOLDS}
        assert by_attr["slo_avg_tpot_ms"] == "inter_token_latency_mean"
        assert by_attr["slo_p95_tpot_ms"] == "inter_token_latency_p95"
        assert by_attr["slo_p99_tpot_ms"] == "inter_token_latency_p99"

    def test_the_includes_ttft_metric_is_only_the_fallback(self):
        metrics = {t.metric for t in bm_schemas.SLO_THRESHOLDS}
        assert "time_per_output_token_mean" not in metrics
        assert "time_per_output_token_p95" not in metrics
        assert "time_per_output_token_p99" not in metrics
        by_attr = {t.attr: t.fallback for t in bm_schemas.SLO_THRESHOLDS}
        assert by_attr["slo_avg_tpot_ms"] == "time_per_output_token_mean"
        assert by_attr["slo_p95_tpot_ms"] == "time_per_output_token_p95"
        assert by_attr["slo_p99_tpot_ms"] == "time_per_output_token_p99"

    def test_only_the_tpot_rows_have_a_fallback(self):
        # TTFT and end-to-end latency are always measurable; giving them a second
        # column to try would only hide a missing percentile.
        for t in bm_schemas.SLO_THRESHOLDS:
            if "tpot" not in t.attr:
                assert t.fallback is None, t.attr

    def test_every_fallback_points_at_a_real_column(self):
        from gpustack.schemas.benchmark import BenchmarkMetricsLite

        for t in bm_schemas.SLO_THRESHOLDS:
            if t.fallback:
                assert t.fallback in BenchmarkMetricsLite.model_fields

    def test_a_non_incremental_response_falls_back_to_the_other_basis(self):
        # A server answering in ONE chunk (whole output at once, common at low load)
        # leaves the first and last token iteration sharing a timestamp, so guidellm
        # reports the decode-only metric as 0. Nothing about the gap between tokens
        # is observable there and total-time-over-tokens is the only per-token
        # number left, so the threshold is judged on it. The alternatives are both
        # wrong: 0 ms clears every budget, and failing outright would bracket the
        # ramp on its first point for any server that batches its stream.
        benchmark = SimpleNamespace(slo_avg_tpot_ms=5.0)
        one_chunk = _point(4, 4600, 0.0, time_per_output_token_mean=4.7)
        assert analysis.meets_slo(benchmark, one_chunk)
        assert analysis.slo_utilization(benchmark, one_chunk) == pytest.approx(0.94)
        assert not analysis.meets_slo(SimpleNamespace(slo_avg_tpot_ms=4.0), one_chunk)

    def test_neither_basis_measured_fails_instead_of_waiving_the_threshold(self):
        # Also the shape of an old row written before either column existed:
        # "we did not measure it" is not evidence that the threshold held.
        benchmark = SimpleNamespace(slo_avg_tpot_ms=5.0)
        assert not analysis.meets_slo(benchmark, _point(4, 4600, 0.0))
        # ... and it must not report 0% of the budget used either, which is what
        # decides whether the SLO is described as binding.
        assert analysis.slo_utilization(benchmark, _point(4, 4600, 0.0)) is None

    def test_a_measured_tpot_still_passes_and_reports_its_utilization(self):
        benchmark = SimpleNamespace(slo_avg_tpot_ms=5.0)
        assert analysis.meets_slo(benchmark, _point(4, 4600, 2.5))
        assert analysis.slo_utilization(benchmark, _point(4, 4600, 2.5)) == 0.5


class TestOneMissingPercentileDoesNotCostThePoint:
    """The report models double as the storage filter, so what they declare decides
    what survives — but what they REQUIRE decides what is thrown away. A required
    percentile means `model_validate` raises, `_aggregate_points` skips the file it
    came from, and a twelve-point curve silently becomes eleven (visible only as
    "1 of 12 measured point(s) could not be read" on state_message)."""

    def test_a_percentile_left_out_validates_as_absent(self):
        from gpustack.worker.schemas.benchmark_runner import Percentiles

        # p50 present, the tail missing: no exception, and the gaps read as unknown.
        p = Percentiles.model_validate({"p50": 12.0})
        assert p.p50 == 12.0
        assert (p.p25, p.p75, p.p90, p.p95, p.p99) == (None, None, None, None, None)

    def test_absence_is_not_zero(self):
        # 0 would be indistinguishable from a real measurement of 0 ms, and it is
        # what a SLO verdict or an IQR band would then be computed from.
        from gpustack.worker.schemas.benchmark_runner import Percentiles

        dumped = Percentiles.model_validate({"p50": 12.0}).model_dump()
        assert dumped["p99"] is None

    def test_every_percentile_is_optional(self):
        # Pinned as a set so the next percentile added has to make the same choice
        # deliberately rather than by copying the line above it.
        from gpustack.worker.schemas.benchmark_runner import Percentiles

        assert not [
            name for name, f in Percentiles.model_fields.items() if f.is_required()
        ]


class TestTheProbesCapIsNotTheUsersRange:
    """`not_saturated` says "raise upper_bound". Never say it about the soft cap.

    The saturation probe derives its own cap from a fresh measurement on every
    run, so raising the range cannot move it — the advice is not merely unhelpful,
    it is unfollowable. Reproduces benchmark 40: configured 4..1024, probe read
    25.6 rps, cap = ceil(25.6 * 1.2) = 31, ramp stopped there with the grid still
    climbing (+63% throughput on the last step) and the page told the user to
    raise the 1024.
    """

    BM40 = {
        "version": 1,
        "bracket_reason": "upper_bound",
        "stop_reason": "upper_bound",
        "stopped_at": 31,
        "probe_ceiling": 25.6,
        "probe_bound": 31,
        "probe_relaxed": 0,
    }

    @staticmethod
    def _points():
        return [
            _point(4, 4635.0, 0.4),
            _point(8, 9220.0, 0.4),
            _point(16, 18350.0, 0.5),
            _point(31, 30022.0, 1.4),
        ]

    def _validity(self, ramp, upper_bound=1024):
        return analysis.compute_validity(
            SimpleNamespace(upper_bound=upper_bound, max_points=12),
            self._points(),
            {"recommended_rate": 31, "peak_rate": 31},
            ramp,
        )

    def test_a_legacy_sidecar_that_calls_the_cap_upper_bound_is_seen_through(self):
        # An older runner reports both bounds as `upper_bound`; the facts it does
        # carry still tell them apart — it ended ON the cap, and the cap was below
        # the range the user asked for.
        assert self._validity(self.BM40)["warnings"] == []

    def test_the_named_reason_needs_no_inference(self):
        ramp = {**self.BM40, "bracket_reason": "probe_bound"}
        assert self._validity(ramp)["warnings"] == []

    def test_the_users_own_range_still_gets_the_advice(self):
        # Same shape, but the run ended at the top of the range it was given:
        # raising it is exactly the right thing to do.
        ramp = {**self.BM40, "stopped_at": 1024, "probe_bound": 1024}
        codes = [w["code"] for w in self._validity(ramp)["warnings"]]
        assert codes == ["not_saturated"]

    def test_a_cap_that_never_bound_anything_still_gets_the_advice(self):
        # The probe read HIGH, so the ramp hit the user's range before the cap:
        # stopped_at < probe_bound means the cap is not what ended this.
        ramp = {**self.BM40, "stopped_at": 512, "probe_bound": 900}
        codes = [w["code"] for w in self._validity(ramp, upper_bound=512)["warnings"]]
        assert codes == ["not_saturated"]

    def test_a_run_with_no_probe_is_unaffected(self):
        # Concurrency axis: no probe, so no cap keys — the verdict must not change.
        ramp = {
            "version": 1,
            "bracket_reason": "upper_bound",
            "stop_reason": "upper_bound",
            "stopped_at": 1024,
        }
        codes = [w["code"] for w in self._validity(ramp)["warnings"]]
        assert codes == ["not_saturated"]


class TestAnUnreadablePointSaysWhy:
    """ "2 result file(s) were unreadable" is not a diagnosis.

    A truncated write, a schema the installed guidellm does not know, and a run
    that died mid-point all reduce to that one sentence, and they are fixed
    differently. The message referred the reader to the worker log, which is the
    hardest artifact to reach from a browser — the log page next to it shows the
    CONTAINER's output, not the agent's. So the first file's actual error
    travels with the count.
    """

    def _collected(self, tmp_path, contents: dict):
        for name, body in contents.items():
            (tmp_path / name).write_text(body)
        mgr = _bare_manager(tmp_path)
        benchmark = SimpleNamespace(
            id=7, name="bm", dataset_input_tokens=128, stages=None
        )
        return mgr._aggregate_points(
            benchmark, [str(tmp_path / name) for name in contents]
        )

    def test_the_first_failure_is_carried(self, tmp_path):
        collected = self._collected(
            tmp_path, {"7__p0.json": "", "7__p1.json": "{not json"}
        )
        assert collected.skipped == 2
        assert collected.loaded == 0
        # Named by file, because which point died is half the answer.
        assert "7__p0.json" in collected.skipped_reason

    def test_the_first_one_wins_not_the_last(self, tmp_path):
        # Files are read in point order, so the earliest failure is the one that
        # explains the run; a later point usually fails as a consequence.
        collected = self._collected(
            tmp_path, {"7__p0.json": "", "7__p1.json": "{not json"}
        )
        assert "7__p1.json" not in collected.skipped_reason

    def test_a_clean_read_carries_no_reason(self, tmp_path):
        mgr = _bare_manager(tmp_path)
        benchmark = SimpleNamespace(
            id=7, name="bm", dataset_input_tokens=128, stages=None
        )
        collected = mgr._aggregate_points(benchmark, [])
        assert collected.skipped == 0
        assert collected.skipped_reason is None


class TestTeardownIsIdempotent:
    """The watch replays terminal rows on every reconnect. Teardown is not
    repeatable — the row and the workload it acts on are already gone — so an
    unguarded replay would spin forever on finalize (404) -> fetch logs (no
    workload) -> delete workload (no workload), starving every queued run
    behind it."""

    def _mgr(self, calls):
        mgr = object.__new__(BenchmarkManager)
        mgr._worker_id_getter = lambda: 7
        mgr._canceled_ids = set()
        mgr._torn_down_ids = bm.OrderedDict()
        mgr._provisioning_processes = {}
        mgr._benchmark_by_id = {}
        mgr._container_log_offset = {}
        mgr._last_log_snapshot_at = {}
        mgr._partial_synced_count = {}
        mgr._last_partial_sync_at = {}
        mgr._last_target_check_at = {}
        mgr._target_missing_since = {}
        mgr._active_benchmark_id = None
        mgr._active_benchmark_started_at = None
        mgr._is_provisioning = lambda _b: False
        mgr._delete_workload_calls = calls
        return mgr

    def _event(self, etype, state):
        return SimpleNamespace(
            type=etype,
            data=SimpleNamespace(
                id=27, name="b", worker_id=7, state=state, namespace=None
            ),
        )

    def _patch_common(self, monkeypatch, mgr, calls):
        monkeypatch.setattr(
            bm.Benchmark, "model_validate", staticmethod(lambda d: d), raising=False
        )
        monkeypatch.setattr(
            bm, "delete_workload", lambda *a, **k: calls.append("delete")
        )

    def test_replayed_delete_tears_down_once(self, monkeypatch):
        calls = []
        mgr = self._mgr(calls)
        self._patch_common(monkeypatch, mgr, calls)

        event = self._event(bm.EventType.DELETED, bm_schemas.BenchmarkStateEnum.RUNNING)
        for _ in range(5):
            mgr._handle_benchmark_event(event)

        assert calls == ["delete"]
        assert mgr._is_torn_down(27)

    def test_replayed_stop_schedules_one_teardown(self, monkeypatch):
        calls = []
        mgr = self._mgr(calls)
        self._patch_common(monkeypatch, mgr, calls)
        scheduled = []
        monkeypatch.setattr(
            bm.asyncio, "create_task", lambda coro: (coro.close(), scheduled.append(1))
        )

        event = self._event(bm.EventType.UPDATED, bm_schemas.BenchmarkStateEnum.STOPPED)
        mgr._handle_benchmark_event(event)
        # The async handler is what calls _stop_benchmark; stand in for it.
        mgr._stop_benchmark(event.data)
        for _ in range(5):
            mgr._handle_benchmark_event(event)

        assert len(scheduled) == 1
        assert calls == ["delete"]

    def test_re_enqueue_rearms_the_guard(self, monkeypatch):
        import asyncio

        calls = []
        mgr = self._mgr(calls)
        self._patch_common(monkeypatch, mgr, calls)
        mgr._benchmark_queue = deque()
        mgr._queue_lock = asyncio.Lock()

        async def _noop_state(_id, **_kw):
            # True = "the patch landed"; False means the server says the row is
            # gone, which would (correctly) make the enqueue drop it.
            return True

        mgr._update_benchmark_state = _noop_state

        deleted = self._event(
            bm.EventType.DELETED, bm_schemas.BenchmarkStateEnum.RUNNING
        )
        mgr._handle_benchmark_event(deleted)
        assert mgr._is_torn_down(27)

        asyncio.run(mgr._enqueue_benchmark(deleted.data))
        assert not mgr._is_torn_down(27)
        assert 27 not in mgr._canceled_ids

        mgr._handle_benchmark_event(deleted)
        assert calls == ["delete", "delete"]

    def test_memo_is_bounded(self):
        mgr = self._mgr([])
        for i in range(bm._TEARDOWN_MEMO_LIMIT + 50):
            mgr._record_teardown(i)

        assert len(mgr._torn_down_ids) == bm._TEARDOWN_MEMO_LIMIT
        assert not mgr._is_torn_down(0)
        assert mgr._is_torn_down(bm._TEARDOWN_MEMO_LIMIT + 49)


class TestVanishedWorkloadIsNotAutomaticallyAFailure:
    """The runner container can be reaped between the run returning and the
    3-second state poll. Every point is already measured and written by then, so
    calling that ERROR puts a red badge on a complete curve."""

    # This worker owns every row these tests feed in. Stated explicitly because
    # `_sync_single_benchmark_state` now refuses rows owned by someone else
    # (D17) -- without an owner the cases below would exit on that guard and
    # pass for the wrong reason, never reaching the workload logic they exist
    # to check.
    OWNER = 7

    def _mgr(self, tmp_path, calls):
        mgr = object.__new__(BenchmarkManager)
        mgr._benchmark_dir = str(tmp_path)
        mgr._torn_down_ids = bm.OrderedDict()
        mgr._worker_id_getter = lambda: self.OWNER
        mgr._is_benchmark_timed_out = lambda _b: False
        mgr._is_provisioning = lambda _b: False
        mgr._handle_benchmark_completion = lambda b: calls.append(("complete", b.id))
        mgr._handle_benchmark_failure = lambda b: calls.append(("fail", b.id))
        mgr._maybe_snapshot_logs = lambda _b: None
        mgr._maybe_sync_partial_metrics = lambda _b: None
        return mgr

    def _run(self, monkeypatch, mgr, workload):
        monkeypatch.setattr(bm, "get_workload", lambda *a, **k: workload)
        mgr._sync_single_benchmark_state(
            SimpleNamespace(id=35, name="b", namespace=None, worker_id=self.OWNER)
        )

    def test_gone_after_writing_the_curve_counts_as_finished(
        self, tmp_path, monkeypatch
    ):
        calls = []
        mgr = self._mgr(tmp_path, calls)
        (tmp_path / "35__curve.json").write_text("{}")

        self._run(monkeypatch, mgr, None)

        assert calls == [("complete", 35)]

    def test_gone_after_writing_the_ramp_sidecar_counts_as_finished(
        self, tmp_path, monkeypatch
    ):
        calls = []
        mgr = self._mgr(tmp_path, calls)
        (tmp_path / "35__ramp.json").write_text("{}")

        self._run(monkeypatch, mgr, None)

        assert calls == [("complete", 35)]

    def test_gone_with_no_terminal_artifact_is_still_a_failure(
        self, tmp_path, monkeypatch
    ):
        calls = []
        mgr = self._mgr(tmp_path, calls)

        self._run(monkeypatch, mgr, None)

        assert calls == [("fail", 35)]

    def test_an_unhealthy_workload_is_a_failure_even_with_the_artifact(
        self, tmp_path, monkeypatch
    ):
        calls = []
        mgr = self._mgr(tmp_path, calls)
        (tmp_path / "35__curve.json").write_text("{}")
        workload = SimpleNamespace(state=bm.WorkloadStatusStateEnum.UNHEALTHY)

        self._run(monkeypatch, mgr, workload)

        assert calls == [("fail", 35)]

    def test_a_torn_down_row_is_not_handled_again(self, tmp_path, monkeypatch):
        calls = []
        mgr = self._mgr(tmp_path, calls)
        (tmp_path / "35__curve.json").write_text("{}")
        mgr._record_teardown(35)

        self._run(monkeypatch, mgr, None)

        assert calls == []


# ── Measured per-interval ITL ─────────────────────────────────────────────────


def _dist(*, mean=1.0, p95=2.0, p99=3.0, maximum=4.0, count=10):
    """A StatusDistributionSummary whose `successful` branch carries the values."""
    from gpustack.worker.schemas.benchmark_runner import (
        DistributionSummary,
        Percentiles,
        StatusDistributionSummary,
    )

    summary = DistributionSummary(
        mean=mean,
        median=mean,
        min=0.0,
        max=maximum,
        count=count,
        percentiles=Percentiles(p95=p95, p99=p99),
    )
    return StatusDistributionSummary(
        successful=summary, incomplete=summary, errored=summary, total=summary
    )


def _metrics(**overrides):
    """A GenerativeMetrics with every required distribution filled in."""
    from gpustack.worker.schemas.benchmark_runner import (
        GenerativeMetrics,
        StatusBreakdown,
    )

    fields = {
        name: _dist()
        for name, field in GenerativeMetrics.model_fields.items()
        if field.is_required() and name != "request_totals"
    }
    fields["request_totals"] = StatusBreakdown[int, int, int, int](
        successful=10, errored=0, incomplete=0, total=10
    )
    fields.update(overrides)
    return GenerativeMetrics(**fields)


def _itl_kwargs(metrics):
    """The flat row kwargs for a point with these metrics.

    ``_point_metrics_kwargs`` only reads ``bm.metrics``, so a namespace stands
    in for the whole GenerativeBenchmark.
    """
    from gpustack.worker.schemas.benchmark_runner import GenerativeBenchmarksReport

    return GenerativeBenchmarksReport._point_metrics_kwargs(
        SimpleNamespace(metrics=metrics)
    )


class TestMeasuredItlMapping:
    """The per-interval ITL is a different metric from the per-request TPOT.

    Both must reach the row, under names that cannot be mistaken for each other,
    and "not measured" must not arrive as 0.0.
    """

    def test_measured_gaps_land_on_the_itl_columns(self):
        metrics = _metrics(
            inter_token_latency_per_chunk_ms=_dist(
                mean=26.4, p95=27.0, p99=800.0, maximum=900.0
            )
        )

        kwargs = _itl_kwargs(metrics)

        assert kwargs["itl_per_chunk_mean"] == 26.4
        assert kwargs["itl_per_chunk_p95"] == 27.0
        assert kwargs["itl_per_chunk_p99"] == 800.0
        assert kwargs["itl_per_chunk_max"] == 900.0

    def test_itl_columns_do_not_shadow_the_tpot_columns(self):
        """The two per-token metrics must not be wired to the same source.

        `inter_token_latency_ms` is guidellm's per-request value (the TPOT);
        `inter_token_latency_per_chunk_ms` is the measured gap distribution. A
        copy-paste that pointed both at one field would make the report compare
        a metric against itself.
        """
        metrics = _metrics(
            inter_token_latency_ms=_dist(mean=500.0, p99=500.0),
            inter_token_latency_per_chunk_ms=_dist(mean=26.4, p99=800.0),
        )

        kwargs = _itl_kwargs(metrics)

        assert kwargs["inter_token_latency_mean"] == 500.0
        assert kwargs["inter_token_latency_p99"] == 500.0
        assert kwargs["itl_per_chunk_mean"] == 26.4
        assert kwargs["itl_per_chunk_p99"] == 800.0

    def test_an_unmeasured_run_leaves_the_itl_columns_null(self):
        """A point from before the gaps were recorded, or a non-streaming run.

        NULL, not 0.0: a zero here would read as a decode with no latency
        between tokens at all.
        """
        kwargs = _itl_kwargs(_metrics())

        assert kwargs["itl_per_chunk_mean"] is None
        assert kwargs["itl_per_chunk_p95"] is None
        assert kwargs["itl_per_chunk_p99"] is None
        assert kwargs["itl_per_chunk_max"] is None

    def test_a_distribution_with_no_successful_samples_is_null(self):
        """Only failed requests produced gaps.

        The `successful` summary still exists and every field on it reads 0.0,
        which would land in the report as a perfect decode.
        """
        kwargs = _itl_kwargs(
            _metrics(
                inter_token_latency_per_chunk_ms=_dist(
                    mean=0.0, p95=0.0, p99=0.0, maximum=0.0, count=0
                )
            )
        )

        assert kwargs["itl_per_chunk_mean"] is None
        assert kwargs["itl_per_chunk_max"] is None

    def test_every_itl_column_exists_on_the_row_schema(self):
        """The kwargs must be real columns, or the upload silently drops them."""
        kwargs = _itl_kwargs(_metrics(inter_token_latency_per_chunk_ms=_dist()))
        columns = bm_schemas.BenchmarkMetricsLite.model_fields

        for name in kwargs:
            assert name in columns, f"{name} is not a BenchmarkMetricsLite column"


class TestForeignBenchmarksAreLeftAlone:
    """A worker must not reconcile a run that belongs to another worker.

    The 3-second state poll asks the server to filter by `worker_id`, and
    FastAPI drops an unknown query parameter without complaining, so a missing
    filter hands every worker the whole cluster's RUNNING rows. For a foreign
    row the reconcile path is wrong at every step -- `get_workload` reads the
    *local* runtime and returns None, `_is_workload_failed(None)` is True, and
    the terminal-artifact escape hatch checks the *local* disk -- so it ends in
    `_handle_benchmark_failure`: the non-owner patches the run to ERROR and
    tears down its workload while the owner is still running it.
    """

    @staticmethod
    def _manager(worker_id):
        mgr = _bare_manager()
        mgr._worker_id_getter = lambda: worker_id

        # Anything reached past the guard would have to go through these; they
        # raise so the test fails loudly rather than quietly taking a branch.
        def _boom(*_a, **_k):  # pragma: no cover - must never run
            raise AssertionError("reconciled a benchmark owned by another worker")

        mgr._is_torn_down = _boom
        mgr._is_benchmark_timed_out = _boom
        mgr._is_provisioning = _boom
        return mgr

    def test_a_run_owned_by_another_worker_is_skipped(self):
        mgr = self._manager(worker_id=4)
        foreign = SimpleNamespace(id=155, worker_id=3, name="pd-1p2-7d2-d3")

        assert mgr._sync_single_benchmark_state(foreign) is None

    def test_a_run_this_worker_owns_still_gets_reconciled(self):
        """The guard must not swallow the worker's own rows."""
        mgr = _bare_manager()
        mgr._worker_id_getter = lambda: 3
        seen = []
        mgr._is_torn_down = lambda bid: seen.append(bid) or True

        own = SimpleNamespace(id=155, worker_id=3, name="pd-1p2-7d2-d3")
        mgr._sync_single_benchmark_state(own)

        assert seen == [155], "own benchmark must reach the reconcile path"

    def test_a_row_with_no_owner_is_not_claimed(self):
        """`worker_id` is Optional on the schema. An unassigned row belongs to
        nobody, so no worker may tear it down -- least of all every worker."""
        mgr = self._manager(worker_id=4)
        orphan = SimpleNamespace(id=200, worker_id=None, name="unassigned")

        assert mgr._sync_single_benchmark_state(orphan) is None


class TestTpotThresholdsAreForwardedAsItlFlags:
    """gpustack says TPOT; benchmark-runner and guidellm say ITL.

    Both names denote the decode-only per-token time. The vocabularies differ on
    purpose: benchmark-runner is a wrapper around guidellm, so it speaks
    guidellm's, while TPOT is what gpustack's API field, form and column have
    always been called. SLO_THRESHOLDS carries the translation — the attr stays
    `slo_*_tpot_ms`, the flag goes out as `--slo-*-itl-ms` — so it happens once
    here instead of at each call site.

    Pinned by a test because the two sides look inconsistent when read alone,
    and "tidying" the flag back to `--slo-*-tpot-ms` would send an argument the
    runner only still accepts as a deprecated alias.
    """

    def test_tpot_thresholds_go_out_under_the_itl_flag(self):
        by_attr = {t.attr: t.flag for t in bm_schemas.SLO_THRESHOLDS}

        assert by_attr["slo_avg_tpot_ms"] == "--slo-avg-itl-ms"
        assert by_attr["slo_p95_tpot_ms"] == "--slo-p95-itl-ms"
        assert by_attr["slo_p99_tpot_ms"] == "--slo-p99-itl-ms"

    def test_the_other_thresholds_keep_their_own_names(self):
        """Only the per-token pair is renamed across the boundary."""
        by_attr = {t.attr: t.flag for t in bm_schemas.SLO_THRESHOLDS}

        assert by_attr["slo_avg_ttft_ms"] == "--slo-avg-ttft-ms"
        assert by_attr["slo_avg_latency_ms"] == "--slo-avg-latency-ms"

    def test_no_threshold_still_emits_the_old_tpot_flag(self):
        flags = {t.flag for t in bm_schemas.SLO_THRESHOLDS}

        assert not any("tpot" in flag for flag in flags)


class TestTargetGoneDetection:
    """A run whose target disappears must fail, not wait out the load
    generator's per-request timeout.

    Observed on b255: the model behind `...-r4-router-yj3i1` was deleted about
    seven minutes into the run. Nothing in the reconcile path noticed -- the
    container was healthy, it was the endpoint that was gone -- so the row sat
    RUNNING at 100% progress for the rest of the hour that the runner's
    `timeout: 3600` backend setting takes to give up. `max_errors` and
    `max_error_rate` were both unset, so the flood of failed requests was not a
    stopping condition either.
    """

    @staticmethod
    def _manager(instances, *, mode=bm_schemas.BenchmarkTargetModeEnum.INSTANCE):
        mgr = _bare_manager()
        calls = []

        def _list():
            calls.append(1)
            if isinstance(instances, Exception):
                raise instances
            return SimpleNamespace(items=list(instances))

        mgr._clientset_getter = lambda: SimpleNamespace(
            model_instances=SimpleNamespace(list=_list)
        )
        mgr._list_calls = calls
        return mgr

    @staticmethod
    def _benchmark(mode=bm_schemas.BenchmarkTargetModeEnum.INSTANCE):
        return SimpleNamespace(
            id=255,
            name="pdc-asc-glm47-pd-1p2-3d2-custom",
            target_mode=mode,
            model_instance_name="router-yj3i1",
        )

    @staticmethod
    def _instance(name, state=bm.ModelInstanceStateEnum.RUNNING):
        return SimpleNamespace(name=name, state=state)

    def test_a_running_target_is_not_gone(self):
        mgr = self._manager([self._instance("router-yj3i1")])

        assert mgr._is_target_gone(self._benchmark()) is False

    def test_a_missing_target_is_tolerated_inside_the_grace_period(self):
        """An instance can leave RUNNING briefly. Twenty minutes of measurement
        must not be thrown away over a restart that resolves in seconds."""
        mgr = self._manager([self._instance("someone-else")])
        benchmark = self._benchmark()

        # First sighting only starts the clock.
        assert mgr._is_target_gone(benchmark) is False
        assert benchmark.id in mgr._target_missing_since

        # Still missing, but not for long enough yet.
        mgr._target_missing_since[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_GONE_GRACE_SECONDS + 30
        )
        mgr._last_target_check_at[benchmark.id] = 0.0
        assert mgr._is_target_gone(benchmark) is False

    def test_a_target_missing_past_the_grace_period_fails_the_run(self):
        mgr = self._manager([self._instance("someone-else")])
        benchmark = self._benchmark()
        mgr._target_missing_since[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_GONE_GRACE_SECONDS - 1
        )

        assert mgr._is_target_gone(benchmark) is True

    def test_a_target_that_left_running_counts_as_missing(self):
        """Present in the list but not RUNNING is not something to measure."""
        mgr = self._manager(
            [self._instance("router-yj3i1", state=bm.ModelInstanceStateEnum.ERROR)]
        )
        benchmark = self._benchmark()
        mgr._target_missing_since[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_GONE_GRACE_SECONDS - 1
        )

        assert mgr._is_target_gone(benchmark) is True

    def test_a_target_coming_back_resets_the_clock(self):
        mgr = self._manager([self._instance("router-yj3i1")])
        benchmark = self._benchmark()
        mgr._target_missing_since[benchmark.id] = time.time() - 120

        assert mgr._is_target_gone(benchmark) is False
        assert benchmark.id not in mgr._target_missing_since

    def test_an_empty_instance_list_is_unknown_not_gone(self):
        """A reconnecting watch clears its cache before replaying it. Reading
        that window as "gone" would fail every running benchmark in the cluster
        at once, on nothing but a dropped stream."""
        mgr = self._manager([])
        benchmark = self._benchmark()
        mgr._target_missing_since[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_GONE_GRACE_SECONDS - 1
        )

        assert mgr._is_target_gone(benchmark) is False

    def test_a_failed_lookup_is_unknown_not_gone(self):
        mgr = self._manager(RuntimeError("connection refused"))
        benchmark = self._benchmark()
        mgr._target_missing_since[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_GONE_GRACE_SECONDS - 1
        )

        assert mgr._is_target_gone(benchmark) is False

    def test_route_mode_is_exempt_and_never_looks_up(self):
        """`route` mode exists precisely because the instances behind a route
        come and go; the same observation is not a failure there."""
        mgr = self._manager([])
        benchmark = self._benchmark(mode=bm_schemas.BenchmarkTargetModeEnum.ROUTE)

        assert mgr._is_target_gone(benchmark) is False
        assert mgr._list_calls == [], "route mode must not query instances at all"

    def test_a_row_without_a_pinned_instance_is_exempt(self):
        mgr = self._manager([])
        benchmark = self._benchmark()
        benchmark.model_instance_name = None

        assert mgr._is_target_gone(benchmark) is False
        assert mgr._list_calls == []

    def test_the_lookup_is_throttled_between_checks(self):
        """The check runs off the 3-second state poll but must not look up that
        often: the instance cache falls back to a full GET while its watch is
        reconnecting."""
        mgr = self._manager([self._instance("router-yj3i1")])
        benchmark = self._benchmark()

        mgr._is_target_gone(benchmark)
        mgr._is_target_gone(benchmark)
        mgr._is_target_gone(benchmark)

        assert mgr._list_calls == [1], "throttle window allows exactly one lookup"

        mgr._last_target_check_at[benchmark.id] = (
            time.time() - bm.BENCHMARK_TARGET_CHECK_INTERVAL_SECONDS - 1
        )
        mgr._is_target_gone(benchmark)

        assert len(mgr._list_calls) == 2, "a lookup resumes once the window passes"


class TestTargetGoneIsCheckedBeforeTheWorkload:
    """Ordering matters: in this failure the container is healthy, so every
    workload-based check below says "still running" and the run is left alone."""

    def test_the_poll_fails_the_run_before_it_looks_at_the_workload(self, monkeypatch):
        mgr = _bare_manager()
        mgr._worker_id_getter = lambda: 3
        mgr._is_torn_down = lambda _bid: False
        mgr._is_benchmark_timed_out = lambda _b: False
        mgr._is_target_gone = lambda _b: True

        handled = []
        mgr._handle_target_gone = lambda b: handled.append(b.id)

        def _boom(*_a, **_k):  # pragma: no cover - must never run
            raise AssertionError("workload was inspected despite a missing target")

        mgr._is_provisioning = _boom
        monkeypatch.setattr(bm, "get_workload", _boom)

        benchmark = SimpleNamespace(id=255, worker_id=3, name="b255")
        mgr._sync_single_benchmark_state(benchmark)

        assert handled == [255]


class TestADeletedRowDoesNotBlockTheQueue:
    """Deleting a RUNNING benchmark must not stall the worker's whole queue.

    `_active_benchmark_id` is what serializes runs on a worker — while it is
    set, `_benchmark_queue_worker` pops nothing — so every path that releases it
    must run regardless of the state patch. Patching a row the user has just
    deleted returns 404; if `raise_for_status` turned that into an exception the
    teardown underneath it would never run, the active id would stay set
    forever, the 3-second poll would re-enter the same 404, and every benchmark
    dispatched to that worker afterwards would sit unstarted while the worker
    still looked `ready`.
    """

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code
            self.raised = False

        def raise_for_status(self):
            if self.status_code >= 400:
                self.raised = True
                raise RuntimeError(f"HTTP {self.status_code}")

    def test_a_404_is_reported_not_raised(self):
        """404 is the server stating the row is gone — information, not a
        failed write."""
        resp = self._Resp(404)

        assert BenchmarkManager._state_patch_landed(255, resp) is False
        assert not resp.raised

    def test_any_other_status_still_raises(self):
        """A real write failure must keep failing loudly, so the callers that
        retry keep retrying."""
        resp = self._Resp(500)

        with pytest.raises(RuntimeError):
            BenchmarkManager._state_patch_landed(255, resp)

    def test_a_204_lands(self):
        assert BenchmarkManager._state_patch_landed(255, self._Resp(204)) is True

    def _mgr(self):
        mgr = _bare_manager()
        mgr._worker_id_getter = lambda: 7
        mgr._canceled_ids = set()
        mgr._torn_down_ids = bm.OrderedDict()
        mgr._provisioning_processes = {}
        mgr._benchmark_by_id = {}
        mgr._container_log_offset = {}
        mgr._last_log_snapshot_at = {}
        mgr._active_benchmark_id = None
        mgr._active_benchmark_started_at = None
        mgr._is_provisioning = lambda _b: False
        return mgr

    def test_the_failure_path_still_tears_down_a_deleted_row(self, monkeypatch):
        """The state patch 404s because the row is gone; the teardown that
        follows it is exactly what still has to happen."""
        deleted = []
        monkeypatch.setattr(
            bm, "delete_workload", lambda *a, **k: deleted.append("workload")
        )
        mgr = self._mgr()
        mgr._active_benchmark_id = 255
        mgr._finalize_partial_analysis = lambda _b: None
        mgr._dump_benchmark_logs_to_file = lambda _b: None
        mgr._update_benchmark_state_sync = lambda _id, **_kw: False

        benchmark = SimpleNamespace(id=255, name="b255", namespace=None)
        mgr._handle_benchmark_failure(benchmark)

        assert deleted == ["workload"]
        assert mgr._active_benchmark_id is None

    def test_one_bad_row_does_not_abort_the_round(self):
        """The condition that raises lives in the row, not in the round, so an
        aborted round would repeat forever — taking every other run on the
        worker down with it."""
        mgr = self._mgr()
        mgr._clientset_getter = lambda: SimpleNamespace(
            benchmarks=SimpleNamespace(
                list=lambda params=None: SimpleNamespace(
                    items=[
                        SimpleNamespace(id=1, name="first"),
                        SimpleNamespace(id=2, name="second"),
                    ]
                )
            )
        )
        seen = []

        def _sync(benchmark):
            seen.append(benchmark.id)
            if benchmark.id == 1:
                raise RuntimeError("HTTP 404")

        mgr._sync_single_benchmark_state = _sync
        mgr.sync_benchmark_state()

        assert seen == [1, 2]

    def _mgr_with_get(self, status_code=None, error=None):
        mgr = self._mgr()
        asked = []

        def _get(path):
            asked.append(path)
            if error is not None:
                raise error
            return SimpleNamespace(status_code=status_code)

        mgr._clientset_getter = lambda: SimpleNamespace(
            http_client=SimpleNamespace(
                get_httpx_client=lambda: SimpleNamespace(get=_get)
            )
        )
        mgr._asked = asked
        return mgr

    def test_an_active_run_deleted_behind_the_watch_is_reaped(self, monkeypatch):
        """The DELETED event is not guaranteed to arrive: a watch that was
        disconnected or replaying when the row went away never delivers it, and
        the poll lists rows FROM the server, so the id is absent from every
        later tick. Nothing else would ever release the queue."""
        deleted = []
        monkeypatch.setattr(
            bm, "delete_workload", lambda *a, **k: deleted.append("workload")
        )
        mgr = self._mgr_with_get(status_code=404)
        mgr._active_benchmark_id = 255
        mgr._benchmark_by_id[255] = SimpleNamespace(id=255, name="b255", namespace=None)

        mgr._reap_deleted_active_benchmark(listed_ids=set())

        assert mgr._asked == ["/benchmarks/255"]
        assert deleted == ["workload"]
        assert mgr._active_benchmark_id is None
        assert mgr._is_torn_down(255)

    def test_a_row_that_still_exists_is_left_alone(self):
        """Absent from the RUNNING page only means "not RUNNING" — queued,
        just-stopped, mid-transition. None of those are this method's business."""
        mgr = self._mgr_with_get(status_code=200)
        mgr._active_benchmark_id = 255

        mgr._reap_deleted_active_benchmark(listed_ids=set())

        assert mgr._active_benchmark_id == 255

    def test_an_unreachable_server_is_not_a_deletion(self):
        """A failed lookup must never read as a deletion, or a blip would tear
        down a healthy run."""
        mgr = self._mgr_with_get(error=RuntimeError("connection refused"))
        mgr._active_benchmark_id = 255

        mgr._reap_deleted_active_benchmark(listed_ids=set())

        assert mgr._active_benchmark_id == 255

    def test_a_listed_run_costs_no_request(self):
        """The normal case: the active run is in the RUNNING page the poll just
        fetched, so the existence check never fires."""
        mgr = self._mgr_with_get(status_code=404)
        mgr._active_benchmark_id = 255

        mgr._reap_deleted_active_benchmark(listed_ids={255})

        assert mgr._asked == []
        assert mgr._active_benchmark_id == 255

    def test_the_queue_is_released_even_with_nothing_to_tear_down(self):
        """A leaked container is recoverable and stated in the log; a stalled
        queue is neither."""
        mgr = self._mgr_with_get(status_code=404)
        mgr._active_benchmark_id = 255

        mgr._reap_deleted_active_benchmark(listed_ids=set())

        assert mgr._active_benchmark_id is None

    def test_a_user_stop_releases_the_queue_even_if_the_analysis_raises(self):
        """The row is no longer RUNNING, so the poll will never revisit it —
        this handler is the only chance to release the active id."""
        import asyncio

        mgr = self._mgr()
        mgr._active_benchmark_id = 255
        mgr._stop_benchmark = lambda b: mgr._clear_active_benchmark(b.id)

        def _boom(_b):
            raise RuntimeError("unreadable point file")

        mgr._finalize_partial_analysis = _boom
        mgr._dump_benchmark_logs_to_file = lambda _b: None

        benchmark = SimpleNamespace(id=255, name="b255", namespace=None)
        asyncio.run(mgr._handle_stop_benchmark_event(benchmark))

        assert mgr._active_benchmark_id is None

    def test_a_row_deleted_before_it_could_be_queued_is_dropped(self):
        """Deleted between the PENDING event and the QUEUED patch. Marked
        canceled rather than popped: the deque is shared with the sync thread,
        and the existing guard drops the entry race-safely when the queue worker
        reaches it."""
        import asyncio

        mgr = self._mgr()
        mgr._benchmark_queue = deque()
        mgr._queue_lock = asyncio.Lock()

        async def _gone(_id, **_kw):
            return False

        mgr._update_benchmark_state = _gone

        benchmark = SimpleNamespace(id=258, name="b258", namespace=None)
        asyncio.run(mgr._enqueue_benchmark(benchmark))

        assert 258 in mgr._canceled_ids
