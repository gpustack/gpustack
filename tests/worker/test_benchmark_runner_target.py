"""What the runner sends to the load generator: the model name, and the path
the tokenizer is read from.

Both are sent explicitly rather than implied by the member the load is aimed
at. Under PD, leaving the name for guidellm to discover fails before a request
is made, and taking the tokenizer path off that member points at a machine that
holds no weights.
"""

import json
from types import SimpleNamespace

import pytest

from gpustack.schemas.benchmark import (
    BenchmarkLoadTypeEnum,
    BenchmarkTargetModeEnum,
    BenchmarkSnapshot,
    ModelInstanceSnapshot,
)
from gpustack.worker.benchmark.runner import BenchmarkRunner, _local_model_snapshot


def _snapshot(name, worker_id, path, ports=None) -> ModelInstanceSnapshot:
    return ModelInstanceSnapshot(
        id=abs(hash(name)) % 1000,
        name=name,
        worker_id=worker_id,
        worker_ip="10.0.0.1",
        ports=ports or [40050],
        resolved_path=path,
        computed_resource_claim=None,
    )


def _benchmark(worker_id, members, **kwargs):
    return SimpleNamespace(
        id=1,
        name="bm",
        worker_id=worker_id,
        model_name=kwargs.pop("model_name", "qwen3"),
        snapshot=BenchmarkSnapshot(
            instances={m.name: m for m in members},
        ),
        **kwargs,
    )


class TestProcessorPath:
    def test_a_group_reads_the_tokenizer_from_the_placement_member(self):
        # The endpoint is the router: no weights, so its `resolved_path` is
        # None and following it would hand guidellm a path that is not there.
        router = _snapshot("r", worker_id=13, path=None)
        decode = _snapshot("d", worker_id=12, path="/cache/qwen3")
        benchmark = _benchmark(worker_id=12, members=[router, decode])
        assert _local_model_snapshot(benchmark, router) is decode

    def test_a_single_member_snapshot_uses_the_endpoint(self):
        # Every non-PD model, and every row written before the snapshot carried
        # the whole group.
        only = _snapshot("mi", worker_id=10, path="/cache/qwen3")
        benchmark = _benchmark(worker_id=10, members=[only])
        assert _local_model_snapshot(benchmark, only) is only

    def test_the_endpoint_wins_when_it_lives_here_too(self):
        # A group whose router happens to share a worker with a GPU member: the
        # endpoint's own path is as local as any, so nothing is gained by
        # picking a different member.
        endpoint = _snapshot("d", worker_id=12, path="/cache/qwen3")
        other = _snapshot("p", worker_id=11, path="/cache/qwen3")
        benchmark = _benchmark(worker_id=12, members=[endpoint, other])
        assert _local_model_snapshot(benchmark, endpoint) is endpoint

    def test_no_local_member_falls_back_rather_than_raising(self):
        # The caller reports "no resolved path" with the whole context; failing
        # here would say only that a dict lookup missed.
        router = _snapshot("r", worker_id=13, path=None)
        decode = _snapshot("d", worker_id=12, path="/cache/qwen3")
        benchmark = _benchmark(worker_id=99, members=[router, decode])
        assert _local_model_snapshot(benchmark, router) is router


class TestModelIsNamed:
    """Discovery is what broke a live 1P1D: guidellm asked the router for
    `GET /v1/models` and the response had no `data` key, so the run died with a
    KeyError before sending a request."""

    def _args(self, model_name):
        runner = object.__new__(BenchmarkRunner)
        runner._benchmark = SimpleNamespace(
            id=1,
            name="bm",
            model_name=model_name,
            load_type=BenchmarkLoadTypeEnum.CONCURRENCY,
            load_mode=None,
            request_rate=4,
            total_requests=None,
            max_seconds=None,
            stages=None,
            dataset_name="Random",
            dataset_input_tokens=1024,
            dataset_output_tokens=128,
            dataset_seed=1,
            dataset_seed_increment=None,
            turns=None,
            prefix_buckets=None,
            warmup=None,
            cooldown=None,
            max_errors=None,
            max_error_rate=None,
            stop_on_saturation=None,
        )
        runner._model_endpoint = "http://10.0.0.1:40050"
        runner._model_path = "/cache/qwen3"
        runner._model_backend_parameters = []
        runner._benchmark_dir = "/var/lib/gpustack/benchmarks"
        runner._api_url = "http://10.0.0.1:9091/v2/benchmarks/1/state"
        runner._api_key = "token"
        runner._progress_insecure_skip_tls_verify = False
        # `_progress_is_https` is derived from the progress URL, which is plain
        # HTTP here, so it answers False on its own.
        return runner._build_command_args()

    def test_the_model_is_passed_explicitly(self):
        args = self._args("qwen3")
        assert "--model" in args
        assert args[args.index("--model") + 1] == "qwen3"

    def test_a_row_without_a_name_does_not_send_the_string_none(self):
        # Nothing the server creates, but "--model None" would be a model id
        # the target rejects rather than a fallback to discovery.
        args = self._args(None)
        assert "--model" not in args
        assert "None" not in args


@pytest.mark.parametrize("path", ["/cache/qwen3", "/other/path"])
def test_the_processor_is_the_local_path(path):
    endpoint = _snapshot("mi", worker_id=10, path=path)
    benchmark = _benchmark(worker_id=10, members=[endpoint])
    assert _local_model_snapshot(benchmark, endpoint).resolved_path == path


class TestRouteModeAimsAtTheDeployment:
    """In `route` mode three things change together: where the load goes, what
    it asks for by name, and whether it authenticates. Getting one without the
    others is a run that 404s or that measures the wrong thing."""

    def _runner(self, *, mode, route_name, insecure=False):
        runner = object.__new__(BenchmarkRunner)
        runner._benchmark = SimpleNamespace(
            id=1,
            name="bm",
            model_name="qwen3",
            target_mode=mode,
            load_type=BenchmarkLoadTypeEnum.CONCURRENCY,
            load_mode=None,
            request_rate=4,
            total_requests=None,
            max_seconds=None,
            stages=None,
            dataset_name="Random",
            dataset_input_tokens=1024,
            dataset_output_tokens=128,
            dataset_seed=1,
            dataset_seed_increment=None,
            turns=None,
            prefix_buckets=None,
            warmup=None,
            cooldown=None,
            max_errors=None,
            max_error_rate=None,
            stop_on_saturation=None,
        )
        runner._route_name = route_name
        runner._model_endpoint = "http://10.0.0.1:40050"
        runner._model_path = "/cache/qwen3"
        runner._model_backend_parameters = []
        runner._benchmark_dir = "/var/lib/gpustack/benchmarks"
        runner._api_url = "http://10.0.0.9:9091/v2/benchmarks/1/state"
        runner._api_key = "worker-token"
        runner._progress_insecure_skip_tls_verify = insecure
        return runner

    def _args(self, **kwargs):
        return self._runner(**kwargs)._build_command_args()

    def test_the_load_goes_at_the_proxy_not_the_member(self):
        args = self._args(mode=BenchmarkTargetModeEnum.ROUTE, route_name="org1/qwen")
        target = args[args.index("--target") + 1]
        # Stops before /v1, which the load generator appends itself.
        assert target == "http://10.0.0.9:9091/v2/benchmark-proxy"

    def test_the_name_asked_for_is_the_routes(self):
        # The proxy matches on the route name; the model's own name may not
        # even be addressable (another Org's route is prefixed).
        args = self._args(mode=BenchmarkTargetModeEnum.ROUTE, route_name="org1/qwen")
        assert args[args.index("--model") + 1] == "org1/qwen"

    def test_it_authenticates_with_the_worker_token(self):
        # The proxy is the server, and the server authenticates. Nothing of the
        # user's is stored on the run.
        args = self._args(mode=BenchmarkTargetModeEnum.ROUTE, route_name="qwen")
        kwargs = json.loads(args[args.index("--backend-kwargs") + 1])
        assert kwargs["api_key"] == "worker-token"

    def test_instance_mode_is_untouched(self):
        args = self._args(mode=BenchmarkTargetModeEnum.INSTANCE, route_name=None)
        assert args[args.index("--target") + 1] == "http://10.0.0.1:40050"
        assert args[args.index("--model") + 1] == "qwen3"
        kwargs = json.loads(args[args.index("--backend-kwargs") + 1])
        assert "api_key" not in kwargs

    def test_route_mode_without_a_resolved_route_stays_on_the_member(self):
        # Belt and braces: the server refuses to create such a run, so this is
        # a row from a downgrade rather than a shape a client can ask for —
        # measuring the member beats sending the load at a proxy with no name
        # to give it.
        args = self._args(mode=BenchmarkTargetModeEnum.ROUTE, route_name=None)
        assert args[args.index("--target") + 1] == "http://10.0.0.1:40050"

    def test_an_insecure_worker_does_not_verify_the_proxys_tls(self):
        # Same switch the progress channel already honours: one worker, one
        # answer about whether it can verify this server.
        args = self._args(
            mode=BenchmarkTargetModeEnum.ROUTE, route_name="qwen", insecure=True
        )
        kwargs = json.loads(args[args.index("--backend-kwargs") + 1])
        assert kwargs["verify"] is False


# ── warmup / cooldown: which axis gets trimmed ───────────────────────────────


class TestTransientPhaseAxis:
    """Warmup/cooldown must trim REQUESTS, not seconds.

    guidellm's `TransientPhaseConfig` defaults to `mode="prefer_duration"`, and
    its `compute_limits` drops the request bound whenever a duration can also be
    computed. Our stages always set both `max_requests` and `max_seconds`, so a
    bare number was always read as a slice of time.

    That biases arm-to-arm comparison: trimming by time removes a different
    population of requests from a saturated arm than from an idle one.
    """

    def test_fraction_is_sent_as_a_request_percentage(self):
        from gpustack.worker.benchmark.runner import _transient_phase_arg

        assert json.loads(_transient_phase_arg(0.1)) == {
            "percent": 0.1,
            "mode": "requests",
        }

    def test_one_and_above_stays_an_absolute_count(self):
        """The column keeps guidellm's scalar convention; only the axis is pinned."""
        from gpustack.worker.benchmark.runner import _transient_phase_arg

        assert json.loads(_transient_phase_arg(10)) == {
            "value": 10,
            "mode": "requests",
        }

    def test_the_boundary_is_one_not_zero(self):
        from gpustack.worker.benchmark.runner import _transient_phase_arg

        assert "percent" in json.loads(_transient_phase_arg(0.999))
        assert "value" in json.loads(_transient_phase_arg(1.0))

    def test_mode_is_always_pinned(self):
        """Without this the duration axis wins and the trim is time-based."""
        from gpustack.worker.benchmark.runner import _transient_phase_arg

        for v in (0.05, 0.5, 1, 300):
            assert json.loads(_transient_phase_arg(v))["mode"] == "requests"
