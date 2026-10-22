"""Worker TLS policy and trust passed to the benchmark container.

The benchmark container is a separate ``gpustack/benchmark-runner`` image whose
entrypoint is the runner CLI, so it never runs ``gpustack-prerun.sh`` and never
sees a private CA an operator imported on the worker. Nothing about the worker's
trust store reaches it by itself.

So the worker's own CA bundle is handed over as container file content, and the
runner is pointed at it with ``--progress-ca-cert``. Both halves matter:

* As *content*, the runtime writes it into the container itself. Nothing is
  written to a directory the container can reach beforehand, so there is no
  destination for it to hijack, and nothing outlives the run.
* The progress client uses ``--progress-ca-cert``. HTTPS route inference also
  receives the same injected bundle through ``SSL_CERT_FILE``; that path is
  covered by the runner target tests.
"""

import re
import ssl
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import certifi
import httpx
import pytest

import gpustack.config.config as config_module
from gpustack import envs
import gpustack.worker.benchmark.runner as bm_runner
from gpustack.worker.benchmark.runner import BenchmarkRunner


class TestProgressCAFile:
    """What gets handed to the container, and when."""

    def _runner(self, api_url, *, insecure=False):
        return SimpleNamespace(
            _api_url=api_url,
            _progress_insecure_skip_tls_verify=insecure,
            _progress_is_https=urlparse(api_url).scheme == "https",
        )

    def _bundle(self, tmp_path, body="-----BEGIN CERTIFICATE-----\n"):
        source = tmp_path / "worker-ca-bundle.crt"
        source.write_text(body)
        return source

    def test_https_hands_over_the_worker_bundle(self, tmp_path, monkeypatch):
        source = self._bundle(tmp_path)
        monkeypatch.setattr(bm_runner, "resolve_ca_bundle", lambda: str(source))

        ca_file = BenchmarkRunner._progress_ca_file(
            self._runner("https://gpustack.internal/v2/benchmarks/1/state")
        )

        assert ca_file is not None
        # Content, not a host path: the runtime writes this into the container.
        assert ca_file.content == source.read_text()
        assert ca_file.path == bm_runner.PROGRESS_CA_BUNDLE_PATH
        assert ca_file.mode == 0o644

    def test_nothing_is_written_on_the_host(self, tmp_path, monkeypatch):
        """The property that removes the whole hijack surface.

        The previous design copied the bundle into benchmark_dir, which is
        mounted ReadWriteMany into a container running as root -- so that
        container could plant a symlink at the destination and have the next
        run, also root, follow it.
        """
        source = self._bundle(tmp_path)
        monkeypatch.setattr(bm_runner, "resolve_ca_bundle", lambda: str(source))
        before = set(tmp_path.iterdir())

        BenchmarkRunner._progress_ca_file(
            self._runner("https://gpustack.internal/v2/benchmarks/1/state")
        )

        assert set(tmp_path.iterdir()) == before

    def test_plain_http_hands_over_nothing(self, tmp_path, monkeypatch):
        import pytest

        monkeypatch.setattr(
            bm_runner,
            "resolve_ca_bundle",
            lambda: pytest.fail("must not resolve a bundle for http://"),
        )

        assert (
            BenchmarkRunner._progress_ca_file(
                self._runner("http://127.0.0.1:80/v2/benchmarks/1/state")
            )
            is None
        )

    def test_insecure_mode_hands_over_nothing(self, tmp_path, monkeypatch):
        import pytest

        monkeypatch.setattr(
            bm_runner,
            "resolve_ca_bundle",
            lambda: pytest.fail("verification is off; no bundle is consulted"),
        )

        assert (
            BenchmarkRunner._progress_ca_file(
                self._runner(
                    "https://gpustack.internal/v2/benchmarks/1/state", insecure=True
                )
            )
            is None
        )

    def test_an_unreadable_bundle_is_not_fatal(self, tmp_path, monkeypatch):
        """The measurement only needs the model endpoint, which is plain HTTP.

        A run must not be lost over progress telemetry, so this warns and the
        caller simply does not forward --progress-ca-cert.
        """
        monkeypatch.setattr(
            bm_runner, "resolve_ca_bundle", lambda: str(tmp_path / "does-not-exist")
        )

        assert (
            BenchmarkRunner._progress_ca_file(
                self._runner("https://gpustack.internal/v2/benchmarks/1/state")
            )
            is None
        )


class TestWorkerCAEnvIsDropped:
    """Inherited CA variables name paths in the WORKER's filesystem.

    filter_env_vars has no reason to know that, so it passes them through. None
    of them fall back when the file is missing: SSL_CERT_FILE loads zero
    certificates, and requests raises OSError outright.
    """

    def test_the_bundle_variables_are_dropped(self):
        env = {
            "SSL_CERT_FILE": "/worker/only/ca.crt",
            "REQUESTS_CA_BUNDLE": "/worker/only/ca.crt",
            "CURL_CA_BUNDLE": "/worker/only/ca.crt",
            "HF_TOKEN": "keep-me",
        }

        BenchmarkRunner._drop_worker_ca_env(SimpleNamespace(), env)

        assert "SSL_CERT_FILE" not in env
        assert "REQUESTS_CA_BUNDLE" not in env
        assert "CURL_CA_BUNDLE" not in env
        assert env["HF_TOKEN"] == "keep-me"

    def test_the_cert_dir_is_kept(self):
        """SSL_CERT_DIR is not the same hazard: a missing capath is ignored, and
        the value IS valid in the container when it names a standard path."""
        env = {"SSL_CERT_DIR": "/etc/ssl/certs"}

        BenchmarkRunner._drop_worker_ca_env(SimpleNamespace(), env)

        assert env["SSL_CERT_DIR"] == "/etc/ssl/certs"


@pytest.mark.asyncio
@pytest.mark.parametrize("trust_env", [True, False])
async def test_httpx_loads_injected_ca_bundle_when_verification_is_enabled(
    tmp_path, monkeypatch, trust_env
):
    """Verify the HTTPX contract used by GuideLLM without making network requests."""
    roots = Path(certifi.where()).read_text()
    certificate = re.search(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", roots, re.S
    ).group()
    bundle = tmp_path / "injected-ca.pem"
    bundle.write_text(certificate + "\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    # GuideLLM leaves trust_env at HTTPX's default. The negative control shows
    # that selecting the injected bundle depends on environment trust.
    options = {} if trust_env else {"trust_env": False}
    async with httpx.AsyncClient(verify=True, **options) as client:
        context = client._transport._pool._ssl_context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        trusted = context.get_ca_certs(binary_form=True)

    injected = ssl.PEM_cert_to_DER_cert(certificate)
    if trust_env:
        assert trusted == [injected]
    else:
        assert injected in trusted
        assert len(trusted) > 1


class TestInsecureFlagForwarding:
    """Whether the runner is told to skip verification.

    This gating decides whether a container starts at all: the option only exists
    on runner images from v0.0.4.post1 on, so forwarding it to an older pinned
    image is a hard startup failure (unknown option, exit 2). It is also the one
    piece of TLS handling that runs under a mirrored deployment, where the rest
    of it is skipped.
    """

    def _runner(self, api_url, *, insecure):
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
        return SimpleNamespace(
            _benchmark=benchmark,
            _model_endpoint="http://127.0.0.1:8000",
            _model_path="/models/qwen3-0.6b",
            _model_backend_parameters=None,
            _benchmark_dir="/var/lib/gpustack/benchmarks",
            _api_url=api_url,
            _api_key="token",
            _progress_insecure_skip_tls_verify=insecure,
            _progress_is_https=urlparse(api_url).scheme == "https",
        )

    def test_the_ca_cert_points_at_the_injected_file(self):
        """Its value is the in-container path, which only the injection defines.

        The two are decided together in start(), so they cannot disagree about
        whether the file is there.
        """
        args = BenchmarkRunner._build_command_args(
            self._runner(
                "https://gpustack.internal/v2/benchmarks/1/state", insecure=False
            ),
            with_ca_cert=True,
        )

        assert (
            args[args.index("--progress-ca-cert") + 1]
            == bm_runner.PROGRESS_CA_BUNDLE_PATH
        )

    def test_no_ca_cert_when_nothing_was_injected(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(
                "https://gpustack.internal/v2/benchmarks/1/state", insecure=False
            ),
            with_ca_cert=False,
        )

        assert "--progress-ca-cert" not in args

    def test_forwarded_for_https(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(
                "https://gpustack.internal/v2/benchmarks/1/state", insecure=True
            )
        )
        assert "--progress-insecure-skip-tls-verify" in args

    def test_omitted_for_plain_http(self):
        # Nothing to skip, and forwarding it over HTTP would risk the
        # unknown-option failure for no gain.
        args = BenchmarkRunner._build_command_args(
            self._runner("http://127.0.0.1:80/v2/benchmarks/1/state", insecure=True)
        )
        assert "--progress-insecure-skip-tls-verify" not in args

    def test_omitted_when_disabled(self):
        args = BenchmarkRunner._build_command_args(
            self._runner(
                "https://gpustack.internal/v2/benchmarks/1/state", insecure=False
            )
        )
        assert "--progress-insecure-skip-tls-verify" not in args

    def test_the_warning_travels_with_the_flag(self, caplog):
        """The log line naming the cause must be emitted wherever the flag is.

        It used to live with the rest of the TLS handling, which a mirrored
        deployment skips -- so exactly the operator whose container died on the
        unknown option got no explanation.
        """
        with caplog.at_level("WARNING"):
            BenchmarkRunner._build_command_args(
                self._runner(
                    "https://gpustack.internal/v2/benchmarks/1/state", insecure=True
                )
            )

        assert "Worker insecure TLS is enabled" in caplog.text
        assert "--progress-insecure-skip-tls-verify" in caplog.text

    def test_the_worker_token_is_passed_to_the_container(self):
        # The container needs the real credential; only the LOG of it is redacted.
        args = BenchmarkRunner._build_command_args(
            self._runner(
                "https://gpustack.internal/v2/benchmarks/1/state", insecure=False
            )
        )
        assert args[args.index("--progress-auth") + 1] == "token"


class TestResolveProgressInsecureTLS:
    """Progress and inference resolve the same environment/config policy as the worker."""

    def test_off_by_default(self):
        assert bm_runner.resolve_progress_insecure_tls() is False

    def test_the_worker_wide_switch_enables_it(self, monkeypatch):
        monkeypatch.setattr(envs, "INSECURE_TLS", True)
        assert bm_runner.resolve_progress_insecure_tls() is True

    @pytest.mark.parametrize(
        "env_insecure,config_insecure,expected",
        [
            (False, False, False),
            (False, True, True),
            (True, False, True),
            (True, True, True),
        ],
    )
    def test_environment_and_config_use_the_worker_policy(
        self, monkeypatch, env_insecure, config_insecure, expected
    ):
        monkeypatch.setattr(envs, "INSECURE_TLS", env_insecure)
        monkeypatch.setattr(
            config_module,
            "get_global_config",
            lambda: SimpleNamespace(insecure_tls=config_insecure),
        )
        assert bm_runner.resolve_progress_insecure_tls() is expected
