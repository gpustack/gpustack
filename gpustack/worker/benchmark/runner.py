import json
import logging
import os
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse

from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config, set_global_config
from gpustack.config.registration import read_worker_token
from gpustack.envs import BENCHMARK_DATASET_SHAREGPT_PATH, BENCHMARK_REQUEST_TIMEOUT
from gpustack.logging import setup_logging
from gpustack.ssl_context import resolve_ca_bundle
from gpustack.schemas.benchmark import (
    DATASET_RANDOM,
    DATASET_SHAREGPT,
    Benchmark,
    BenchmarkDeploymentMetadata,
    BenchmarkStateEnum,
    ModelInstanceSnapshot,
)
from gpustack.utils.command import find_bool_parameter, sanitize_args
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.envs import filter_env_vars, get_gpustack_env_bool, sanitize_env
from gpustack_runtime.logging import setup_logging as setup_runtime_logging
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import ContainerFile, ContainerMount

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerRestartPolicyEnum,
)

from gpustack.utils.profiling import time_decorator
from gpustack.utils.runtime import transform_workload_plan

logger = logging.getLogger(__name__)

# Where the CA bundle lands inside the benchmark container. It is injected as
# file content by the runtime (see BenchmarkRunner._progress_ca_file), so this
# path only has to make sense in the container -- nothing is written on the host.
PROGRESS_CA_BUNDLE_PATH = "/etc/gpustack/progress-ca-bundle.crt"


def resolve_progress_insecure_tls() -> bool:
    """Whether the benchmark container should skip TLS verification on progress.

    Driven by ``GPUSTACK_INSECURE_TLS`` alone, which a worker started with the
    enterprise plugin's ``--insecure-tls`` sets (operators without that plugin
    can set the variable directly). It means "this worker cannot verify the
    server's certificate", and progress reporting is exactly the worker talking
    to the server, so it already covers this case -- a benchmark-only switch
    would let benchmarks skip verification while the worker's own connection to
    the same server could not, which is not a distinction worth configuring.

    It has to be read here because it cannot propagate on its own: the
    enterprise shim carries it into spawned *interpreters* via
    ``PYTHONPATH``/``sitecustomize``, which a separate benchmark image does not
    have, and ``filter_env_vars`` strips ``GPUSTACK_*`` before the container env
    is built. So this process (a spawn child, which does inherit the variable) is
    the last place that can act on it.
    """
    return bool(get_gpustack_env_bool("INSECURE_TLS"))


class BenchmarkRunner:
    _clientset: ClientSet
    _config: Config
    _benchmark: Benchmark
    _model_path: str
    _model_endpoint: str
    _model_backend_parameters: Optional[List[str]]
    _api_url: str
    _api_key: str
    _benchmark_dir: Optional[str]
    _progress_insecure_skip_tls_verify: bool
    _fallback_registry: Optional[str] = None
    """The fallback container registry to use if needed."""

    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        benchmark: Benchmark,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        setup_logging(debug=cfg.debug)
        setup_runtime_logging()
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._benchmark = benchmark
            self._config = cfg
            self._fallback_registry = fallback_registry

            if (
                benchmark.snapshot is None
                or benchmark.snapshot.instances is None
                or len(benchmark.snapshot.instances) == 0
                or benchmark.snapshot.instances.get(benchmark.model_instance_name)
                is None
            ):
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) has no snapshot for model instance {benchmark.model_instance_name}"
                )

            instance_snapshot: ModelInstanceSnapshot = benchmark.snapshot.instances.get(
                benchmark.model_instance_name
            )
            if instance_snapshot.resolved_path is None:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no resolved path"
                )

            if instance_snapshot.worker_ip is None:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no worker IP"
                )

            if instance_snapshot.ports is None or len(instance_snapshot.ports) == 0:
                raise ValueError(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) snapshot for model instance {benchmark.model_instance_name} has no ports"
                )

            self._benchmark_dir = self._config.benchmark_dir
            self._model_path = instance_snapshot.resolved_path
            self._model_endpoint = f"http://{instance_snapshot.worker_ip}:{instance_snapshot.ports[0] if instance_snapshot.ports else ''}"
            self._model_backend_parameters = instance_snapshot.backend_parameters

            _api_key = read_worker_token(self._config.data_dir)
            if _api_key is None:
                raise ValueError(
                    f"Worker token not found for benchmark {benchmark.name}(id={benchmark.id}) progress reporting"
                )
            self._api_key = _api_key

            _server_url = self._clientset.base_url
            if not _server_url:
                raise ValueError(
                    f"Server URL not configured for benchmark {benchmark.name}(id={benchmark.id}) progress reporting"
                )
            self._api_url = (
                f"{_server_url.rstrip('/')}/v2/benchmarks/{self._benchmark.id}/state"
            )
            self._progress_insecure_skip_tls_verify = resolve_progress_insecure_tls()

        except Exception as e:
            error_message = f"Failed to initialize: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": BenchmarkStateEnum.ERROR,
                }
                self._update_benchmark_state(benchmark.id, **patch_dict)
            except Exception as ue:
                logger.error(
                    f"Failed to update benchmark {benchmark.name}(id={benchmark.id}) state: {ue}"
                )
            sys.exit(1)

    def start(self):
        deployment_metadata = self._benchmark.get_deployment_metadata()

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = filter_env_vars(os.environ)
            self._drop_worker_ca_env(env)

        # Resolved before the args so the two agree: --progress-ca-cert is only
        # forwarded when the file it names is actually going to be there.
        ca_file = self._progress_ca_file()
        command_args = self._build_command_args(with_ca_cert=ca_file is not None)
        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=["benchmark-runner"],
            command_args=command_args,
            env=env,
            files=[ca_file] if ca_file is not None else None,
        )

    @property
    def _progress_is_https(self) -> bool:
        """Whether progress reporting has to verify anything at all.

        Read from two places that must agree -- the env/bundle handling and the
        flag forwarding -- so it lives in one place rather than as two copies of
        the same urlparse.
        """
        return urlparse(self._api_url).scheme == "https"

    def _drop_worker_ca_env(self, env: Dict[str, str]) -> None:
        """Strip CA-bundle variables inherited from the worker.

        They name paths in the WORKER's filesystem, which the benchmark image
        does not have, and none of them fall back on their own when the file is
        missing: ``SSL_CERT_FILE`` makes OpenSSL load ZERO certificates, and
        ``REQUESTS_CA_BUNDLE`` makes requests raise OSError outright rather than
        verify against anything. Passing them through is worse than dropping
        them, since the container's own trust store is at least coherent.

        ``SSL_CERT_DIR`` is deliberately left alone: a missing capath is simply
        ignored, and the value IS valid in the container when it names a standard
        path both images have.
        """
        for stale in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            env.pop(stale, None)

    def _progress_ca_file(self) -> Optional[ContainerFile]:
        """The CA bundle to hand the container, injected as file content.

        The progress endpoint is the server's own URL, so on an HTTPS deployment
        fronted by a private CA the container has to verify a certificate the
        worker already trusts -- and it cannot. The benchmark image is a separate
        ``gpustack/benchmark-runner`` image whose entrypoint is the runner CLI, so
        it never runs ``gpustack-prerun.sh`` and never sees the CAs an operator
        dropped into ``/usr/local/share/ca-certificates/``. Every progress PATCH
        then dies with CERTIFICATE_VERIFY_FAILED.

        So hand it the bundle the worker itself verifies against -- private CAs
        already merged in by ``update-ca-certificates``. It travels as container
        file content rather than through a shared directory: the runtime writes
        it into the container itself (a tar upload before start on Docker, a
        ConfigMap on Kubernetes), so nothing is written anywhere the container
        can reach beforehand, nothing outlives the run, and the path inside the
        container is ours to choose rather than one that has to exist identically
        on the host.

        Returns None when there is nothing to verify (plain HTTP), when
        verification is switched off, or when the bundle cannot be read -- the
        last of which warns rather than raising, since the measurement only needs
        the model endpoint and a run must not be lost over progress telemetry.
        """
        if not self._progress_is_https:
            # Progress goes out over plain HTTP; no trust store is consulted.
            return None

        if self._progress_insecure_skip_tls_verify:
            # Verification is off; a bundle would not be consulted. The warning
            # that explains this lives next to the flag in _build_command_args,
            # which is reached under a mirrored deployment too -- this is not.
            return None

        try:
            source = resolve_ca_bundle()
            with open(source, "r", encoding="utf-8") as bundle:
                content = bundle.read()
        except OSError as e:
            logger.warning(
                f"Failed to read the CA bundle for benchmark progress reporting: "
                f"{e}. Progress updates to {self._api_url} may fail TLS "
                "verification; the benchmark itself is unaffected."
            )
            return None

        logger.debug(
            f"Handing CA bundle {source} to the benchmark container at "
            f"{PROGRESS_CA_BUNDLE_PATH}."
        )
        return ContainerFile(
            path=PROGRESS_CA_BUNDLE_PATH,
            content=content,
            mode=0o644,
        )

    def _create_workload(
        self,
        deployment_metadata: BenchmarkDeploymentMetadata,
        command: Optional[List[str]],
        command_args: List[str],
        env: Dict[str, str],
        files: Optional[List[ContainerFile]] = None,
    ):
        image = apply_registry_override_to_image(
            self._config, self._config.benchmark_image_repo, self._fallback_registry
        )
        if not image:
            raise ValueError("Failed to get image for benchmark runner workload")

        mounts = self._get_configured_mounts()

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=False,
                args=command_args,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            mounts=mounts,
            files=files or [],
        )

        logger.info(
            f"Creating benchmark container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, "
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(sanitize_args(command_args))}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=10 * 1 << 30,  # 10 GiB
            containers=[run_container],
            labels=deployment_metadata.labels,
        )
        create_workload(
            transform_workload_plan(
                self._config, workload_plan, self._fallback_registry
            )
        )

        logger.info(f"Created benchmark container workload: {deployment_metadata.name}")

    def _build_command_args(self, with_ca_cert: bool = False) -> List[str]:
        backend_kwargs = {
            "timeout": BENCHMARK_REQUEST_TIMEOUT,
            "response_handlers": {
                "chat_completions": "chat_completions_with_reasoning"
            },
        }

        command_args = [
            "benchmark",
            "run",
            "--target",
            self._model_endpoint,
            "--profile",
            "constant",
            "--rate",
            str(self._benchmark.request_rate),
            "--sample-requests",
            "0",
            "--processor",
            self._model_path,
            "--output-dir",
            f"{self._benchmark_dir}",
            "--outputs",
            f"{self._benchmark.id}.dual_json",
            "--progress-url",
            self._api_url,
            "--progress-auth",
            self._api_key,
            "--backend-kwargs",
            json.dumps(backend_kwargs),
            "--backend",
            "openai_http_error_detail",
        ]

        # Points the runner at the bundle the caller is injecting as file content.
        # Scoped to the progress channel on purpose: SSL_CERT_FILE would replace
        # the trust store for every TLS call the container makes, so a bundle
        # holding only a private CA would leave the runner unable to verify
        # anything else.
        if with_ca_cert:
            command_args.extend(["--progress-ca-cert", PROGRESS_CA_BUNDLE_PATH])

        # Only forwarded for an HTTPS progress endpoint: on plain HTTP the flag is
        # a no-op the runner would still have to recognize, and an operator who
        # left the setting on while pinning an older --benchmark-image-repo gets
        # an unknown-option failure at most where it could have mattered.
        #
        # The warning belongs here rather than with the rest of the TLS handling:
        # that runs only outside a mirrored deployment, while this append does
        # not, so an operator hitting exactly the failure described below would
        # otherwise get no log line pointing at its cause.
        if self._progress_insecure_skip_tls_verify and self._progress_is_https:
            logger.warning(
                "GPUSTACK_INSECURE_TLS is set: TLS verification is disabled for "
                f"benchmark progress reporting to {self._api_url}. Prefer "
                "importing the server's CA into /usr/local/share/ca-certificates/ "
                "on this worker. Note the benchmark image must support "
                "--progress-insecure-skip-tls-verify; an older pinned image fails "
                "to start on the unknown option."
            )
            command_args.append("--progress-insecure-skip-tls-verify")

        if find_bool_parameter(self._model_backend_parameters, ["trust-remote-code"]):
            command_args.extend(
                [
                    "--processor-args",
                    json.dumps({"trust_remote_code": True}),
                ]
            )

        if self._benchmark.dataset_name == DATASET_SHAREGPT:
            data = BENCHMARK_DATASET_SHAREGPT_PATH
            command_args.extend(["--data", data])
        elif (
            self._benchmark.dataset_name == DATASET_RANDOM
            and self._benchmark.dataset_input_tokens is not None
            and self._benchmark.dataset_output_tokens is not None
        ):
            data = f"prompt_tokens={self._benchmark.dataset_input_tokens},output_tokens={self._benchmark.dataset_output_tokens}"
            command_args.extend(["--data", data])

            if self._benchmark.dataset_seed is not None:
                command_args.extend(
                    [
                        "--random-seed",
                        f"{self._benchmark.dataset_seed}",
                    ]
                )

        if (
            self._benchmark.total_requests is not None
            and self._benchmark.total_requests > 0
        ):
            command_args.extend(
                [
                    "--max-requests",
                    f"{self._benchmark.total_requests}",
                ]
            )

        return command_args

    def _update_benchmark_state(self, id: int, **kwargs):
        resp = self._clientset.http_client.get_httpx_client().patch(
            "/benchmarks/{id}/state".format(id=id), json=kwargs
        )
        resp.raise_for_status()

    def _get_configured_mounts(self) -> List[ContainerMount]:
        """
        Get the volume mounts for the model instance.
        If runtime mirrored deployment is enabled, no mounts will be set up.

        Returns:
            A list of ContainerMount objects for the model instance.
        """
        mounts: List[ContainerMount] = []
        if (
            self._model_path
            and self._benchmark_dir
            and not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
        ):
            model_dir = os.path.dirname(self._model_path)
            mounts.extend(
                [
                    ContainerMount(
                        path=model_dir,
                    ),
                    ContainerMount(
                        path=self._benchmark_dir,
                    ),
                ]
            )
        return mounts
