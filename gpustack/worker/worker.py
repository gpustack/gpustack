import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import logging
from typing import Optional, Tuple
import json

import aiohttp
from fastapi import FastAPI
import setproctitle
import tenacity
import uvicorn
from urllib.parse import urlparse
from starlette.middleware.base import BaseHTTPMiddleware
from gpustack_runtime.deployer.k8s.deviceplugin import (
    serve_async as kdp_serve_async,
    get_resource_injection_policy,
)

from gpustack.api import exceptions
from gpustack.config.config import (
    Config,
    WorkerConfig,
)
from gpustack.schemas.config import (
    GatewayModeEnum,
    PredefinedConfigNoDefaults,
)
from gpustack import envs
from gpustack.routes import config as route_config, debug, probes
from gpustack.routes.worker import logs, proxy, filesystem
from gpustack.routes.token import worker_auth
from gpustack.server import catalog
from gpustack.utils.network import (
    get_first_non_loopback_ip,
    get_ifname_by_ip_hostname,
)
from gpustack.client import ClientSet
from gpustack.logging import setup_logging
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.utils.system_check import check_glibc_version
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.benchmark_manager import BenchmarkManager
from gpustack.worker.inference_backend_manager import InferenceBackendManager
from gpustack.worker.model_file_manager import ModelFileManager
from gpustack.worker.runtime_metrics_aggregator import RuntimeMetricsAggregator
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.exporter import MetricExporter
from gpustack.worker.tools_manager import ToolsManager
from gpustack.worker.worker_manager import WorkerManager
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.config.registration import read_worker_token
from gpustack.config import registration
from gpustack.gateway import init_async_k8s_config
from gpustack.client.generated_http_client import default_versioned_prefix
from gpustack.worker.workload_cleaner import WorkloadCleaner
from gpustack.utils.uuid import get_worker_name, get_legacy_uuid

logger = logging.getLogger(__name__)


class Worker:
    _default_config: PredefinedConfigNoDefaults
    _clientset: ClientSet
    _register_clientset: ClientSet
    _status_collector: WorkerStatusCollector
    _worker_manager: WorkerManager
    _serve_manager: ServeManager
    _benchmark_manager: BenchmarkManager
    _workload_cleaner: WorkloadCleaner
    _config: Config
    _worker_ip: Optional[str] = None
    _worker_ifname: Optional[str] = None
    _worker_id: Optional[int] = None
    _worker_name: Optional[str] = None
    _worker_uuid: Optional[str] = None
    _cluster_id: Optional[int] = None

    def worker_ip(self) -> str:
        return self._config.worker_ip or self._worker_ip

    def worker_ifname(self) -> str:
        return self._config.worker_ifname or self._worker_ifname

    def worker_name(self) -> Optional[str]:
        return (
            self._config.worker_name
            or self._worker_name
            or get_worker_name(self._config.data_dir)
        )

    def worker_uuid(self) -> str:
        return self._worker_uuid or get_legacy_uuid(self._config.data_dir) or ""

    def worker_id(self) -> int:
        return self._worker_id

    def clientset(self) -> ClientSet:
        return self._clientset

    def cluster_id(self) -> Optional[int]:
        return self._cluster_id

    def __init__(self, cfg: Config):
        self._config = cfg
        self._is_embedded = cfg.server_role() == Config.ServerRole.BOTH
        self._log_dir = cfg.log_dir
        self._address = "0.0.0.0"
        self._exporter_enabled = not cfg.disable_worker_metrics
        self._async_tasks = []
        self._worker_ip, self._worker_ifname = self._detect_worker_ip_and_ifname()

        self._runtime_metrics_cache = defaultdict()

        self._status_collector = WorkerStatusCollector(
            cfg=cfg,
            worker_ip_getter=self.worker_ip,
            worker_ifname_getter=self.worker_ifname,
            worker_id_getter=self.worker_id,
            worker_uuid_getter=self.worker_uuid,
        )

        self._worker_manager = WorkerManager(
            cfg=cfg,
            is_embedded=self._is_embedded,
            collector=self._status_collector,
        )

        self._exporter = MetricExporter(
            cfg=cfg,
            collector=self._status_collector,
            worker_ip_getter=self.worker_ip,
            worker_id_getter=self.worker_id,
            worker_name_getter=self.worker_name,
            clientset_getter=self.clientset,
            cache=self._runtime_metrics_cache,
        )

        self._serve_manager = ServeManager(
            worker_id_getter=self.worker_id,
            clientset_getter=self.clientset,
            cfg=self._config,
        )

        self._benchmark_manager = BenchmarkManager(
            worker_id_getter=self.worker_id,
            clientset_getter=self.clientset,
            cfg=self._config,
        )

        self._workload_cleaner = WorkloadCleaner(
            worker_id_getter=self.worker_id,
            clientset_getter=self.clientset,
        )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_fixed(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying to get worker ID (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
        ),
    )
    def _register(self):
        self._clientset, self._default_config = (
            self._worker_manager.register_with_server()
        )
        # Worker ID is available after the worker registration.
        worker_list = self._clientset.workers.list(
            params={"me": 'true'},
        )
        name = self.worker_name() or "<not specified>"
        if len(worker_list.items) != 1:
            raise Exception(f"Worker {name} not registered.")
        self._worker_id = worker_list.items[0].id
        self._cluster_id = worker_list.items[0].cluster_id
        self._worker_name = worker_list.items[0].name
        self._worker_uuid = worker_list.items[0].worker_uuid

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    def start(self):
        setup_logging(self._config.debug)

        if self._is_embedded:
            setproctitle.setproctitle("gpustack_worker")

        check_glibc_version()
        init_async_k8s_config(cfg=self._config)

        tools_manager = ToolsManager(
            tools_download_base_url=self._config.tools_download_base_url,
            pipx_path=self._config.pipx_path,
            data_dir=self._config.data_dir,
            bin_dir=self._config.bin_dir,
        )
        tools_manager.prepare_tools()
        catalog.prepare_chat_templates(self._config.data_dir)

        try:
            asyncio.run(self.start_async())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error(f"Error serving worker APIs: {e}")
        finally:
            logger.info("Worker has shut down.")

    def log_worker_config(self):
        fields = {
            k: v
            for k, v in self._config.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                exclude={'token'},
            ).items()
            if k in WorkerConfig.model_fields
        }
        hf_token = fields.get("huggingface_token", None)
        if hf_token is not None:
            fields["huggingface_token"] = "*" * len(hf_token)
        logger.info(
            "Worker starting with config: %s",
            json.dumps(fields, indent=2, ensure_ascii=False),
        )

    async def start_async(self):
        """
        Start the worker.
        """

        logger.info("Starting GPUStack worker.")

        add_signal_handlers_in_loop()

        self._register()
        self._config.reload_worker_config(self._default_config)
        self.log_worker_config()

        self._serve_manager.reconnect_existing_model_instances()

        if self._exporter_enabled:
            # Start the runtime metrics cacher.
            _runtime_metrics_aggregator = RuntimeMetricsAggregator(
                cache=self._runtime_metrics_cache,
                worker_id_getter=self.worker_id,
                clientset=self._clientset,
            )
            run_periodically_in_thread(_runtime_metrics_aggregator.aggregate, 3, 30)

            # Start the metric exporter with retry.
            run_periodically_in_thread(self._exporter.start, 15)

        # Monitor the ip change, if not fixed.
        if self._config.worker_ip is None or self._config.worker_ifname is None:
            # Check worker ip change every 15 seconds.
            run_periodically_in_thread(self._check_worker_ip_change, 15)

        # Send heartbeat to the server every WORKER_HEARTBEAT_INTERVAL seconds.
        run_periodically_in_thread(self._heartbeat, envs.WORKER_HEARTBEAT_INTERVAL)

        # Report the worker node status to the server every WORKER_STATUS_SYNC_INTERVAL seconds.
        run_periodically_in_thread(
            self._worker_manager.sync_worker_status, envs.WORKER_STATUS_SYNC_INTERVAL
        )

        # Start the worker server to expose APIs.
        self._create_async_task(self._serve_apis())

        inference_backend_manager = InferenceBackendManager(self._clientset)
        # Start InferenceBackend listener to cache backend data
        self._create_async_task(inference_backend_manager.start_listener())
        # Trigger cache refresh
        registration.determine_default_registry(
            self._config.system_default_container_registry
        )

        self._serve_manager._inference_backend_manager = inference_backend_manager
        run_periodically_in_thread(
            self._serve_manager.sync_model_instances_state,
            envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL,
        )
        run_periodically_in_thread(
            self._workload_cleaner.cleanup_orphan_workloads, 120, 15
        )
        run_periodically_in_thread(self._benchmark_manager.sync_benchmark_state, 3, 15)

        self._create_async_task(self._serve_manager.watch_models())
        self._create_async_task(self._serve_manager.watch_model_instances_event())
        self._create_async_task(self._serve_manager.watch_model_instances())
        self._create_async_task(self._benchmark_manager.watch_benchmarks_event())

        model_file_manager = ModelFileManager(
            worker_id=self._worker_id, clientset=self._clientset, cfg=self._config
        )
        self._create_async_task(model_file_manager.watch_model_files())

        # Start Kubernetes Device Plugin server if allowed.
        if get_resource_injection_policy() == "kdp":
            self._create_async_task(kdp_serve_async(stop_event=asyncio.Event()))

        # wait for a while to let other tasks start
        await asyncio.sleep(0.5)
        logger.info("GPUStack worker startup completed.")

        await asyncio.gather(*self._async_tasks)

    async def _serve_apis(self):
        """
        Start the worker server to expose APIs.
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            connector = aiohttp.TCPConnector(
                limit=envs.TCP_CONNECTOR_LIMIT,
                force_close=True,
            )
            app.state.http_client = aiohttp.ClientSession(
                connector=connector, trust_env=True
            )
            app.state.http_client_no_proxy = aiohttp.ClientSession(connector=connector)
            yield
            await app.state.http_client.close()
            await app.state.http_client_no_proxy.close()

        app = FastAPI(
            title="GPUStack Worker",
            response_model_exclude_unset=True,
            lifespan=lifespan,
        )
        app.state.config = self._config
        app.state.token = read_worker_token(self._config.data_dir)
        app.state.worker_ip_getter = self.worker_ip
        app.state.instance_port_by_model_name = (
            self._serve_manager.get_instance_ports_by_model_name
        )
        app.add_middleware(BaseHTTPMiddleware, dispatch=proxy.set_port_from_model_name)
        app.include_router(route_config.router, prefix=default_versioned_prefix)
        app.include_router(debug.router, prefix="/debug")
        app.include_router(probes.router)
        app.include_router(logs.router)
        app.include_router(proxy.router)
        app.include_router(filesystem.router)
        app.add_api_route(
            path="/token-auth",
            endpoint=worker_auth,
            methods=["GET"],
        )
        exceptions.register_handlers(app)

        config = uvicorn.Config(
            app,
            host=self._address,
            port=self._config.get_api_port(self._is_embedded),
            access_log=False,
            log_level="error",
        )

        setup_logging()
        worker_api_message = f"Serving worker APIs on {config.host}:{config.port}."
        if not self._is_embedded:
            logger.debug(worker_api_message)
            logger.info(f"Worker gateway mode: {self._config.gateway_mode.value}.")
            if self._config.gateway_mode == GatewayModeEnum.embedded:
                logger.info(f"Serving worker on {self._config.get_gateway_port()}.")
        else:
            logger.info(worker_api_message)
        server = uvicorn.Server(config)

        await server.serve()

    def _detect_worker_ip_and_ifname(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect the worker IP and ifname.
        """
        static_worker_ip = self._config.worker_ip
        static_worker_ifname = self._config.worker_ifname
        detected_ifname = None
        detected_ip = None
        if static_worker_ip is not None and static_worker_ifname is not None:
            pass

        elif static_worker_ip is not None:
            # if ip is set, use it to detect ifname
            detected_ifname = get_ifname_by_ip_hostname(static_worker_ip)

        elif static_worker_ifname is not None:
            # if ifname is set, used it to detect ip
            detected_ip = get_first_non_loopback_ip(
                expected_ifname=static_worker_ifname
            )

        else:
            # detect both ip and ifname

            # detect_ifname may be None if the hostname resolves to a loopback address.
            # This typically happens when the worker and server run on the same host, or for embedded workers.
            detected_ifname = get_ifname_by_ip_hostname(
                urlparse(self._config.get_server_url()).hostname
            )

            try:
                # if the expected_ifname is none, it will scan all interfaces
                detected_ip = get_first_non_loopback_ip(expected_ifname=detected_ifname)
            except Exception:
                logger.warning(
                    f"Failed to detect worker IP from interface {detected_ifname}. Using first non-loopback IP."
                )
                # avoid edge case where the detected_ifname has no valid IPv4 address
                detected_ip = get_first_non_loopback_ip()

            if detected_ifname is None:
                detected_ifname = get_ifname_by_ip_hostname(detected_ip)
        return detected_ip, detected_ifname

    def _check_worker_ip_change(self):
        """
        Detect if the worker IP has changed. If so, delete legacy model
        instances so they can be recreated with the new worker IP.
        """

        new_ip, new_ifname = self._detect_worker_ip_and_ifname()
        old_ip, old_ifname = self._worker_ip, self._worker_ifname
        if new_ip == old_ip and new_ifname == old_ifname:
            return

        logger.info(
            f"Worker IP changed from {old_ip}({old_ifname}) to {new_ip}{new_ifname}"
        )
        self._worker_ip = new_ip
        self._worker_ifname = new_ifname
        self._worker_manager.sync_worker_status()

        for instance in self._clientset.model_instances.list(
            params={"worker_id": str(self._worker_id)}
        ).items:
            self._clientset.model_instances.delete(instance.id)

    def _heartbeat(self):
        """
        Send heartbeat to the server to indicate the worker is alive.
        """
        if self._worker_id is None:
            logger.debug("Worker ID is not set, skipping heartbeat.")
            return
        try:
            resp = self._clientset.http_client.get_httpx_client().post(
                "/worker-heartbeat", json={}
            )
            if resp.status_code != 204:
                logger.error(
                    f"Failed to send heartbeat to server, status code: {resp.status_code}"
                )
        except Exception as e:
            logger.error(f"Failed to send heartbeat to server: {e}")
