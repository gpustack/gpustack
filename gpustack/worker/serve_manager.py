import asyncio
from datetime import datetime, timezone
import multiprocessing
import psutil
import requests
import setproctitle
import os
from typing import Dict, Optional, Set
import logging

from gpustack_runtime.deployer import (
    get_workload,
    WorkloadStatusStateEnum,
    delete_workload,
    UnsupportedError,
    OperationError,
)

from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.utils import network, platform
from gpustack.utils.attrs import set_attr
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.worker.backends.llama_box import LlamaBoxServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.custom import CustomServer
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.inference_backend_manager import InferenceBackendManager

logger = logging.getLogger(__name__)

_SERVER_CLASS_MAPPING = {
    BackendEnum.LLAMA_BOX: LlamaBoxServer,
    BackendEnum.VLLM: VLLMServer,
    BackendEnum.VOX_BOX: VoxBoxServer,
    BackendEnum.ASCEND_MINDIE: AscendMindIEServer,
}


class ServeManager:
    def __init__(
        self,
        worker_id: int,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._worker_id = worker_id
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._serving_model_instances: Dict[int, multiprocessing.Process] = {}
        self._serving_model_instance_ports: Dict[int, Set[int]] = {}
        self._starting_model_instances: Dict[int, ModelInstance] = {}
        self._error_model_instances: Dict[int, ModelInstance] = {}
        self._model_cache_by_instance: Dict[int, Model] = {}
        # Track consecutive health endpoint failures for RUNNING instances
        self._ready_failures: Dict[int, int] = {}
        self._clientset = clientset
        self._cache_dir = cfg.cache_dir

        self.inference_backend_manager = InferenceBackendManager(clientset)

        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_model_instances(self):
        while True:
            try:
                logger.info("Started watching model instances.")
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed watching model instances: {e}")
                await asyncio.sleep(5)

    async def monitor_error_instances(self):
        """Periodically checks cached ERROR state instances and attempts to restart them."""
        while True:
            try:
                logger.trace(
                    f"Monitoring error instances, instances: {self._error_model_instances.keys()}"
                )

                for mi_id, mi in list(self._error_model_instances.items()):
                    self._restart_error_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error while monitoring instances: {e}")
                await asyncio.sleep(5)

    def _handle_model_instance_event(self, event: Event):  # noqa: C901
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received model instance event: {event.type} {mi.name} {mi.state}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            # Return if all subordinate workers aren't running.
            if (
                mi.distributed_servers
                and mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.RUN_FIRST
                and mi.distributed_servers.subordinate_workers
            ):
                ready = all(
                    sw.state == ModelInstanceStateEnum.RUNNING
                    for sw in mi.distributed_servers.subordinate_workers
                )
                if not ready:
                    logger.info(
                        f"Model instance {mi.name} waits for all subordinate workers to be ready."
                    )
                    return
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return
            # Return if it's a delegated distribution,
            # which means the main worker is responsible for serving.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.DELEGATED
            ):
                return
            # Return if it isn't the member of the distribution serving.
            joined = any(
                sw.worker_id == self._worker_id
                for sw in mi.distributed_servers.subordinate_workers or []
            )
            if not joined:
                return
            # Return if the main worker isn't initialized.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                and mi.state
                not in [
                    ModelInstanceStateEnum.STARTING,
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]
            ):
                logger.info(
                    f"Model instance {mi.name} waits for main worker {mi.worker_ip} to be initialized."
                )
                return
            # FIXME: This is a temporary solution to prevent the main worker from being unable to start due to phantom reads.
            #        We confirm whether the operation should be performed by checking the state of the earlier subordinate worker.
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id == self._worker_id:
                    break
                if sw.state not in [
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]:
                    logger.info(
                        f"Model instance {mi.name} waits for previous subordinate worker {sw.worker_ip} to be ready."
                    )
                    return

        if mi.state == ModelInstanceStateEnum.ERROR and event.type == EventType.DELETED:
            self._error_model_instances.pop(mi.id, None)
            return
        elif mi.state == ModelInstanceStateEnum.ERROR:
            m = self._get_model_with_cache(mi)
            if m.restart_on_error:
                self._error_model_instances[mi.id] = mi
            return
        elif mi.id in self._serving_model_instances and event.type == EventType.DELETED:
            self._stop_model_instance(mi)
        elif (
            mi.id in self._serving_model_instances
            and mi.state == ModelInstanceStateEnum.SCHEDULED
        ):
            # In case when the worker is offline and reconnected, the model instance state
            # is out of sync with existing serving process. Restart it.
            self._restart_serve_process(mi)
        elif (
            event.type in {EventType.CREATED, EventType.UPDATED}
        ) and not self._serving_model_instances.get(mi.id):
            self._start_serve_process(mi)

    def _start_serve_process(self, mi: ModelInstance):  # noqa: C901
        is_main_worker = mi.worker_id == self._worker_id

        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"
        if os.path.exists(log_file_path) and platform.system() != "windows":
            # TODO Windows does not support os.remove() on open files.
            # Investigate file occupation issue.
            os.remove(log_file_path)

        sw_pos: Optional[int] = None
        sw: Optional[ModelInstanceSubordinateWorker] = None
        if not is_main_worker:
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                    if sw.worker_id == self._worker_id
                ),
            )
            sw = mi.distributed_servers.subordinate_workers[sw_pos]

        try:
            model = self._get_model_with_cache(mi)
            backend = get_backend(model)

            # Assign port.
            if not mi.port:
                if self._serving_model_instance_ports:
                    unavailable_ports = set.union(
                        *self._serving_model_instance_ports.values()
                    )
                else:
                    unavailable_ports = set()
                mi.port = network.get_free_port(
                    port_range=self._config.service_port_range,
                    unavailable_ports=unavailable_ports,
                )
                mi.ports = [mi.port]
                if (
                    mi.distributed_servers
                    and mi.distributed_servers.subordinate_workers
                ):
                    if backend == BackendEnum.ASCEND_MINDIE:
                        # Get port for subordinate worker watching.
                        unavailable_ports.add(mi.port)
                        connecting_port = network.get_free_port(
                            port_range=self._config.service_port_range,
                            unavailable_ports=unavailable_ports,
                        )
                        mi.ports.append(connecting_port)

            logger.info(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f'on port {mi.ports if mi.ports else [mi.port]}'}"
            )

            process = multiprocessing.Process(
                target=ServeManager.serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    self._worker_id,
                    self.inference_backend_manager.get_backend_by_name(backend),
                ),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process
            self._serving_model_instance_ports[mi.id] = set(mi.ports)
            self._starting_model_instances[mi.id] = mi

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                    "pid": process.pid,
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.INITIALIZING
                # For Ascend MindIE, the state is set to RUNNING directly,
                if backend == BackendEnum.ASCEND_MINDIE:
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name} "
                f"{'' if not is_main_worker else f'on port {mi.ports if mi.ports else [mi.port]}'}, "
                f"pid {process.pid}"
            )

        except Exception as e:
            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.ERROR
                sw.state_message = f"Failed to start model instance: {e}"
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

    def _restart_serve_process(self, mi: ModelInstance):
        logger.debug(f"Restart serving model instance {mi.name}")
        self._stop_model_instance(mi)
        self._start_serve_process(mi)

    @staticmethod
    def serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
    ):

        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.server_url,
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
                    server_cls(
                        clientset,
                        mi,
                        cfg,
                        worker_id,
                        inference_backend,
                    ).start()
                except Exception as e:
                    logger.exception(f"Failed to start model instance {mi.name}")
                    raise e

    def _update_model_instance(self, id: int, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            set_attr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _handle_running_ready_failure(
        self, mi: ModelInstance, failures_threshold: int = 3
    ) -> bool:
        """
        Handle health endpoint failures for RUNNING state instances.
        Increments the failure counter and updates the model instance state to ERROR
        when the consecutive failure count reaches the threshold.
        Returns True if threshold is reached and state updated to ERROR, False otherwise.
        """
        if mi.state != ModelInstanceStateEnum.RUNNING:
            return False
        failures = self._ready_failures.get(mi.id, 0) + 1
        self._ready_failures[mi.id] = failures
        if failures >= failures_threshold:
            patch_dict = {
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": f"Health check failed {failures} times; endpoint not ready.",
            }
            try:
                self._update_model_instance(mi.id, **patch_dict)
            except NotFoundException:
                pass
            # Reset failure counter after transitioning to ERROR
            self._ready_failures.pop(mi.id, None)
            return True
        return False

    def _get_model_with_cache(self, mi: ModelInstance) -> Model:
        """Get model from cache or fetch from clientset."""
        if mi.id in self._model_cache_by_instance:
            return self._model_cache_by_instance[mi.id]

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _stop_model_instance(self, mi: ModelInstance):
        instance_name = mi.name or f"ModelInstance-{mi.id}"
        if mi.id not in self._serving_model_instances:
            logger.warning(f"Model instance {instance_name} is not running. Skipping.")
            return
        else:
            pid = self._serving_model_instances[mi.id].pid
            try:
                terminate_process_tree(pid)
                logger.info(f"Stopped model instance {instance_name} with pid {pid}.")
            except psutil.NoSuchProcess:
                logger.warning(
                    f"Model instance {instance_name} with pid {pid} is already stopped."
                )
                pass
            except Exception as e:
                logger.error(f"Failed to stop model instance: {pid}: {e}")

            # For CustomServer backend, also delete the Docker container
            self._cleanup_inference_server_container(mi)

            self._post_stop_model_instance(mi)

    def _cleanup_inference_server_container(self, mi: ModelInstance):
        """
        Clean up Docker container for CustomServer backend.

        This method checks if the model instance uses CustomServer backend
        and calls the container deletion functionality.
        """
        try:
            # Get the model to check backend type
            logger.info(f"Cleaning up container for {mi.name}")

            delete_workload(mi.name)

        except OperationError as oe:
            logger.warning(
                f"Operation error during container cleanup for {mi.name}: {oe}"
            )
        except UnsupportedError as ue:
            logger.warning(
                f"Unsupported error during container cleanup for {mi.name}: {ue}"
            )
        except Exception as e:
            logger.error(f"Error during container cleanup for {mi.name}: {e}")
            # Don't raise the exception to avoid affecting the normal stop process

    def _restart_error_instance(self, mi: ModelInstance):
        """Attempts to restart a model instance that is in error state with exponential backoff."""
        if mi.id in self._serving_model_instances:
            logger.warning(
                f"Model instance {mi.name} is already running, skipping restart."
            )
            return

        restart_count = mi.restart_count or 0
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(
            10 * (2 ** (restart_count - 1)), 300
        )  # Exponential backoff, max 5 minutes
        if last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} (attempt {restart_count + 1}) after {delay} seconds delay."
        )

        self._update_model_instance(
            mi.id,
            restart_count=restart_count + 1,
            last_restart_time=current_time,
            state=ModelInstanceStateEnum.SCHEDULED,
            state_message="",
        )

        self._error_model_instances.pop(mi.id, None)

    def health_check_serving_instances(self):  # noqa: C901
        for id, process in list(self._serving_model_instances.items()):
            # Skip if the process is alive and not in starting instances.
            if process.is_alive() and id not in self._starting_model_instances:
                continue
            try:
                mi = self._clientset.model_instances.get(id=id)
            except NotFoundException as e:
                logger.warning(
                    f"Model instance {id} not found because of {str(e)}, stopping serving process."
                )
                self._stop_model_instance(ModelInstance(id=id))
                continue

            is_main_worker = mi.worker_id == self._worker_id

            workload_status = get_workload(mi.name)
            workload_alive = True
            if not workload_status or workload_status.state in [
                WorkloadStatusStateEnum.INACTIVE,
                WorkloadStatusStateEnum.FAILED,
                WorkloadStatusStateEnum.PENDING,
                WorkloadStatusStateEnum.UNKNOWN,
            ]:
                workload_alive = False

            # Monitor inference server process exit
            if not process.is_alive() and not workload_alive:
                try:
                    if mi.state != ModelInstanceStateEnum.ERROR:
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": "Inference server exited.",
                            }
                        # Get patch dict for subordinate worker.
                        else:
                            sw_pos = next(
                                (
                                    i
                                    for i, sw in enumerate(
                                        mi.distributed_servers.subordinate_workers
                                    )
                                    if sw.worker_id == self._worker_id
                                ),
                            )
                            sw = mi.distributed_servers.subordinate_workers[sw_pos]
                            sw.state = ModelInstanceStateEnum.ERROR
                            sw.state_message = "Inference server exited."
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(mi.id, **patch_dict)
                except NotFoundException:
                    pass
                # Post process stop model instance.
                self._post_stop_model_instance(mi)
                return

            # Otherwise, check if the process is ready to serve.
            model = self._get_model_with_cache(mi)
            backend = get_backend(model)
            inference_backend = self.inference_backend_manager.get_backend_by_name(
                backend
            )
            try:
                # Get patch dict for main worker.
                if is_main_worker:
                    sw_error_msg = None
                    if (
                        mi.distributed_servers
                        and mi.distributed_servers.subordinate_workers
                    ):
                        for sw in mi.distributed_servers.subordinate_workers:
                            if sw.state == ModelInstanceStateEnum.ERROR:
                                sw_error_msg = f"Distributed serving error in subordinate worker {sw.worker_ip}: {sw.state_message}."
                                break
                    # If there is no error message from subordinate workers,
                    # check the main worker's health.
                    if not sw_error_msg:
                        ready_be_requested = is_ready(backend, mi, inference_backend)
                        if not ready_be_requested:
                            self._handle_running_ready_failure(mi)
                            continue
                        self._ready_failures.pop(mi.id, None)
                        if mi.state == ModelInstanceStateEnum.RUNNING:
                            continue
                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "state_message": "",
                        }
                    # Otherwise, update the main worker state to ERROR.
                    else:
                        patch_dict = {
                            "state": ModelInstanceStateEnum.ERROR,
                            "state_message": sw_error_msg,
                        }
                # Get patch dict for subordinate worker.
                else:
                    # Skip health check for Ascend MindIE subordinate workers
                    if backend == BackendEnum.ASCEND_MINDIE:
                        # Remove from starting instances directly.
                        self._starting_model_instances.pop(mi.id, None)
                        continue
                    # Otherwise, update subordinate worker state to RUNNING.
                    sw_pos = next(
                        (
                            i
                            for i, sw in enumerate(
                                mi.distributed_servers.subordinate_workers
                            )
                            if sw.worker_id == self._worker_id
                        ),
                    )
                    sw = mi.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    sw.state = ModelInstanceStateEnum.RUNNING
                    sw.state_message = ""
                    patch_dict = {
                        f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                    }
                # Update model instance.
                self._update_model_instance(mi.id, **patch_dict)
                # Remove from starting instances if it was started.
                self._starting_model_instances.pop(mi.id, None)
            except NotFoundException:
                pass

    def _post_stop_model_instance(self, mi: ModelInstance):
        """
        Post process after stopping a model instance.
          - Remove from serving model instances.
          - Remove ports from serving model instance ports.
          - Remove from starting model instances.
          - Remove from model cache by instance.
        """

        self._serving_model_instances.pop(mi.id, None)
        self._serving_model_instance_ports.pop(mi.id, None)
        self._starting_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)


def is_ready(
    backend: str, mi: ModelInstance, inference_backend: Optional[InferenceBackend]
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.
    """

    try:
        hostname = "127.0.0.1"
        if backend == BackendEnum.ASCEND_MINDIE:
            # Connectivity to the loopback address does not work for Ascend MindIE.
            # Use worker IP instead.
            hostname = mi.worker_ip

        # Check /v1/models by default if dedicated health check endpoint is not available.
        # This is served by all backends (llama-box, vox-box, vllm, mindIE)
        health_check_domain = f"http://{hostname}:{mi.port}"
        health_check_url = "/v1/models"
        if backend == BackendEnum.LLAMA_BOX:
            # For llama-box, use /health to avoid printing error logs.
            health_check_url = "/health"
        elif not any(b.value == backend for b in BackendEnum):
            if inference_backend and inference_backend.health_check_path:
                health_check_url = inference_backend.health_check_path

        response = requests.get(health_check_domain + health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception as e:
        logger.error(f"Error checking model instance health: {e}")
        pass
    return False
