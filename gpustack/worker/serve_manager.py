import asyncio
import contextlib
from datetime import datetime, timezone
import multiprocessing

import requests
import setproctitle
import os
from typing import Dict, Optional, Set, List
import logging

from gpustack_runtime.deployer import (
    get_workload,
    WorkloadStatusStateEnum,
    delete_workload,
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
    _worker_id: int
    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """
    _serve_log_dir: str
    """
    The directory to store logs of serving model instances(in subprocess).
    """
    _clientset: ClientSet
    """
    The clientset to access the API server.
    """
    _inference_backend_manager: InferenceBackendManager
    """
    The inference backend manager.
    """
    _provisioning_processes: Dict[int, multiprocessing.Process] = {}
    """
    The mapping of model instance ID to provisioning (sub)process.
    When the (sub)process is alive, the model instance is provisioning.
    If the (sub)process exited, the model instance is either running or failed.
    """
    _assigned_ports: Dict[int, Set[int]] = {}
    """
    The mapping of model instance ID to assigned ports.
    Used to avoid port conflicts when assigning ports to new model instances.
    """
    _error_model_instances: Dict[int, ModelInstance] = {}
    """
    The mapping of model instance ID to error model instances.
    Used to restart error model instances.
    """
    _model_cache_by_instance: Dict[int, Model] = {}
    """
    The cache of models by model instance ID.
    Used to avoid redundant API calls to get model information.
    """

    def __init__(
        self,
        worker_id: int,
        clientset: ClientSet,
        cfg: Config,
        inference_backend_manager: InferenceBackendManager,
    ):
        self._worker_id = worker_id
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset = clientset
        self._inference_backend_manager = inference_backend_manager

        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        logger.debug("Watching model instances event.")

        while True:
            try:
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching model instances: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances(self):
        """
        Loop to post process model instances, for example, restarting error instances.

        """

        logger.debug("Watching model instances.")

        while True:
            try:
                for mi in list(self._error_model_instances.values()):
                    self._restart_error_model_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error restarting model instances: {e}")
                await asyncio.sleep(5)

    def sync_model_instances_state(self):  # noqa: C901
        """
        Synchronize model instances' state.

        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, inactive or failed, update the model instance state to ERROR.
        - If everything is fine, update the model instance state to RUNNING.
        """

        # Get all model instances assigned to this worker.
        #
        # FIXME(thxCode): This may cause performance issues when there are many model instances in the system.
        #                 A mechanism is needed to improve efficiency here.
        model_instances_page = self._clientset.model_instances.list()
        if not model_instances_page.items:
            return
        model_instances: List[ModelInstance] = []
        for model_instance in model_instances_page.items:
            if model_instance.worker_id == self._worker_id:
                model_instances.append(model_instance)
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            ):
                for sw in model_instance.distributed_servers.subordinate_workers:
                    if sw.worker_id == self._worker_id:
                        model_instances.append(model_instance)
                        break

        for model_instance in model_instances:
            # Skip if the provision process has not exited yet.
            if self._is_provisioning(model_instance):
                logger.trace(
                    f"Model instance {model_instance.name} is provisioning. Skipping sync."
                )
                continue

            # Skip if the workload is still launching.
            workload = get_workload(model_instance.name)
            if workload and workload.state in [
                WorkloadStatusStateEnum.PENDING,
                WorkloadStatusStateEnum.INITIALIZING,
            ]:
                logger.trace(
                    f"Model instance {model_instance.name} workload is still launching. Skipping sync."
                )
                continue

            is_main_worker = model_instance.worker_id == self._worker_id

            # Update model instance state to ERROR if the workload is not existed, unhealthy, inactive or failed.
            if not workload or workload.state in [
                WorkloadStatusStateEnum.UNKNOWN,  # Rare, but possible, for example, leaving pause container.
                WorkloadStatusStateEnum.UNHEALTHY,
                WorkloadStatusStateEnum.INACTIVE,
                WorkloadStatusStateEnum.FAILED,
            ]:
                # NB(thxCode): Since the `sync_model_instances_state` and `watch_model_instances_event` are in different loops,
                # subordinate workers haven't had time to create the workload yet even though the model instance's state is expected.
                # So we skip if the subordinate worker didn't have workload yet.
                #
                # FIXME(thxCode): Another problem caused by skipping this check is that if we actively delete the workload on the subordinate worker,
                #                 we may not be able to correct the state of the subordinate worker.
                if not is_main_worker and not workload:
                    return
                # Only if not in ERROR state yet.
                if model_instance.state != ModelInstanceStateEnum.ERROR:
                    with contextlib.suppress(NotFoundException):
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": "Inference server exited or unhealthy.",
                            }
                        # Get patch dict for subordinate worker.
                        else:
                            sw_pos = next(
                                (
                                    i
                                    for i, sw in enumerate(
                                        model_instance.distributed_servers.subordinate_workers
                                    )
                                    if sw.worker_id == self._worker_id
                                ),
                            )
                            sw = model_instance.distributed_servers.subordinate_workers[
                                sw_pos
                            ]
                            sw.state = ModelInstanceStateEnum.ERROR
                            sw.state_message = "Inference server exited or unhealthy."
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(model_instance.id, **patch_dict)
                return

            # Otherwise, update model instance state to RUNNING if everything is fine.
            model = self._get_model(model_instance)
            backend = get_backend(model)
            inference_backend = self._inference_backend_manager.get_backend_by_name(
                backend
            )
            with contextlib.suppress(NotFoundException):
                # Get patch dict for main worker.
                if is_main_worker:
                    sw_error_msg = None
                    if (
                        model_instance.distributed_servers
                        and model_instance.distributed_servers.subordinate_workers
                    ):
                        for (
                            sw
                        ) in model_instance.distributed_servers.subordinate_workers:
                            if sw.state == ModelInstanceStateEnum.ERROR:
                                sw_error_msg = f"Distributed serving error in subordinate worker {sw.worker_ip}: {sw.state_message}."
                                break
                    # If there is no error message from subordinate workers,
                    # check whether the main worker is healthy.
                    if not sw_error_msg:
                        if not is_ready(backend, model_instance, inference_backend):
                            continue
                        if model_instance.state == ModelInstanceStateEnum.RUNNING:
                            continue
                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "restart_count": 0,  # Reset restart count on successful run.
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
                    # For initialize later mode, the state is set to RUNNING directly,
                    # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                    if (
                        model_instance.distributed_servers.mode
                        == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    ):
                        continue
                    # Otherwise, update subordinate worker state to RUNNING.
                    sw_pos = next(
                        (
                            i
                            for i, sw in enumerate(
                                model_instance.distributed_servers.subordinate_workers
                            )
                            if sw.worker_id == self._worker_id
                        ),
                    )
                    sw = model_instance.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    sw.state = ModelInstanceStateEnum.RUNNING
                    sw.state_message = ""
                    patch_dict = {
                        f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                    }
                # Update model instance.
                self._update_model_instance(model_instance.id, **patch_dict)

    @staticmethod
    def _serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
    ):
        """
        Serve model instance in a subprocess.
        Exits the subprocess when serving ends.

        Args:
            mi: The model instance to serve.
            backend: The backend of the model instance.
            client_headers: The headers for the clientset.
            log_file_path: The path to the log file.
            cfg: The configuration.
            worker_id: The ID of the worker.
            inference_backend: The inference backend configuration.
        """

        setproctitle.setproctitle(f"gpustack_model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.server_url,
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
                    server_ins = server_cls(
                        clientset,
                        mi,
                        cfg,
                        worker_id,
                        inference_backend,
                    )
                    logger.info(f"Provisioning model instance {mi.name}")
                    server_ins.start()
                    logger.info(f"Finished provisioning model instance {mi.name}")
                except Exception as e:
                    logger.exception(
                        f"Error provisioning model instance {mi.name}: {e}"
                    )
                    raise e

    def _handle_model_instance_event(self, event: Event):  # noqa: C901
        """
        Handle model instance events.

        Args:
            event: The model instance event to handle.

        """
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
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
                and (
                    mi.state
                    not in [
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                        ModelInstanceStateEnum.ERROR,
                    ]
                )
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

        if event.type == EventType.DELETED:
            self._stop_model_instance(mi)
            logger.trace(f"DELETED event: stopped deleted model instance {mi.name}.")
            return

        if event.type == EventType.UPDATED:
            # Caching matched ERROR instances for restart handling.
            if mi.state == ModelInstanceStateEnum.ERROR:
                model = self._get_model(mi)
                if model.restart_on_error:
                    self._error_model_instances[mi.id] = mi
                    logger.trace(
                        f"UPDATED event: cached error model instance {mi.name} for restart."
                    )
                return

            # Restart if scheduled.
            if mi.state == ModelInstanceStateEnum.SCHEDULED:
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            # Start on subordinate worker if not started yet.
            if not is_main_worker:
                workload = get_workload(mi.name)
                if not workload:
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: started model instance {mi.name} on subordinate worker."
                    )

            return

        if event.type == EventType.CREATED:
            self._start_model_instance(mi)
            logger.trace(f"CREATED event: started created model instance {mi.name}.")

    def _start_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        Start model instance through a subprocess.

        Args:
            mi: The model instance to start.

        """
        if mi.id in self._provisioning_processes:
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

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
            model = self._get_model(mi)
            backend = get_backend(model)

            # Assign port.
            if not mi.port:
                if self._assigned_ports:
                    unavailable_ports = set.union(*self._assigned_ports.values())
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

            logger.debug(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

            process = multiprocessing.Process(
                target=ServeManager._serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    self._worker_id,
                    self._inference_backend_manager.get_backend_by_name(backend),
                ),
            )
            process.daemon = False
            process.start()
            self._provisioning_processes[mi.id] = process
            self._assigned_ports[mi.id] = set(mi.ports)

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
                # For initialize later mode, the state is set to RUNNING directly,
                # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                if (
                    mi.distributed_servers.mode
                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                ):
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Clean up provisioning process if started.
            if mi.id in self._provisioning_processes:
                self._stop_model_instance(mi)

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

    def _restart_model_instance(self, mi: ModelInstance):
        """
        Restart model instance.

        Args:
            mi: The model instance to restart.
        """

        self._stop_model_instance(mi)
        self._start_model_instance(mi)

    def _update_model_instance(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            set_attr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _stop_model_instance(self, mi: ModelInstance):
        """
        Stop model instance and clean up.

        Args:
            mi: The model instance to stop.
        """

        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Teardown provisioning process if still alive.
        if self._is_provisioning(mi):
            terminate_process_tree(self._provisioning_processes[mi.id].pid)

        # Delete workload.
        delete_workload(mi.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(mi.id, None)
        self._assigned_ports.pop(mi.id, None)
        self._error_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        When `sync_model_instances_state` catches once RUNNING,
        the accumulated `restart_count` will be reset.

        Args:
            mi: The model instance to restart.
        """
        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        restart_count = mi.restart_count or 0
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(10 * (2 ** (restart_count - 1)), 300)
        if restart_count > 0 and last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} (attempt {restart_count + 1}) after {delay} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._update_model_instance(
                mi.id,
                restart_count=restart_count + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message="",
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

    def _get_model(self, mi: ModelInstance) -> Model:
        """
        Efficiently get model related to the model instance with caching.

        Args:
            mi: The model instance whose model to get.
        """
        if model := self._model_cache_by_instance.get(mi.id):
            return model

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _is_provisioning(self, mi: ModelInstance) -> bool:
        """
        Check if the model instance is still provisioning.

        Args:
            mi: The model instance to check.
        """
        if process := self._provisioning_processes.get(mi.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False


def is_ready(
    backend: str,
    mi: ModelInstance,
    inference_backend: Optional[InferenceBackend],
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
        logger.error(f"Error checking model instance {mi.name} health: {e}")
        pass
    return False
