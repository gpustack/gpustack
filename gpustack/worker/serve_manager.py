import asyncio
import multiprocessing
import psutil
import requests
import setproctitle
import os
from typing import Dict, Optional
import logging

import tenacity


from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.utils import network, platform
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.backends.llama_box import LlamaBoxServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
)
from gpustack.server.bus import Event, EventType


logger = logging.getLogger(__name__)


class ServeManager:
    def __init__(
        self,
        worker_name: str,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._worker_name = worker_name
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._serving_model_instances: Dict[str, multiprocessing.Process] = {}
        self._starting_model_instances: Dict[str, ModelInstance] = {}
        self._model_cache_by_instance: Dict[str, Model] = {}
        self._clientset = clientset
        self._cache_dir = cfg.cache_dir

        os.makedirs(self._serve_log_dir, exist_ok=True)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(2),
        reraise=True,
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying to get worker ID (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
        ),
    )
    def _get_current_worker_id(self):
        workers = self._clientset.workers.list()

        if workers and workers.items:
            for worker in workers.items:
                if worker.name == self._worker_name:
                    self._worker_id = worker.id
                    logger.debug(f"Successfully found worker ID: {worker.id}")
                    return

        raise Exception(f"Worker {self._worker_name} not found.")

    async def watch_model_instances(self):
        while True:
            try:
                if not hasattr(self, "_worker_id"):
                    self._get_current_worker_id()

                await asyncio.sleep(5)
                logger.info("Started watching model instances.")
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except Exception as e:
                logger.error(f"Failed watching model instances: {e}")

    def _handle_model_instance_event(self, event: Event):
        mi = ModelInstance(**event.data)

        if mi.worker_id != self._worker_id:
            # Ignore model instances that are not assigned to this worker node.
            return

        logger.trace(
            f"Received model instance event: {event.type} {mi.name} {mi.state}"
        )

        if mi.state == ModelInstanceStateEnum.ERROR:
            return

        if mi.id in self._serving_model_instances and event.type == EventType.DELETED:
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

    def _start_serve_process(self, mi: ModelInstance):
        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"
        if os.path.exists(log_file_path) and platform.system() != "windows":
            # TODO Windows does not support os.remove() on open files.
            # Investigate file occupation issue.
            os.remove(log_file_path)

        try:
            if mi.port is None:
                mi.port = network.get_free_port()

            logger.info(f"Start serving model instance {mi.name} on port {mi.port}")

            model = self._clientset.models.get(mi.model_id)
            backend = get_backend(model)

            process = multiprocessing.Process(
                target=ServeManager.serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                ),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process
            self._starting_model_instances[mi.id] = mi
            self._model_cache_by_instance[mi.id] = model

            patch_dict = {
                "state": ModelInstanceStateEnum.INITIALIZING,
                "port": mi.port,
                "pid": process.pid,
            }
            self._update_model_instance(mi.id, **patch_dict)

        except Exception as e:
            patch_dict = {
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": f"{e}",
            }
            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to serve model instance: {e}")

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
    ):

        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.server_url,
            headers=client_headers,
        )
        model = clientset.models.get(mi.model_id)
        backend = get_backend(model)

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                if backend == BackendEnum.LLAMA_BOX:
                    LlamaBoxServer(clientset, mi, cfg).start()
                elif backend == BackendEnum.VLLM:
                    VLLMServer(clientset, mi, cfg).start()
                elif backend == BackendEnum.VOX_BOX:
                    VoxBoxServer(clientset, mi, cfg).start()
                else:
                    raise ValueError(f"Unsupported backend {backend}")

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _stop_model_instance(self, mi: ModelInstance):
        id = mi.id
        if id not in self._serving_model_instances:
            logger.warning(f"Model instance {mi.name} is not running. Skipping.")
            return
        else:
            pid = self._serving_model_instances[id].pid
            try:
                terminate_process_tree(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Failed to terminate process {pid}: {e}")

            self._post_stop_model_instance(id)

    def health_check_serving_instances(self):
        for id, process in list(self._serving_model_instances.items()):
            if not process.is_alive():
                # monitor inference server process exit
                exitcode = process.exitcode
                try:
                    mi = self._clientset.model_instances.get(id=id)
                    if mi.state != ModelInstanceStateEnum.ERROR:
                        self._update_model_instance(
                            id,
                            state=ModelInstanceStateEnum.ERROR,
                            state_message=f"Inference server exited with code {exitcode}.",
                        )
                except NotFoundException:
                    pass
                except Exception:
                    logger.error(f"Failed to update model instance {id} state.")
                self._post_stop_model_instance(id)
            elif id in self._starting_model_instances:
                # health check for starting model instances
                mi = self._starting_model_instances[id]
                model = self._model_cache_by_instance[id]
                if is_running(mi, model.backend):
                    mi = self._clientset.model_instances.get(id=id)
                    if mi.state != ModelInstanceStateEnum.ERROR:
                        self._update_model_instance(
                            id, state=ModelInstanceStateEnum.RUNNING
                        )
                    self._starting_model_instances.pop(id, None)

    def _post_stop_model_instance(self, id: str):
        self._serving_model_instances.pop(id, None)
        self._starting_model_instances.pop(id, None)
        self._model_cache_by_instance.pop(id, None)


def is_running(mi: ModelInstance, backend: Optional[str]) -> bool:
    try:
        # Check /v1/models by default if dedicated health check endpoint is not available.
        # This is served by all backends (llama-box, vox-box, vllm)
        health_check_url = f"http://127.0.0.1:{mi.port}/v1/models"
        if backend == BackendEnum.LLAMA_BOX:
            # For llama-box, use /health to avoid printing error logs.
            health_check_url = f"http://127.0.0.1:{mi.port}/health"

        response = requests.get(health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False
