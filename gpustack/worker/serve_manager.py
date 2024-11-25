import multiprocessing
import psutil
import setproctitle
import os
import signal
import time
from typing import Dict
import logging


from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.logging import merged_stderr_stdout, stdout_redirected
from gpustack.utils import network
from gpustack.utils.process import terminate_process_tree
from gpustack.utils.signal import signal_handler
from gpustack.worker.backends.llama_box import LlamaBoxServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
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
        self._clientset = clientset
        self._cache_dir = cfg.cache_dir

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def _get_current_worker_id(self):
        for _ in range(3):
            workers = None
            try:
                workers = self._clientset.workers.list()
            except Exception as e:
                logger.debug(f"Failed to get workers: {e}")

            if workers:
                for worker in workers.items:
                    if worker.name == self._worker_name:
                        self._worker_id = worker.id
                        break
            time.sleep(1)

        if not hasattr(self, "_worker_id"):
            raise Exception("Failed to get current worker id.")

    def watch_model_instances(self):
        if not hasattr(self, "_worker_id"):
            self._get_current_worker_id()

        logger.debug("Started watching model instances.")

        self._clientset.model_instances.watch(
            callback=self._handle_model_instance_event
        )

    def _handle_model_instance_event(self, event: Event):
        mi = ModelInstance(**event.data)

        if mi.worker_id != self._worker_id:
            # Ignore model instances that are not assigned to this worker node.
            return

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
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        try:
            if mi.port is None:
                mi.port = network.get_free_port()

            logger.info(f"Start serving model instance {mi.name} on port {mi.port}")

            process = multiprocessing.Process(
                target=ServeManager.serve_model_instance,
                args=(
                    mi,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                ),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process

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
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
    ):

        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{mi.id}")
        signal.signal(signal.SIGTERM, signal_handler)

        clientset = ClientSet(
            base_url=cfg.server_url,
            headers=client_headers,
        )
        model = clientset.models.get(mi.model_id)
        backend = get_backend(model)

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with stdout_redirected(log_file), merged_stderr_stdout():
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

            self._serving_model_instances.pop(id)

    def monitor_processes(self):
        for id in list(self._serving_model_instances.keys()):
            process = self._serving_model_instances[id]
            if not process.is_alive():
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
                self._serving_model_instances.pop(id)
