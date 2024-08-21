import multiprocessing
import psutil
import setproctitle
import os
import signal
import socket
import time
from typing import Dict
import logging
from contextlib import redirect_stdout, redirect_stderr


from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.utils import network
from gpustack.worker.inference_server import InferenceServer
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
)
from gpustack.server.bus import Event, EventType


logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signum)
    os._exit(0)


class ServeManager:
    def __init__(self, clientset: ClientSet, cfg: Config):
        self._hostname = socket.gethostname()
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._serving_model_instances: Dict[str, multiprocessing.Process] = {}
        self._clientset = clientset
        self._cache_dir = os.path.join(cfg.data_dir, "cache")

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def _get_current_worker_id(self):
        for _ in range(3):
            try:
                workers = self._clientset.workers.list()
            except Exception as e:
                logger.debug(f"Failed to get workers: {e}")

            for worker in workers.items:
                if worker.hostname == self._hostname:
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

        if (
            event.type in {EventType.CREATED, EventType.UPDATED}
        ) and not self._serving_model_instances.get(mi.id):
            self._start_serve_process(mi)
        elif event.type == EventType.DELETED and mi.id in self._serving_model_instances:
            self._stop_model_instance(mi)

    def _start_serve_process(self, mi: ModelInstance):
        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"

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
        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                InferenceServer(clientset, mi, cfg).start()

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
                self._terminate_process_tree(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Failed to terminate process {pid}: {e}")

            self._serving_model_instances.pop(id)

    def _terminate_process_tree(self, pid: int):
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            process.terminate()
            process.wait(timeout=3)
        except psutil.TimeoutExpired:
            process.kill()

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
