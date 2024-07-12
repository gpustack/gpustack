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
    def __init__(
        self, server_url: str, log_dir: str, clientset: ClientSet, data_dir: str
    ):
        self._hostname = socket.gethostname()
        self._server_url = server_url
        self._serve_log_dir = f"{log_dir}/serve"
        self._serving_model_instances: Dict[str, multiprocessing.Process] = {}
        self._clientset = clientset
        self._cache_dir = os.path.join(data_dir, "cache")

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

        if mi.state == ModelInstanceStateEnum.error:
            return

        if (
            event.type in {EventType.CREATED, EventType.UPDATED}
        ) and not self._serving_model_instances.get(mi.id):
            self._start_serve_process(mi)
        elif event.type == EventType.DELETED and mi.id in self._serving_model_instances:
            self._stop_model_instance(mi.id)

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
                    self._server_url,
                    self._clientset.headers,
                    log_file_path,
                    self._cache_dir,
                ),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process

            patch_dict = {
                "state": ModelInstanceStateEnum.initializing,
                "port": mi.port,
                "pid": process.pid,
            }
            self._update_model_instance(mi.id, **patch_dict)

        except Exception as e:
            patch_dict = {
                "state": ModelInstanceStateEnum.error,
                "state_message": f"{e}",
            }
            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to serve model instance: {e}")

    @staticmethod
    def serve_model_instance(
        mi: ModelInstance,
        server_url: str,
        client_headers: dict,
        log_file_path: str,
        cache_dir: str,
    ):
        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{mi.id}")
        signal.signal(signal.SIGTERM, signal_handler)

        clientset = ClientSet(
            base_url=server_url,
            headers=client_headers,
        )
        with open(log_file_path, "w", buffering=1) as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                InferenceServer(clientset, mi, cache_dir).start()

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _stop_model_instance(self, id: str):
        if id not in self._serving_model_instances:
            logger.error(f"Task {id} is not currently executing or does not exist.")
            return
        else:
            os.kill(self._serving_model_instances[id].pid, signal.SIGTERM)
            self._serving_model_instances.pop(id)

    def monitor_processes(self):
        for id in list(self._serving_model_instances.keys()):
            process = self._serving_model_instances[id]
            if not process.is_alive():
                exitcode = process.exitcode
                if exitcode != 0:
                    print(f"Process {process.pid} exited with exitcode {exitcode}")
                logger.info(f"Model instance {id} has stopped.")
                self._serving_model_instances.pop(id)
