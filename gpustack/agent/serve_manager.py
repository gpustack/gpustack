import json
import multiprocessing
import setproctitle
import os
import signal
import socket
import time
from typing import Dict
import requests
import logging
import uvicorn
from contextlib import redirect_stdout, redirect_stderr
from starlette.applications import Starlette
from starlette.routing import Route


from gpustack import utils
from gpustack.agent.serve import InferenceServer
from gpustack.api.exceptions import is_error_response
from gpustack.generated_client.api.model_instances import (
    get_model_instance_v1_model_instances_id_get,
    update_model_instance_v1_model_instances_id_put,
)
from gpustack.generated_client.api.nodes import get_nodes_v1_nodes_get
from gpustack.generated_client.client import Client
from gpustack.schemas.model_instances import ModelInstance
from gpustack.server.bus import Event, EventType


logger = logging.getLogger(__name__)


class ServeManager:
    def __init__(self, server_url: str, log_dir: str, client: Client):
        self._hostname = socket.gethostname()
        self._server_url = server_url
        self._watch_url = f"{server_url}/v1/model_instances?watch=true"
        self._serve_log_dir = f"{log_dir}/serve"
        self._serving_model_instances: Dict[str, multiprocessing.Process] = {}
        self._client = client

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def _get_current_node_id(self):
        for _ in range(3):
            try:
                nodes = get_nodes_v1_nodes_get.sync(client=self._client)
            except Exception as e:
                logger.debug(f"Failed to get nodes: {e}")

            if is_error_response(nodes):
                continue
            else:
                for node in nodes.items:
                    if node.hostname == self._hostname:
                        self._node_id = node.id
                        break
            time.sleep(1)

        if not hasattr(self, "_node_id"):
            raise Exception("Failed to get current node id.")

    def watch_model_instances(self):
        if not hasattr(self, "_node_id"):
            self._get_current_node_id()

        # TODO better client

        logger.debug("Start watching model instances.")
        with requests.get(self._watch_url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        event = Event.from_json(json.loads(line.decode("utf-8")))
                        self._handle_model_instance_event(event)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                    except Exception as e:
                        print(f"Error processing instance change: {e}")

    def _handle_model_instance_event(self, event: Event):
        mi = ModelInstance(**event.data)

        if mi.node_id != self._node_id:
            # Ignore model instances that are not assigned to this node.
            return

        if (
            event.type in {EventType.CREATED, EventType.UPDATED}
        ) and not self._serving_model_instances.get(mi.id):
            self._start_serve_process(mi)
        elif event.type == EventType.DELETED and mi.id in self._serving_model_instances:
            self._stop_model_instance(mi.id)

    def _start_serve_process(self, mi: ModelInstance):
        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"

        port = utils.get_free_port()

        try:
            logger.info(f"Starting serving model instance {mi.id} on port {port}")

            process = multiprocessing.Process(
                target=ServeManager.serve_model_instance,
                args=(mi.id, port, log_file_path),
            )
            process.daemon = False
            process.start()
            self._serving_model_instances[mi.id] = process

            patch_dict = {"state": "Running", "port": port, "pid": process.pid}
            self._update_model_instance(mi.id, **patch_dict)

        except Exception as e:
            patch_dict = {"state": "Failed"}
            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to serve model instance: {e}")

    @staticmethod
    def serve_model_instance(id: int, port: int, log_file_path: str):
        setproctitle.setproctitle(f"gpustack_serving_process: model_instance_{id}")

        with open(log_file_path, "a", buffering=1) as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                app = Starlette(
                    debug=True,
                    routes=[
                        Route("/", InferenceServer().__call__),
                    ],
                )
                uvicorn.run(app, host="0.0.0.0", port=port)

    def _update_model_instance(self, id: str, **kwargs):
        result = get_model_instance_v1_model_instances_id_get.sync(
            client=self._client, id=id
        )
        if is_error_response(result):
            raise Exception(f"Failed to get model instance: {result.message}")

        for key, value in kwargs.items():
            setattr(result, key, value)

        result = update_model_instance_v1_model_instances_id_put.sync(
            client=self._client, id=id, body=result
        )
        if is_error_response(result):
            raise Exception(f"Failed to update model instance: {result.message}")

    def _stop_model_instance(self, id: str):
        if id not in self._serving_model_instances:
            logger.error(f"Task {id} is not currently executing or does not exist.")
            return
        else:
            os.kill(self._serving_model_instances[id].pid, signal.SIGTERM)
            self._serving_model_instances.pop(id)

    def monitor_processes(self):
        logger.debug("Started serving process monitor.")
        interval = 60
        while True:
            time.sleep(interval)
            for id in list(self._serving_model_instances.keys()):
                process = self._serving_model_instances[id]
                if not process.is_alive():
                    exitcode = process.exitcode
                    if exitcode != 0:
                        print(f"Process {process.pid} exited with exitcode {exitcode}")
                    logger.info(f"Model instance {id} has stopped.")
                    self._serving_model_instances.pop(id)
