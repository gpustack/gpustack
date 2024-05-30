import importlib
import inspect
import json
import multiprocessing
import os
import signal
import socket
import requests
import logging
import uvicorn
from contextlib import redirect_stdout, redirect_stderr
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route


from gpustack.agent.serve import InferenceServer
from gpustack.api.exceptions import is_error_response
from gpustack.generated_client.api.model_instances import (
    get_model_instance_v1_model_instances_id_get,
    update_model_instance_v1_model_instances_id_put,
)
from gpustack.generated_client.client import Client
from gpustack.schemas.model_instances import ModelInstance
from gpustack.server.bus import Event, EventType


logger = logging.getLogger(__name__)


def import_from_path(import_path: str):
    module_name, obj_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def is_function(import_path: str):
    func = import_from_path(import_path)
    return inspect.isfunction(func)


def has_call_method_with_request_param(cls):
    if not hasattr(cls, "__call__"):
        return False

    call_method = getattr(cls, "__call__")

    if not inspect.ismethod(call_method) and not inspect.isfunction(call_method):
        return False

    signature = inspect.signature(call_method)
    for param in signature.parameters.values():
        if param.annotation == Request:
            return True

    return False


def is_serving_class(import_path):
    cls = import_from_path(import_path)

    if not inspect.isclass(cls):
        return False

    return has_call_method_with_request_param(cls)


class ServeManager:
    def __init__(self, server_url: str, log_dir: str):
        self._hostname = socket.gethostname()
        self._server_url = server_url
        self._watch_url = f"{server_url}/v1/model_instances?watch=true"
        self._serve_log_dir = f"{log_dir}/serve"
        self._serving_model_instances = {}
        self._client = Client(base_url=server_url)

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def watch_model_instances(self):
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

        if event.type == EventType.CREATED or event.type == EventType.UPDATED:
            process = multiprocessing.Process(
                target=self._serve_model_instance, args=(mi,)
            )
            process.start()
            self._serving_model_instances[mi.id] = process
        elif event.type == EventType.DELETED:
            self._stop_model_instance(mi.id)

    def _update_model_instance_state(self, id: str, state: str, **kwargs):
        result = get_model_instance_v1_model_instances_id_get.sync(
            client=self._client, id=id
        )
        if is_error_response(result):
            raise Exception(f"Failed to get model instance: {result.message}")

        result.state = state
        for key, value in kwargs.items():
            setattr(result, key, value)

        result = update_model_instance_v1_model_instances_id_put.sync(
            client=self._client, id=id, body=result
        )
        if is_error_response(result):
            raise Exception(
                f"Failed to update model instance state to {state}: {result.message}"
            )

    def _serve_model_instance(self, mi: ModelInstance):
        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"
        self._update_model_instance_state(mi.id, "Started")

        try:
            with open(log_file_path, "a", buffering=1) as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    app = Starlette(
                        debug=True,
                        routes=[
                            Route("/", InferenceServer().__call__),
                        ],
                    )
                    uvicorn.run(app, port=8000)
        except Exception as e:
            logger.error(f"Failed to serve model instance: {e}")
            self._update_model_instance_state(mi.id, "Failed")
        finally:
            self._serving_model_instances.pop(mi.id)

    def _stop_model_instance(self, id: str):
        if id not in self._serving_model_instances:
            logger.error(f"Task {id} is not currently executing or does not exist.")
            return
        else:
            os.kill(self._serving_model_instances[id].pid, signal.SIGTERM)
            self._serving_model_instances.pop(id)
