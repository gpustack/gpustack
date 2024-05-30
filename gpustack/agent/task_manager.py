import importlib
import inspect
import json
import multiprocessing
import os
import signal
import requests
import logging
import uvicorn
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing.pool import Pool
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route


from gpustack.api.exceptions import is_error_response
from gpustack.generated_client.api.tasks import (
    get_task_v1_tasks_id_get,
    update_task_v1_tasks_id_put,
)
from gpustack.generated_client.client import Client
from gpustack.schemas.tasks import Task
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


class TaskManager:
    def __init__(self, server_url: str, log_dir: str):
        self._server_url = server_url
        self._watch_url = f"{server_url}/v1/tasks?watch=true"
        self._log_dir = f"{log_dir}/tasks"
        self._executing_tasks = set()
        self._serving_tasks = {}
        self._client = Client(base_url=server_url)

        os.makedirs(self._log_dir, exist_ok=True)

    def watch_tasks(self, pool: Pool):
        # TODO better client

        logger.debug("Start watching tasks in task manager.")
        with requests.get(self._watch_url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        event = Event.from_json(json.loads(line.decode("utf-8")))
                        self._handle_task_event(pool, event)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                    except Exception as e:
                        print(f"Error processing task change: {e}")

    def _handle_task_event(self, pool: Pool, event: Event):
        if event.type == EventType.CREATED or event.type == EventType.UPDATED:
            task = Task(**event.data)
            if (
                is_function(task.method_path)
                and task.id not in self._executing_tasks
                and task.state != "Completed"
            ):
                pool.apply_async(self._execute_function_task, (task,))
                self._executing_tasks.add(task.id)
            elif is_serving_class(task.method_path):
                process = multiprocessing.Process(
                    target=self._execute_serving_task, args=(task,)
                )
                process.start()
                self._serving_tasks[task.id] = process

    def _update_task_state(self, task_id: str, state: str):
        result = get_task_v1_tasks_id_get.sync(client=self._client, id=task_id)
        if is_error_response(result):
            raise Exception(f"Failed to get task: {result.message}")

        result.state = state
        result = update_task_v1_tasks_id_put.sync(
            client=self._client, id=task_id, body=result
        )
        if is_error_response(result):
            raise Exception(f"Failed to update task state to {state}: {result.message}")

    def _execute_serving_task(self, task: Task):
        cls = import_from_path(task.method_path)
        instance = cls()

        self._update_task_state(task.id, "Started")

        app = Starlette(
            debug=True,
            routes=[
                Route("/", instance.__call__),
            ],
        )
        uvicorn.run(app, port=8000)

    def _execute_function_task(self, task: Task):
        log_file_path = f"{self._log_dir}/{task.id}.log"

        try:
            logger.debug(f"Executing task {task.id}")
            self._update_task_state(task.id, "Started")

            func = import_from_path(task.method_path)

            # redirect stdout and stderr to the log file
            with open(log_file_path, "a", buffering=1) as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    func(*task.args)

            self._update_task_state(task.id, "Completed")
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            self._update_task_state(task.id, "Failed")
        finally:
            self._executing_tasks.remove(task.id)

    def _cancel_task(self, task_id: str):
        if task_id in self._executing_tasks:
            process = self._task_processes[task_id]
            os.kill(process.pid, signal.SIGTERM)
            self._update_task_state(task_id, "Cancelled")
        else:
            logger.error(
                f"Task {task_id} is not currently executing or does not exist."
            )
