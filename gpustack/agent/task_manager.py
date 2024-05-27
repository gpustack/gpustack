import importlib
import multiprocessing
import json
import requests

from gpustack.schemas.tasks import Task
from gpustack.server.bus import Event, EventType
from gpustack.logging import logger


def execute_task(task: Task):
    try:
        module_name, func_name = task.method_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        func(*task.args)
    except Exception as e:
        logger.error(f"Failed to execute task: {e}")


class TaskManager:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._watch_url = f"{server_url}/v1/tasks?watch=true"

    def watch_tasks(self, pool: multiprocessing.Pool):
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

    def _handle_task_event(self, pool: multiprocessing.Pool, event: Event):
        if event.type == EventType.CREATED or event.type == EventType.UPDATED:
            task = Task(**event.data)
            pool.apply_async(execute_task, (task,))

    def update_task_state(self, task_id: str, state: str):
        try:
            update_url = f"{self._server_url}/v1/tasks/{task_id}"
            response = requests.put(update_url, json={"state": state})
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to update task {task_id} state to {state}: {e}")
