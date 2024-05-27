import importlib
import json
import requests

from multiprocessing.pool import Pool
from gpustack.api.exceptions import is_error_response
from gpustack.generated_client.api.tasks import (
    get_task_v1_tasks_id_get,
    update_task_v1_tasks_id_put,
)
from gpustack.generated_client.client import Client
from gpustack.schemas.tasks import Task
from gpustack.server.bus import Event, EventType
from gpustack.logging import logger


class TaskManager:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._watch_url = f"{server_url}/v1/tasks?watch=true"
        self._executing_tasks = set()
        self._client = Client(base_url=server_url)

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
            if task.id not in self._executing_tasks and task.state != "Completed":
                self._executing_tasks.add(task.id)
                pool.apply_async(self._execute_task, (task,))

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

    def _execute_task(self, task: Task):
        try:
            logger.debug(f"Executing task {task.id}")
            self._update_task_state(task.id, "Started")

            module_name, func_name = task.method_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            func(*task.args)

            self._update_task_state(task.id, "Completed")
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            self._update_task_state(task.id, "Failed")
        finally:
            self._executing_tasks.remove(task.id)
