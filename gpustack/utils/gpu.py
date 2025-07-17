import re

from typing import Tuple, Union, List, Callable

from gpustack.schemas.workers import GPUDeviceInfo, WorkerBase

pattern = r"^(?P<worker_name>.+):(?P<device>[^:]+):(?P<gpu_index>\d+)$"


def parse_gpu_id(input: str) -> Tuple[bool, dict]:
    """
    Parse the input string to check if it matches the format worker_name:device:gpu_index.

    Args:
        input_string (str): The input string to parse.

    Returns:
        tuple: (bool, dict)
               - If matched, the first value is True, and the second value is a dictionary
                 containing worker_name, device, and gpu_index.
               - If not matched, the first value is False, and the second value is None.
    """

    match = re.match(pattern, input)
    if match:
        return True, match.groupdict()
    return False, None


def parse_gpu_ids_by_worker(gpu_ids: list) -> dict:
    """
    Group GPU IDs by worker name.

    Args:
        gpu_ids (list): List of GPU IDs.

    Returns:
        dict: A dictionary where the keys are worker names and the values are lists of GPU IDs.
    """

    worker_gpu_ids = {}
    for gpu_id in gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise ValueError(f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        if worker_name not in worker_gpu_ids:
            worker_gpu_ids[worker_name] = []
        worker_gpu_ids[worker_name].append(gpu_id)

    for worker_name, gpu_ids in worker_gpu_ids.items():
        worker_gpu_ids[worker_name] = sorted(gpu_ids)
    return worker_gpu_ids


def all_gpu_match(
    worker: Union[List[WorkerBase], WorkerBase], verify: Callable[[GPUDeviceInfo], bool]
) -> bool:
    """
    Check if all GPUs in the worker match the given callable condition.

    Args:
        worker (Union[List[WorkerBase], WorkerBase]): A worker or a list of workers.
        verify (Callable[GPUDeviceInfo], bool): A function that takes a GPU device and returns a boolean.

    Returns:
        bool: True if all GPUs match the condition, False otherwise.
    """
    if isinstance(worker, list):
        return all(all_gpu_match(w, verify) for w in worker)

    if not worker.status or not worker.status.gpu_devices:
        return False
    return all(verify(gpu) for gpu in worker.status.gpu_devices)


def any_gpu_match(
    worker: Union[List[WorkerBase], WorkerBase], verify: Callable[[GPUDeviceInfo], bool]
) -> bool:
    """
    Check if any GPU in the worker matches the given callable condition.

    Args:
        worker (Union[List[WorkerBase], WorkerBase]): A worker or a list of workers.
        verify (Callable[GPUDeviceInfo], bool): A function that takes a GPU device and returns a boolean.

    Returns:
        bool: True if any GPU matches the condition, False otherwise.
    """
    if isinstance(worker, list):
        return any(any_gpu_match(w, verify) for w in worker)

    if not worker.status or not worker.status.gpu_devices:
        return False
    return any(verify(gpu) for gpu in worker.status.gpu_devices)
