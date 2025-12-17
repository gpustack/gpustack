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
    if not worker:
        return False

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


def find_one_gpu(
    worker: Union[List[WorkerBase], WorkerBase]
) -> Union[GPUDeviceInfo, None]:
    if isinstance(worker, list):
        for w in worker:
            gpu = find_one_gpu(w)
            if gpu is not None:
                return gpu
    elif worker.status and worker.status.gpu_devices:
        return worker.status.gpu_devices[0]

    return None


def compare_compute_capability(current: str | None, target: str | None) -> int:
    """
    Safely compares two CUDA compute capability version strings.

    Args:
        current: The compute capability of the current device (e.g., "7.5").
                 Accepts None, empty, or whitespace-only strings as invalid.
        target:  The required or reference compute capability (e.g., "8.0").
                 Also accepts None or invalid strings.

    Returns:
        -1 if `current` is less than `target`,
         0 if they are equal (including both being invalid),
         1 if `current` is greater than `target`.

    Invalid inputs (None, empty, whitespace, or malformed "X.Y" format)
    are treated as the lowest possible version. Thus:
      - Any valid version > any invalid version.
      - Two invalid versions are considered equal.
    """

    def parse_cc(cc: str | None) -> tuple[int, int] | None:
        """Parse a compute capability string into (major, minor) integers."""
        if cc is None:
            return None
        cc = cc.strip()
        if not cc:
            return None
        parts = cc.split('.', 1)
        if len(parts) != 2:
            return None
        try:
            major = int(parts[0])
            minor = int(parts[1])
            # Compute Capability versions are non-negative
            if major < 0 or minor < 0:
                return None
            return major, minor
        except (ValueError, TypeError):
            return None

    cur_parsed = parse_cc(current)
    tgt_parsed = parse_cc(target)

    # Both invalid → considered equal
    if cur_parsed is None and tgt_parsed is None:
        return 0
    # Current is invalid, target is valid → current < target
    if cur_parsed is None:
        return -1
    # Target is invalid, current is valid → current > target
    if tgt_parsed is None:
        return 1

    # Both are valid: compare numerically
    cur_major, cur_minor = cur_parsed
    tgt_major, tgt_minor = tgt_parsed

    if cur_major > tgt_major:
        return 1
    elif cur_major < tgt_major:
        return -1
    else:
        if cur_minor > tgt_minor:
            return 1
        elif cur_minor < tgt_minor:
            return -1
        else:
            return 0


def abbreviate_gpu_indexes(indexes, max_show=3):
    """Return abbreviated string of GPU indexes, e.g. [0,1,2...(more 4)]"""
    if not indexes:
        return "[]"
    if len(indexes) <= max_show:
        return str(indexes)

    shown = indexes[:max_show]
    hidden_count = len(indexes) - max_show
    return f"[{','.join(map(str, shown))}...(more {hidden_count})]"


def abbreviate_worker_gpu_indexes(
    worker_name: str,
    gpu_indexes: list[int],
    other_worker_count: int,
    other_gpu_count: int,
    max_show_gpu=3,
) -> str:
    """Return abbreviated string of worker GPU indexes, e.g. worker1:[0,1,2...(more 4)]"""
    abbreviated_indexes = abbreviate_gpu_indexes(gpu_indexes, max_show_gpu)
    msg = f"worker {worker_name} GPU indexes {abbreviated_indexes}"
    if other_gpu_count > 0 and other_worker_count > 0:
        msg += f" and {other_gpu_count} {'GPUs' if other_gpu_count > 1 else 'GPU'}"
        msg += f" from other {other_worker_count} {'workers' if other_worker_count > 1 else 'worker'}"
    return msg
