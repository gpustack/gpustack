import json
import os
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)


def worker_macos_metal(reserved=False):
    return load_worker_from_file("worker_macos_metal.json", reserved=reserved)


def worker_linux_nvidia_1_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_1_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4080_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_4_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_4_4080_gpu.json", reserved=reserved
    )


def worker_linux_cpu_1(reserved=False):
    return load_worker_from_file("worker_linux_cpu_1.json", reserved=reserved)


def worker_linux_cpu_2(reserved=False):
    return load_worker_from_file("worker_linux_cpu_2.json", reserved=reserved)


def load_worker_from_file(file_name, reserved=False) -> Worker:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        worker_dict = json.loads(file.read())
        status_dict = worker_dict.get("status")
        memory = status_dict.get("memory")
        gpu_devices = status_dict.get("gpu_devices")

        status = WorkerStatus(**status_dict)
        status.memory = MemoryInfo(**memory)

        if gpu_devices:
            status.gpu_devices = [GPUDeviceInfo(**device) for device in gpu_devices]

        worker = Worker(**worker_dict)
        worker.status = status
        worker.system_reserved = SystemReserved(memory=0, gpu_memory=0)

        if reserved:
            system_reserved_dict = worker_dict.get("system_reserved")
            system_reserved = SystemReserved(**system_reserved_dict)
            worker.system_reserved = system_reserved
    return worker
