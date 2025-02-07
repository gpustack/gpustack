import json
import os
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)


def worker_macos_metal_1(reserved=False):
    return load_worker_from_file("worker_macos_metal_1.json", reserved=reserved)


def worker_macos_metal_2(reserved=True):
    return load_worker_from_file("worker_macos_metal_2.json", reserved=reserved)


def worker_linux_nvidia_1_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_1_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_3_4090_gpu(reserved=True):
    return load_worker_from_file(
        "worker_linux_nvidia_3_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4080_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_4_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_4_4080_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_5_a100_gpu(reserved=True):
    return load_worker_from_file(
        "worker_linux_nvidia_5_A100_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_6_a100_gpu(reserved=True):
    return load_worker_from_file(
        "worker_linux_nvidia_6_A100_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_7_a100_gpu(reserved=True):
    return load_worker_from_file(
        "worker_linux_nvidia_7_A100_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_8_3090_gpu(reserved=True):
    return load_worker_from_file(
        "worker_linux_nvidia_8_3090_gpu.json", reserved=reserved
    )


def worker_linux_rocm_1_7800_gpu(reserved=True):
    return load_worker_from_file("worker_linux_rocm_1_7800_gpu.json", reserved=reserved)


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
        worker.system_reserved = SystemReserved(ram=0, vram=0)

        if reserved:
            system_reserved_dict = worker_dict.get("system_reserved")
            system_reserved = SystemReserved(
                ram=system_reserved_dict.get("memory")
                or system_reserved_dict.get("ram")
                or 0,
                vram=system_reserved_dict.get("gpu_memory")
                or system_reserved_dict.get("vram")
                or 0,
            )
            worker.system_reserved = system_reserved
    return worker
