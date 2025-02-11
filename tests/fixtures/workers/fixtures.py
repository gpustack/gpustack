import json
import os
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)


def macos_metal_1_m1pro(reserved=False):
    return load_from_file("macos_metal_1_m1pro.json", reserved=reserved)


def macos_metal_2_m2(reserved=True):
    return load_from_file("macos_metal_2_m2.json", reserved=reserved)


def linux_nvidia_1_4090x1(reserved=False):
    return load_from_file("linux_nvidia_1_4090x1.json", reserved=reserved)


def linux_nvidia_2_4090x2(reserved=False):
    return load_from_file("linux_nvidia_2_4090x2.json", reserved=reserved)


def linux_nvidia_3_4090x1(reserved=True):
    return load_from_file("linux_nvidia_3_4090x1.json", reserved=reserved)


def linux_nvidia_2_4080x2(reserved=False):
    return load_from_file("linux_nvidia_2_4080x2.json", reserved=reserved)


def linux_nvidia_4_4080x4(reserved=False):
    return load_from_file("linux_nvidia_4_4080x4.json", reserved=reserved)


def linux_nvidia_5_a100x2(reserved=True):
    return load_from_file("linux_nvidia_5_A100x2.json", reserved=reserved)


def linux_nvidia_6_a100x2(reserved=True):
    return load_from_file("linux_nvidia_6_A100x2.json", reserved=reserved)


def linux_nvidia_7_a100x2(reserved=True):
    return load_from_file("linux_nvidia_7_A100x2.json", reserved=reserved)


def linux_nvidia_8_3090x8(reserved=True):
    return load_from_file("linux_nvidia_8_3090x8.json", reserved=reserved)


def linux_nvidia_9_3090x8(reserved=True):
    return load_from_file("linux_nvidia_9_3090x8.json", reserved=reserved)


def linux_nvidia_10_3090x8(reserved=True):
    return load_from_file("linux_nvidia_10_3090x8.json", reserved=reserved)


def linux_rocm_1_7800x1(reserved=True):
    return load_from_file("linux_rocm_1_7800x1.json", reserved=reserved)


def linux_cpu_1(reserved=False):
    return load_from_file("linux_cpu_1.json", reserved=reserved)


def linux_cpu_2(reserved=False):
    return load_from_file("linux_cpu_2.json", reserved=reserved)


def load_from_file(file_name, reserved=False) -> Worker:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        dict = json.loads(file.read())
        status_dict = dict.get("status")
        memory = status_dict.get("memory")
        gpu_devices = status_dict.get("gpu_devices")

        status = WorkerStatus(**status_dict)
        status.memory = MemoryInfo(**memory)

        if gpu_devices:
            status.gpu_devices = [GPUDeviceInfo(**device) for device in gpu_devices]

        worker = Worker(**dict)
        worker.status = status
        worker.system_reserved = SystemReserved(ram=0, vram=0)

        if reserved:
            system_reserved_dict = dict.get("system_reserved")
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
