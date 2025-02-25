import json
import os
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)


def macos_metal_1_m1pro_21g(reserved=False):
    return load_from_file("macos_metal_1_m1pro_21g.json", reserved=reserved)


def macos_metal_2_m2_24g(reserved=True):
    return load_from_file("macos_metal_2_m2_24g.json", reserved=reserved)


def linux_nvidia_1_4090_24gx1(reserved=False):
    return load_from_file("linux_nvidia_1_4090_24gx1.json", reserved=reserved)


def linux_nvidia_3_4090_24gx1(reserved=True):
    return load_from_file("linux_nvidia_3_4090_24gx1.json", reserved=reserved)


def linux_nvidia_2_4080_16gx2(reserved=False):
    return load_from_file("linux_nvidia_2_4080_16gx2.json", reserved=reserved)


def linux_nvidia_4_4080_16gx4(reserved=False):
    return load_from_file("linux_nvidia_4_4080_16gx4.json", reserved=reserved)


def linux_nvidia_5_a100_80gx2(reserved=True):
    return load_from_file("linux_nvidia_5_A100_80gx2.json", reserved=reserved)


def linux_nvidia_6_a100_80gx2(reserved=True):
    return load_from_file("linux_nvidia_6_A100_80gx2.json", reserved=reserved)


def linux_nvidia_7_a100_80gx2(reserved=True):
    return load_from_file("linux_nvidia_7_A100_80gx2.json", reserved=reserved)


def linux_nvidia_8_3090_24gx8(reserved=True):
    return load_from_file("linux_nvidia_8_3090_24gx8.json", reserved=reserved)


def linux_nvidia_9_3090_24gx8(reserved=True):
    return load_from_file("linux_nvidia_9_3090_24gx8.json", reserved=reserved)


def linux_nvidia_10_3090_24gx8(reserved=True):
    return load_from_file("linux_nvidia_10_3090_24gx8.json", reserved=reserved)


def linux_nvidia_11_V100_32gx2(reserved=True):
    return load_from_file("linux_nvidia_11_V100_32gx2.json", reserved=reserved)


def linux_nvidia_12_A40_48gx2(reserved=True):
    return load_from_file("linux_nvidia_12_A40_48gx2.json", reserved=reserved)


def linux_nvidia_13_A100_80gx8(reserved=True):
    return load_from_file("linux_nvidia_13_A100_80gx8.json", reserved=reserved)


def linux_nvidia_14_A100_40gx2(reserved=True):
    return load_from_file("linux_nvidia_14_A100_40gx2.json", reserved=reserved)


def linux_nvidia_15_4080_16gx8(reserved=True):
    return load_from_file("linux_nvidia_15_4080_16gx8.json", reserved=reserved)


def linux_nvidia_16_5000_16gx8(reserved=True):
    return load_from_file("linux_nvidia_16_5000_16gx8.json", reserved=reserved)


def linux_nvidia_17_4090_24gx8(reserved=True):
    return load_from_file("linux_nvidia_17_4090_24gx8.json", reserved=reserved)


def linux_nvidia_18_4090_24gx4_4080_16gx4(reserved=True):
    return load_from_file(
        "linux_nvidia_18_4090_24gx4_4080_16gx4.json", reserved=reserved
    )


def linux_nvidia_19_4090_24gx2(reserved=False):
    return load_from_file("linux_nvidia_19_4090_24gx2.json", reserved=reserved)


def linux_nvidia_20_3080_12gx8(reserved=False):
    return load_from_file("linux_nvidia_20_3080_12gx8.json", reserved=reserved)


def linux_nvidia_21_4090_24gx4_3060_12gx4(reserved=False):
    return load_from_file(
        "linux_nvidia_21_4090_24gx4_3060_12gx4.json", reserved=reserved
    )


def linux_rocm_1_7800_16gx1(reserved=True):
    return load_from_file("linux_rocm_1_7800_16gx1.json", reserved=reserved)


def linux_cpu_1(reserved=False):
    return load_from_file("linux_cpu_1.json", reserved=reserved)


def linux_cpu_2(reserved=False):
    return load_from_file("linux_cpu_2.json", reserved=reserved)


def linux_cpu_3(reserved=False):
    return load_from_file("linux_cpu_3.json", reserved=reserved)


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
