import json
import os
from typing import Optional

from gpustack.schemas.workers import (
    GPUDeviceStatus,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)


def macos_metal_1_m1pro_21g(
    reserved=False,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with a M1 Pro GPU device with 21GB of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """
    worker = load_from_file(
        "macos_metal_1_m1pro_21g.json", reserved=reserved, return_devices=return_device
    )
    if callback:
        callback(worker)
    return worker


def macos_metal_2_m2_24g(
    reserved=True,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with a M2 GPU device with 24GB (actual allocatable 16GB) of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "macos_metal_2_m2_24g.json", reserved=reserved, return_devices=return_device
    )
    if callback:
        callback(worker)
    return worker


def macos_metal_3_m2ultra_192g(
    reserved=True,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with a M2 Ultra GPU device with 192GB (actual allocatable 187GB) of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "macos_metal_3_m2ultra_192g.json",
        reserved=reserved,
        return_devices=return_device,
    )
    if callback:
        callback(worker)
    return worker


def linux_ascend_1_910b_64gx8(
    reserved=False,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with 8 Ascend 910B devices, each with 64GB of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "linux_ascend_1_910b_64gx8.json",
        reserved=reserved,
        return_devices=return_device,
    )
    if callback:
        callback(worker)
    return worker


def linux_ascend_2_910b_64gx8(
    reserved=False,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with 8 Ascend 910B devices, each with 64GB of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device:  The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "linux_ascend_2_910b_64gx8.json",
        reserved=reserved,
        return_devices=return_device,
    )
    if callback:
        callback(worker)
    return worker


def linux_ascend_3_910b_64gx8(
    reserved=False,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with 8 Ascend 910B devices, each with 64GB of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "linux_ascend_3_910b_64gx8.json",
        reserved=reserved,
        return_devices=return_device,
    )
    if callback:
        callback(worker)
    return worker


def linux_ascend_4_910b_64gx8(
    reserved=False,
    return_device: Optional[int] = None,
    callback=None,
):
    """
    Return a worker with 8 Ascend 910B devices, each with 64GB of memory.
    :param reserved: If True, the worker will have reserved system resources.
    :param return_device: The number of devices to return.
    :param callback: A callback function to be executed after loading the worker.
    :return: Worker object with the specified configuration.
    """

    worker = load_from_file(
        "linux_ascend_4_910b_64gx8.json",
        reserved=reserved,
        return_devices=return_device,
    )
    if callback:
        callback(worker)
    return worker


def linux_nvidia_0_4090_24gx1(reserved=False):
    return load_from_file("linux_nvidia_0_4090_24gx1.json", reserved=reserved)


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


def linux_nvidia_22_H100_80gx8(reserved=False):
    return load_from_file("linux_nvidia_22_H100_80gx8.json", reserved=reserved)


def linux_nvidia_23_H100_80gx8(reserved=False):
    return load_from_file("linux_nvidia_23_H100_80gx8.json", reserved=reserved)


def linux_nvidia_24_H100_80gx8(reserved=False):
    return load_from_file("linux_nvidia_24_H100_80gx8.json", reserved=reserved)


def linux_nvidia_25_H100_80gx8(reserved=False):
    return load_from_file("linux_nvidia_25_H100_80gx8.json", reserved=reserved)


def linux_nvidia_26_H200_141gx8(reserved=False):
    return load_from_file("linux_nvidia_26_H200_141gx8.json", reserved=reserved)


def linux_rocm_1_7800_16gx1(reserved=True):
    return load_from_file("linux_rocm_1_7800_16gx1.json", reserved=reserved)


def linux_rocm_2_7800_16gx2(reserved=True):
    return load_from_file("linux_rocm_2_7800_16gx2.json", reserved=reserved)


def linux_cpu_1(reserved=False):
    return load_from_file("linux_cpu_1.json", reserved=reserved)


def linux_cpu_2(reserved=False):
    return load_from_file("linux_cpu_2.json", reserved=reserved)


def linux_cpu_3(reserved=False):
    return load_from_file("linux_cpu_3.json", reserved=reserved)


def linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1(reserved=False):
    return load_from_file(
        "linux_mix_1_nvidia_4080_16gx1_rocm_7800_16gx1.json", reserved=reserved
    )


def load_from_file(
    file_name, reserved=False, return_devices: Optional[int] = None
) -> Worker:
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
            status.gpu_devices = [GPUDeviceStatus(**device) for device in gpu_devices]

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
    if return_devices is not None:
        worker.status.gpu_devices = worker.status.gpu_devices[
            : max(0, min(return_devices, len(worker.status.gpu_devices)))
        ]
    return worker
