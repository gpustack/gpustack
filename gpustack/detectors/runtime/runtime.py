from gpustack_runtime.detector import (
    detect_devices,
    manufacturer_to_backend,
    ManufacturerEnum,
)

from gpustack.detectors.base import GPUDetector
from gpustack.schemas import GPUDeviceInfo, GPUDevicesInfo
from gpustack.schemas.workers import GPUCoreInfo, MemoryInfo, GPUNetworkInfo
from gpustack.utils.convert import safe_int


class Runtime(GPUDetector):
    """
    Detect GPUs using gpustack-runtime.
    """

    def is_available(self) -> bool:
        return True

    def gather_gpu_info(self) -> GPUDevicesInfo:
        ret: GPUDevicesInfo = []

        # Detect devices.
        devs = detect_devices(fast=False)
        if not devs:
            return ret

        # Convert to GPUDevicesInfo.
        for dev in devs:
            gpudev = GPUDeviceInfo(
                vendor=dev.manufacturer.value,
                type=manufacturer_to_backend(dev.manufacturer),
                index=dev.index,
                device_index=dev.index,
                device_chip_index=0,
                name=dev.name,
                uuid=dev.uuid,
                driver_version=dev.driver_version,
                runtime_version=dev.runtime_version,
                compute_capability=dev.compute_capability,
                core=GPUCoreInfo(
                    total=dev.cores or 0,
                    utilization_rate=dev.cores_utilization,
                ),
                memory=MemoryInfo(
                    total=dev.memory << 20,  # MiB -> Bytes
                    used=dev.memory_used << 20,  # MiB -> Bytes
                    utilization_rate=dev.memory_utilization,
                ),
                temperature=dev.temperature,
            )
            # Correct device_index if possible.
            if "card_id" in dev.appendix and dev.appendix["card_id"] is not None:
                gpudev.device_index = safe_int(dev.appendix["card_id"])
            # Correct device_chip_index if possible.
            if "device_id" in dev.appendix and dev.appendix["device_id"] is not None:
                gpudev.device_chip_index = safe_int(dev.appendix["device_id"])
            # Record architecture if possible.
            if (
                "arch_family" in dev.appendix
                and dev.appendix["arch_family"] is not None
            ):
                gpudev.arch_family = str(dev.appendix["arch_family"])
            # Record network for Ascend devices if possible.
            if dev.manufacturer == ManufacturerEnum.ASCEND:
                gpudev_network = GPUNetworkInfo(
                    inet=dev.appendix["roce_ip"] if "roce_ip" in dev.appendix else "",
                    netmask=(
                        dev.appendix["roce_mask"] if "roce_mask" in dev.appendix else ""
                    ),
                    gateway=(
                        dev.appendix["roce_gateway"]
                        if "roce_gateway" in dev.appendix
                        else ""
                    ),
                )
                if gpudev_network.inet:
                    gpudev_network.status = "up"
                    gpudev.network = gpudev_network

            ret.append(gpudev)

        return ret
