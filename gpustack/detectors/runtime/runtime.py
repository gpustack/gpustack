from gpustack_runtime.deployer import supported_list as supported_deployers
from gpustack_runtime.detector import (
    detect_devices,
    expand_mig_devices,
    manufacturer_to_backend,
    ManufacturerEnum,
)

from gpustack.detectors.base import GPUDetector
from gpustack.schemas import GPUDeviceStatus, GPUDevicesStatus
from gpustack.schemas.workers import GPUCoreInfo, MemoryInfo, GPUNetworkInfo
from gpustack.utils.convert import safe_int


def _mig_devices_addressable() -> bool:
    """
    Whether a MIG-partitioned card's MIG devices are what this worker deploys
    onto, which is what decides whether they belong in the inventory.

    Detection reports the physical card and carries its MIG devices in the
    ``mig_devices`` appendix. On Kubernetes clusters that is the whole story:
    the operator's device-manager partitions on demand, so the card is the only
    stable inventory item and a model asks for a slice of it through an
    InstanceType. Docker clusters have no such partitioner — MIG instances are
    created out-of-band with ``nvidia-smi mig``, not per workload — so there the
    instances are the allocatable devices, and a MIG-enabled card itself can run
    nothing.

    The deployer that would run the workload owns the answer, so that the
    inventory can never list a device the deploy path cannot address.
    """
    deployers = supported_deployers()
    return bool(deployers) and deployers[0].allowed_mig_devices


class Runtime(GPUDetector):
    """
    Detect GPUs using gpustack-runtime.
    """

    def is_available(self) -> bool:
        return True

    def gather_gpu_info(self) -> GPUDevicesStatus:
        ret: GPUDevicesStatus = []

        # Detect devices.
        devs = detect_devices(fast=False)
        if not devs:
            return ret

        # Substitute a MIG-partitioned card with its MIG devices where they are
        # the deployable ones.
        if _mig_devices_addressable():
            devs = expand_mig_devices(devs)

        # Convert to GPUDevicesInfo.
        for dev in devs:
            gpudev = GPUDeviceStatus(
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
                    total=(
                        int(dev.memory * (1 << 20)) if dev.memory is not None else 0
                    ),  # MiB -> Bytes
                    used=(
                        int(dev.memory_used * (1 << 20))
                        if dev.memory_used is not None
                        else 0
                    ),  # MiB -> Bytes
                    utilization_rate=dev.memory_utilization,
                ),
                temperature=dev.temperature,
                power=dev.power,
                power_used=dev.power_used,
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
