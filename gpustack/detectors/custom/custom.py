from typing import Optional
from gpustack.detectors.base import GPUDetector, SystemInfoDetector
from gpustack.schemas.workers import (
    GPUDevicesStatus,
    SystemInfo,
)


class Custom(GPUDetector, SystemInfoDetector):
    def __init__(
        self,
        gpu_devices: Optional[GPUDevicesStatus] = None,
        system_info: Optional[SystemInfo] = None,
    ) -> None:
        self._gpu_devices = gpu_devices
        self._system_info = system_info

    def is_available(self) -> bool:
        return True

    def gather_gpu_info(self) -> GPUDevicesStatus:
        return self._gpu_devices

    def gather_system_info(self) -> SystemInfo:
        return self._system_info
