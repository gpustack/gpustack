from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUDevicesInfo,
)


class Custom(GPUDetector):
    def __init__(self, gpu_devices: GPUDevicesInfo) -> None:
        self._gpu_devices = gpu_devices

    def is_available(self) -> bool:
        return True

    def gather_gpu_info(self) -> GPUDevicesInfo:
        return self._gpu_devices
