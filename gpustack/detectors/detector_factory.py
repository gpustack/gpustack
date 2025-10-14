import logging
from typing import Dict, Optional, List
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
    SystemInfoDetector,
)
from gpustack.detectors.runtime.runtime import Runtime
from gpustack.schemas.workers import SystemInfo
from gpustack.detectors.fastfetch.fastfetch import Fastfetch

logger = logging.getLogger(__name__)


class DetectorFactory:
    def __init__(
        self,
        device: Optional[str] = None,
        gpu_detectors: Optional[Dict[str, List[GPUDetector]]] = None,
        system_info_detector: Optional[SystemInfoDetector] = None,
    ):
        self.system_info_detector = system_info_detector or Fastfetch()
        self.device = device
        if device:
            self.gpu_detectors = gpu_detectors.get(device) or []
        else:
            self.gpu_detectors = [Runtime()]

    def detect_gpus(self) -> GPUDevicesInfo:
        for detector in self.gpu_detectors:
            if detector.is_available():
                gpus = detector.gather_gpu_info()
                if gpus:
                    return self._filter_gpu_devices(gpus)

        return []

    def detect_system_info(self) -> SystemInfo:
        return self.system_info_detector.gather_system_info()

    @staticmethod
    def _filter_gpu_devices(gpu_devices: GPUDevicesInfo) -> GPUDevicesInfo:
        filtered: GPUDevicesInfo = []
        for device in gpu_devices:
            if not device.memory or not device.memory.total or device.memory.total <= 0:
                logger.debug(
                    f"Skipping GPU device {device.name} ({device.device_index}, {device.device_chip_index}) due to invalid memory info"
                )
                continue
            filtered.append(device)
        return filtered
