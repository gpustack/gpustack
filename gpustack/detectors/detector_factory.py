import logging
from typing import Dict, Optional
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
)
from gpustack.detectors.nvidia_smi.nvidia_smi import NvidiaSMI
from gpustack.schemas.workers import SystemInfo
from gpustack.detectors.fastfetch.fastfetch import Fastfetch
from gpustack.detectors.npu_smi.npu_smi import NPUSMI
from gpustack.utils import platform


logger = logging.getLogger(__name__)


class DetectorFactory:
    def __init__(
        self,
        device: Optional[str] = None,
        gpu_detectors: Optional[Dict[str, GPUDetector]] = None,
    ):
        self.system_info_detector = Fastfetch()
        self.device = device if device else platform.device()
        if self.device:
            self.gpu_detectors = gpu_detectors or self._get_builtin_gpu_detectors()
            self.gpu_detector = self._get_gpu_detector()

        self._validate_detectors()

    def _get_builtin_gpu_detectors(self) -> Dict[str, GPUDetector]:
        fastfetch = Fastfetch()
        return {
            "cuda": NvidiaSMI(),
            "npu": NPUSMI(),
            "mps": fastfetch,
            "musa": fastfetch,
        }

    def _get_gpu_detector(self) -> Optional[GPUDetector]:
        return self.gpu_detectors.get(self.device)

    def _validate_detectors(self):
        if not self.system_info_detector.is_available():
            raise Exception(
                f"System info detector {self.system_info_detector.__class__.__name__} is not available"
            )

        if self.device:
            if not self.gpu_detector:
                raise Exception(f"GPU detector for {self.device} not supported")
            if not self.gpu_detector.is_available():
                raise Exception(
                    f"GPU detector {self.gpu_detector.__class__.__name__} is not available"
                )

    def detect_gpus(self) -> GPUDevicesInfo:
        if not self.device:
            return []
        gpus = self.gpu_detector.gather_gpu_info()
        return self._filter_gpu_devices(gpus)

    def detect_system_info(self) -> SystemInfo:
        return self.system_info_detector.gather_system_info()

    def _filter_gpu_devices(self, gpu_devices: GPUDevicesInfo) -> GPUDevicesInfo:
        # Ignore the device without memory.
        return [device for device in gpu_devices if device.memory.total > 0]
