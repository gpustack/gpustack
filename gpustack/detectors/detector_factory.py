import logging
from typing import Dict, Optional, List
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
)
from gpustack.detectors.nvidia_smi.nvidia_smi import NvidiaSMI
from gpustack.schemas.workers import SystemInfo
from gpustack.detectors.fastfetch.fastfetch import Fastfetch
from gpustack.detectors.npu_smi.npu_smi import NPUSMI
from gpustack.detectors.rocm_smi.rocm_smi import RocmSMI
from gpustack.detectors.regredit.regredit import Regredit
from gpustack.utils import platform


logger = logging.getLogger(__name__)


class DetectorFactory:
    def __init__(
        self,
        device: Optional[str] = None,
        gpu_detectors: Optional[Dict[str, List[GPUDetector]]] = None,
    ):
        self.system_info_detector = Fastfetch()
        self.device = device if device else platform.device()
        if self.device:
            all_gpu_detectors = gpu_detectors or self._get_builtin_gpu_detectors()
            self.gpu_detectors = all_gpu_detectors.get(self.device)

        self._validate_detectors()

    def _get_builtin_gpu_detectors(self) -> Dict[str, GPUDetector]:
        fastfetch = Fastfetch()
        return {
            platform.DeviceTypeEnum.CUDA.value: [NvidiaSMI()],
            platform.DeviceTypeEnum.NPU.value: [NPUSMI()],
            platform.DeviceTypeEnum.MPS.value: [fastfetch],
            platform.DeviceTypeEnum.MUSA.value: [fastfetch],
            platform.DeviceTypeEnum.ROCM.value: [RocmSMI(), Regredit()],
        }

    def _validate_detectors(self):
        if not self.system_info_detector.is_available():
            raise Exception(
                f"System info detector {self.system_info_detector.__class__.__name__} is not available"
            )

        if self.device:
            if not self.gpu_detectors:
                raise Exception(f"GPU detectors for {self.device} not supported")

            available = False
            for detector in self.gpu_detectors:
                if detector.is_available():
                    available = True

            if not available:
                raise Exception(f"No GPU detectors available for {self.device}")

    def detect_gpus(self) -> GPUDevicesInfo:
        if not self.device:
            return []

        for detector in self.gpu_detectors:
            if detector.is_available():
                gpus = detector.gather_gpu_info()
                return self._filter_gpu_devices(gpus)

        return []

    def detect_system_info(self) -> SystemInfo:
        return self.system_info_detector.gather_system_info()

    def _filter_gpu_devices(self, gpu_devices: GPUDevicesInfo) -> GPUDevicesInfo:
        # Ignore the device without memory.
        return [device for device in gpu_devices if device.memory.total > 0]
