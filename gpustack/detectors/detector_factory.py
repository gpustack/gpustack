import logging
from typing import Dict
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
)
from gpustack.schemas.workers import SystemInfo, VendorEnum
from gpustack.detectors.fastfetch.fastfetch import Fastfetch
from gpustack.detectors.npu_smi.npu_smi import NPUSMI


logger = logging.getLogger(__name__)


class DetectorFactory:
    def __init__(
        self,
        gpu_detectors: Dict[str, GPUDetector] = None,
    ):
        fastfetch = Fastfetch()
        self._check_detector_availability(fastfetch, "default detector")

        self.default_gpu_detector = fastfetch
        self.system_info_detector = fastfetch
        self.gpu_vendor_detector = fastfetch
        self.gpu_detectors = gpu_detectors or {VendorEnum.Huawei: NPUSMI()}

    def detect_gpus(self) -> GPUDevicesInfo:
        gpu_devices = []
        vendors = self.gpu_vendor_detector.gather_gpu_vendor_info()
        for vendor in vendors:
            detector = self._get_gpu_detector_by_vendor(vendor)
            gpus = detector.gather_gpu_info()
            gpu_devices.extend(self._filter_gpu_devices(gpus))

        return gpu_devices

    def detect_system_info(self) -> SystemInfo:
        return self.system_info_detector.gather_system_info()

    def _filter_gpu_devices(self, gpu_devices: GPUDevicesInfo) -> GPUDevicesInfo:
        filtered_gpu_devices = []
        for device in gpu_devices:
            # Ignore the device without memory.
            if device.memory.total == 0:
                continue

            if device.vendor.lower() not in [
                vendor.lower()
                for vendor in [
                    VendorEnum.NVIDIA,
                    VendorEnum.Apple,
                    VendorEnum.MTHREADS,
                    VendorEnum.Huawei,
                ]
            ]:
                continue

            filtered_gpu_devices.append(device)

        return filtered_gpu_devices

    def _get_gpu_detector_by_vendor(self, vendor: str) -> GPUDetector:
        detector = next(
            (v for k, v in self.gpu_detectors.items() if k.lower() in vendor.lower()),
            self.default_gpu_detector,
        )
        self._check_detector_availability(detector, f"GPU detector for {vendor}")
        return detector

    def _check_detector_availability(self, detector, name: str):
        if not detector:
            raise Exception(f"{name} is empty")
        if not detector.is_available():
            raise Exception(f"{name} {detector.__class__.__name__} is not available")
