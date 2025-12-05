import logging
import threading
import time
from typing import Dict, Optional, List
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
    SystemInfoDetector,
)
from gpustack.detectors.runtime.runtime import Runtime
from gpustack.envs import WORKER_GPU_CACHE_TTL, WORKER_GPU_DETECTION_TIMEOUT
from gpustack.schemas.workers import SystemInfo
from gpustack.detectors.fastfetch.fastfetch import Fastfetch
from gpustack.utils.process import threading_stop_event


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

        # GPU Cache
        self._gpu_cache: Optional[GPUDevicesInfo] = None
        self._gpu_cache_ts: float = 0
        self._gpu_cache_ttl = WORKER_GPU_CACHE_TTL
        self._gpu_timeout = WORKER_GPU_DETECTION_TIMEOUT
        self._gpu_lock = threading.Lock()

        # Start background thread to update GPU cache
        self._stop_event = threading_stop_event
        self._bg_thread = threading.Thread(
            target=self._update_gpu_cache_loop, daemon=True
        )
        self._bg_thread.start()

    def __getstate__(self):
        "Define custom serialization to avoid pickling threads and locks"

        state = self.__dict__.copy()
        del state['_gpu_lock']
        del state['_bg_thread']
        del state['_stop_event']
        return state

    def __setstate__(self, state):
        "Define custom deserialization to restore threads and locks"

        self.__dict__.update(state)
        self._stop_event = threading_stop_event
        self._gpu_lock = threading.Lock()
        self._bg_thread = threading.Thread(
            target=self._update_gpu_cache_loop, daemon=True
        )

    def detect_gpus(self) -> GPUDevicesInfo:
        if self._gpu_cache is not None:
            with self._gpu_lock:
                now = time.time()
                delta = now - self._gpu_cache_ts
                if now - self._gpu_cache_ts > self._gpu_cache_ttl:
                    logger.debug(
                        "Using stale cached GPU detection result, now=%s, gpu_cache_ts=%s, delta=%s",
                        now,
                        self._gpu_cache_ts,
                        delta,
                    )
                return self._gpu_cache

        logger.info("No cached GPU detection result, performing synchronous detection")
        result = self._detect_gpus_once()

        with self._gpu_lock:
            self._gpu_cache = result
            self._gpu_cache_ts = time.time()
        return result

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

    def _detect_gpus_once(self) -> GPUDevicesInfo:
        """
        Actual synchronous method to detect GPUs with filtering logic.
        """
        for detector in self.gpu_detectors:
            if detector.is_available():
                gpus = detector.gather_gpu_info()
                if gpus:
                    return self._filter_gpu_devices(gpus)
        return []

    def _update_gpu_cache_loop(self):
        """
        Background thread loop to update GPU cache.
        """
        while not self._stop_event.is_set():
            try:
                result = [None]

                def worker():
                    try:
                        result[0] = self._detect_gpus_once()
                    except Exception as e:
                        logger.error(f"GPU detection error: {e}")

                t = threading.Thread(target=worker)
                t.daemon = True
                t.start()
                t.join(self._gpu_timeout)

                if t.is_alive():
                    logger.warning(
                        f"GPU detection timed out after {self._gpu_timeout}s"
                    )
                    # Thread continues to run in the background, but cache is not updated
                else:
                    with self._gpu_lock:
                        self._gpu_cache = result[0]
                        self._gpu_cache_ts = time.time()

                time.sleep(10)
            except Exception as e:
                logger.error(f"Background GPU cache updater error: {e}")
                time.sleep(10)
