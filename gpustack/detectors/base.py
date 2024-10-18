from abc import ABC, abstractmethod

from gpustack.schemas.workers import GPUDevicesInfo


class GPUDetector(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def gather_gpu_info(self) -> GPUDevicesInfo:
        pass
