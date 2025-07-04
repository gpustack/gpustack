from abc import ABC, abstractmethod

from gpustack.schemas.workers import GPUDevicesInfo, SystemInfo


class GPUDetector(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def gather_gpu_info(self) -> GPUDevicesInfo:
        pass


class SystemInfoDetector(ABC):
    @abstractmethod
    def gather_system_info(self) -> SystemInfo:
        pass


# This exception assigns the error message to state_message and transitions the state to NOT_READY
# Example: raise GPUDetectException("GPU device not detected in the system")
# Then the state will be NOT_READY and state_message will be "GPU device not detected in the system"
class GPUDetectExepction(Exception):
    pass
