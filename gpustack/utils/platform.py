from enum import Enum
import os
import platform
import logging
import threading

from gpustack.utils.command import is_command_available
from gpustack.schemas.workers import VendorEnum

logger = logging.getLogger(__name__)


def system() -> str:
    return platform.uname().system.lower()


def get_native_arch() -> str:
    system = platform.system()
    if system == "Windows":
        import pythoncom

        if threading.current_thread() is not threading.main_thread():
            pythoncom.CoInitialize()

        # Windows emulation will mask the native architecture
        # https://learn.microsoft.com/en-us/windows/arm/apps-on-arm-x86-emulation
        try:
            import wmi

            c = wmi.WMI()
            processor_info = c.Win32_Processor()
            arch_num = processor_info[0].Architecture

            # https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-processor
            arch_map = {
                0: 'x86',
                1: 'MIPS',
                2: 'Alpha',
                3: 'PowerPC',
                5: 'ARM',
                6: 'ia64',
                9: 'AMD64',
                12: 'ARM64',
            }

            arch = arch_map.get(arch_num, 'unknown')
            if arch != 'unknown':
                return arch.lower()
        except Exception as e:
            logger.warning(f"Failed to get native architecture from WMI, {e}")
        finally:
            if threading.current_thread() is not threading.main_thread():
                pythoncom.CoUninitialize()

    return platform.machine().lower()


def arch() -> str:
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "i386": "386",
        "i686": "386",
        "arm64": "arm64",
        "aarch64": "arm64",
        "armv7l": "arm",
        "arm": "arm",
        "ppc64le": "ppc64le",
        "s390x": "s390x",
        "x86": "x86",
        "mips": "mips",
        "alpha": "alpha",
        "powerpc": "powerpc",
        "ia64": "ia64",
    }
    return arch_map.get(get_native_arch(), "unknown")


class DeviceTypeEnum(str, Enum):
    CUDA = "cuda"
    NPU = "npu"
    MPS = "mps"
    ROCM = "rocm"
    MUSA = "musa"


def device() -> str:
    """
    Returns the customized device type. This is similar to the device types in PyTorch but includes some additional types. Examples include:
    - cuda
    - musa
    - npu
    - mps
    - rocm
    - etc.
    """
    if (
        is_command_available("nvidia-smi")
        or os.path.exists("/usr/local/cuda")
        or os.path.exists("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")
    ):
        return DeviceTypeEnum.CUDA.value

    if (
        is_command_available("mthreads-gmi")
        or os.path.exists("/usr/local/musa")
        or os.path.exists("/opt/musa")
    ):
        return DeviceTypeEnum.MUSA.value

    if is_command_available("npu-smi"):
        return "npu"

    if system() == "darwin" and arch() == "arm64":
        return DeviceTypeEnum.MPS.value

    if is_command_available("rocm-smi") or os.path.exists(
        "C:\\Program Files\\AMD\\ROCm"
    ):
        return DeviceTypeEnum.ROCM.value
    return ""


def device_type_from_vendor(vendor: VendorEnum) -> str:
    mapping = {
        VendorEnum.NVIDIA.value: DeviceTypeEnum.CUDA.value,
        VendorEnum.Huawei.value: DeviceTypeEnum.NPU.value,
        VendorEnum.Apple.value: DeviceTypeEnum.MPS.value,
        VendorEnum.AMD.value: DeviceTypeEnum.ROCM.value,
        VendorEnum.MTHREADS.value: DeviceTypeEnum.MUSA.value,
    }

    return mapping.get(vendor, "")
