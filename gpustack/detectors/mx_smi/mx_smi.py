from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    MemoryInfo,
    VendorEnum,
)
from gpustack.utils import platform
from gpustack.utils.command import is_command_available

# using: pip install /opt/maca/share/mxsml/*.whl
try:
    import pymxsml as mxsmi
except ImportError:
    mxsmi = None


class MXSMI(GPUDetector):

    def __init__(self):
        super().__init__()
        if mxsmi:
            mxsmi.mxSmlInit()

    def is_available(self) -> bool:
        return is_command_available("mx-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        # 1.mxsmi not ready
        if mxsmi is None:
            return []
        # 2.not device
        device_num = mxsmi.mxSmlGetDeviceCount()
        if device_num == 0:
            return []
        # 3.get gpu info
        devices = []
        for i in range(device_num):
            device_info = mxsmi.mxSmlGetDeviceInfo(i)
            memory_info = mxsmi.mxSmlGetMemoryInfo(i)
            temperature_info = mxsmi.mxSmlGetTemperatureInfo(
                i, mxsmi.MXSML_TEMPERATURE_HOTSPOT
            )
            utilization_info = mxsmi.mxSmlGetDeviceIpUsage(i, mxsmi.MXSML_USAGE_XCORE)
            # i ,device_info.deviceName,memory_info.vramTotal,memory_info.vramUse,utilization_info,int(temperature_info / 100),
            # 0 X201 67108864 846264 0 38
            memory_used = memory_info.vramUse * 1024
            memory_total = memory_info.vramTotal * 1024
            temperature_core = int(temperature_info / 100)
            # device info
            device = GPUDeviceInfo(
                index=i,
                name=device_info.deviceName,
                vendor=VendorEnum.Metax.value,
                memory=MemoryInfo(
                    is_unified_memory=False,
                    used=memory_used,
                    total=memory_total,
                    utilization_rate=(
                        (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    ),
                ),
                core=GPUCoreInfo(
                    utilization_rate=utilization_info,
                    total=100,
                ),
                temperature=temperature_core,
                type=platform.DeviceTypeEnum.MACA.value,
            )
            devices.append(device)

        return devices
