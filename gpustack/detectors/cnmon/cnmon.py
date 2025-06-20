import json
import logging
import subprocess
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
from gpustack.utils.convert import safe_float, safe_int

logger = logging.getLogger(__name__)


class Cnmon(GPUDetector):
    def is_available(self) -> bool:
        cnmon_available = is_command_available("cnmon")
        if not cnmon_available:
            return False

        return True

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_mlu()
        results = self._run_command(command)

        if results is None:
            return []

        return self.decode_mlu_devices(results)

    def decode_mlu_devices(self, result) -> GPUDevicesInfo:
        devices = []
        parsed_json = json.loads(result)

        # Get CnmonInfo
        cnmon_info = parsed_json.get("CnmonInfo", [])

        for card_data in cnmon_info:
            index = card_data.get("CardNum")
            name = card_data.get("ProductName")
            uuid = card_data.get("UUID")

            memory_usage = card_data.get("PhysicalMemUsage", {})
            memory_total = safe_int(memory_usage.get("Total", 0)) * 1024 * 1024
            memory_used = safe_int(memory_usage.get("Used", 0)) * 1024 * 1024

            temperature_info = card_data.get("Temperature", {})
            temperature_gpu = safe_float(temperature_info.get("Chip", 0))

            utilization_info = card_data.get("Utilization", {})
            utilization_gpu = safe_float(utilization_info.get("MLUAverage"), 0)

            vendor = VendorEnum.Cambricon.value
            device_type = platform.DeviceTypeEnum.MLU.value

            device = GPUDeviceInfo(
                index=index,
                device_index=index,
                device_chip_index=0,
                name=name,
                uuid=uuid,
                vendor=vendor,
                memory=MemoryInfo(
                    is_unified_memory=False,
                    used=memory_used,
                    total=memory_total,
                    utilization_rate=(
                        (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    ),
                ),
                core=GPUCoreInfo(
                    utilization_rate=utilization_gpu,
                    total=0,
                ),
                temperature=temperature_gpu,
                type=device_type,
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        result = None
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True,
                shell=True,
                text=True,
                encoding="utf-8",
            )
            output = result.stdout

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

            return output
        except Exception as e:
            raise Exception(
                f"Failed to execute {command}: {e},"
                f" stdout: {result.stdout}, stderr: {result.stderr}"
            )

    def _command_gather_mlu(self):
        executable_command = [
            "cnmon info -e -m -u -j > /dev/null && cat cnmon_info.json",
        ]
        return executable_command
