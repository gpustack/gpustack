import csv
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


class IXSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("ixsmi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)
        if results is None:
            return []

        return self.decode_gpu_devices(results)

    def decode_gpu_devices(self, result) -> GPUDevicesInfo:  # noqa: C901
        """
        results example:
        $ixsmi --format=csv,noheader --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu
        0, Iluvatar MR-V50, 16384 MiB, 116 MiB, 0 %, 30 C
        1, Iluvatar MR-V100, 32768 MiB, 27996 MiB, 0 %, 36 C
        """

        devices = []
        reader = csv.reader(result.splitlines())
        for row in reader:
            if len(row) < 6:
                continue
            index, name, memory_total, memory_used, utilization_gpu, temperature_gpu = (
                row
            )

            index = safe_int(index)
            name = name.strip()
            # Convert MiB to bytes
            memory_total = safe_int(memory_total.split()[0]) * 1024 * 1024
            # Convert MiB to bytes
            memory_used = safe_int(memory_used.split()[0]) * 1024 * 1024
            utilization_gpu = safe_float(
                utilization_gpu.split()[0]
            )  # Remove the '%' sign
            temperature_gpu = safe_float(temperature_gpu)

            device = GPUDeviceInfo(
                index=index,
                name=name,
                vendor=VendorEnum.Iluvatar.value,
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
                    total=0,  # Total cores information is not provided by ixsmi
                ),
                temperature=temperature_gpu,
                type=platform.DeviceTypeEnum.CUDA.value,
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        result = None
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )

            if result is None or result.stdout is None:
                return None

            output = result.stdout
            if "no devices" in output.lower():
                return None

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

            return output
        except Exception as e:
            error_message = f"Failed to execute {command}: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            raise Exception(error_message)

    def _command_gather_gpu(self):
        executable_command = [
            "ixsmi",
            "--format=csv,noheader",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        ]
        return executable_command
