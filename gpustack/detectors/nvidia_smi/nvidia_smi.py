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
from gpustack.utils.command import is_command_available


class NvidiaSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("nvidia-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)
        return self.decode_gpu_devices(results)

    def decode_gpu_devices(self, result) -> GPUDevicesInfo:  # noqa: C901
        """
        results example:
        $nvidia-smi --format=csv,noheader --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu
        0, NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 1309 MiB, 0 %, 41
        1, NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 13625 MiB, 0 %, 39
        """

        devices = []
        reader = csv.reader(result.splitlines())
        for row in reader:
            if len(row) < 6:
                continue
            index, name, memory_total, memory_used, utilization_gpu, temperature_gpu = (
                row
            )
            # Convert MiB to bytes
            memory_total = int(memory_total.split()[0]) * 1024 * 1024
            # Convert MiB to bytes
            memory_used = int(memory_used.split()[0]) * 1024 * 1024
            utilization_gpu = float(utilization_gpu.split()[0])  # Remove the '%' sign
            temperature_gpu = float(temperature_gpu)

            device = GPUDeviceInfo(
                index=int(index),
                name=name,
                vendor=VendorEnum.NVIDIA.value,
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
                    total=0,  # Total cores information is not provided by nvidia-smi
                ),
                temperature=temperature_gpu,
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
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

    def _command_gather_gpu(self):
        executable_command = [
            "nvidia-smi",
            "--format=csv,noheader",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        ]
        return executable_command
