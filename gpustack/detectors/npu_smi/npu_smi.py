import re
import subprocess
from typing import Dict
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

# match |, / and space
split_pattern = re.compile(r"[|/\s]+")


class NPUSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("npu-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)

        mapping = self.gather_gpu_mapping()
        return self.decode_gpu_devices(results, mapping)

    def gather_gpu_mapping(self) -> Dict[tuple[int], int]:
        command = self._command_gather_gpu_mapping()
        results = self._run_command(command)
        mapping = self.decode_gpu_device_mapping(results)
        return mapping

    def decode_gpu_devices(  # noqa: C901
        self, result, gpu_mapping: Dict[tuple[int], int]
    ) -> GPUDevicesInfo:
        """
        results example:
        $npu-smi info
        +------------------------------------------------------------------------------------------------+
        | npu-smi 23.0.rc3.3              Version: 23.0.rc3.3                                           |
        +---------------------------+---------------+----------------------------------------------------+
        | NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
        | Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
        +===========================+===============+====================================================+
        | 0     910B1               | OK            | 271.1       41                0    / 0             |
        | 0                         | 0000:C1:00.0  | 55          0    / 0          65099/ 65536         |
        +===========================+===============+====================================================+
        | 1     910B1               | OK            | 275.0       42                0    / 0             |
        | 0                         | 0000:01:00.0  | 66          0    / 0          65098/ 65536         |
        +===========================+===============+====================================================+
        +---------------------------+---------------+----------------------------------------------------+
        | NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
        +===========================+===============+====================================================+
        | 0       0                 | 576489        | python                   | 60703                   |
        +===========================+===============+====================================================+
        | 1       0                 | 576491        | python                   | 60704                   |
        +===========================+===============+====================================================+
        """

        devices = []
        header_indexes = {}
        for block in result.split('+\n'):
            if "NPU" in block and "Health" in block:
                header_indexes = self._parse_table_header(block)

            if "Process id" in block and "Process name" in block:
                break

            device = self._decode_gpu_device_block(block, header_indexes, gpu_mapping)
            if device is not None:
                devices.append(device)

        return devices

    def _decode_gpu_device_block(
        self,
        block: str,
        header_indexes: Dict[str, int],
        gpu_mapping: Dict[tuple[int], int],
    ) -> GPUDeviceInfo:
        """
        block example:
        | 3     910B1               | OK            | 153.2       39                0    / 0             |
        | 0                         | 0000:02:00.0  | 94          0    / 0          65097/ 65536         |
        +===========================+===============+====================================================
        """
        block_lines = block.split('\n')
        if len(block_lines) != 3:
            return None

        device = GPUDeviceInfo()
        line_num = 1
        npu = None
        chip = None
        for line in block_lines:
            arr = re.split(split_pattern, line)
            arr = [item for item in arr if item]  # remove empty string

            if len(arr) == 0:
                continue

            if not arr[0].isdigit():
                continue

            if line_num == 1:
                npu = safe_int(arr[header_indexes.get("NPU")])
                device.name = arr[header_indexes.get("Name")]
                device.temperature = safe_float(arr[header_indexes.get("Temp(C)")])
                device.vendor = VendorEnum.Huawei.value
                device.type = platform.DeviceTypeEnum.NPU.value

            if line_num == 2:

                vram_used_index = header_indexes.get("Memory-Usage(MB)")
                vram_total_index = vram_used_index + 1
                if header_indexes.get("HBM-Usage(MB)"):
                    vram_used_index = vram_total_index + 1
                    vram_total_index = vram_used_index + 1

                vram_used = safe_int(arr[vram_used_index]) * 1024 * 1024
                vram_total = safe_int(arr[vram_total_index]) * 1024 * 1024
                utilization_rate = vram_used / vram_total * 100 if vram_total > 0 else 0

                chip = safe_int(arr[header_indexes.get("Chip")])
                device.core = GPUCoreInfo(
                    utilization_rate=safe_float(arr[header_indexes.get("AICore(%)")]),
                    total=0,
                )
                device.memory = MemoryInfo(
                    is_unified_memory=False,
                    used=vram_used,
                    total=vram_total,
                    utilization_rate=utilization_rate,
                )
            line_num += 1

        if npu is not None and chip is not None:
            device.index = gpu_mapping.get((npu, chip))

        if device.name != "" and device.index is not None:
            return device
        return None

    def _parse_table_header(self, header_block: str) -> Dict[str, int]:
        """
        header example:
        | NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
        | Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
        +======================+===============+=========================================================
        """
        header_indexes = {}
        for line in header_block.split('\n'):
            arr = re.split(split_pattern, line)
            arr = [item for item in arr if item]
            for i, item in enumerate(arr):
                header_indexes[item] = i
        return header_indexes

    def decode_gpu_device_mapping(self, result: str) -> Dict[tuple[int], int]:
        """
        mapping example:
        NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
        3                              0                              0                              Ascend 910B3
        3                              1                              -                              Mcu
        4                              0                              1                              Ascend 910B3
        4                              1                              -                              Mcu
        """
        mapping = {}
        lines = result.split('\n')
        for line in lines:
            arr = re.split(split_pattern, line)
            arr = [item for item in arr if item]

            if len(arr) < 3:
                continue

            npu_id = arr[0]
            chip_id = arr[1]
            logic_id = arr[2]

            if logic_id.isdigit() and npu_id.isdigit() and chip_id.isdigit():
                mapping[(int(npu_id), int(chip_id))] = int(logic_id)

        return mapping

    def _run_command(self, command):
        result = None
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
            )

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            output = result.stdout
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
            "npu-smi",
            "info",
        ]
        return executable_command

    def _command_gather_gpu_mapping(self):
        executable_command = [
            "npu-smi",
            "info",
            "-m",
        ]
        return executable_command
