# flake8: noqa: C901
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


class HTSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("ht-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)
        if results is None:
            return []

        return self.decode_gpu_devices(results)

    def decode_gpu_devices(self, result) -> GPUDevicesInfo:  # noqa: C901
        """
        results example:
        0, Mars X201, 65536, 826, 0, 35
        1, Mars X201, 65536, 826, 0, 36
        2, Mars X201, 65536, 826, 0, 36
        3, Mars X201, 65536, 826, 0, 35
        4, Mars X201, 65536, 826, 0, 34
        5, Mars X201, 65536, 826, 0, 35
        6, Mars X201, 65536, 826, 0, 37
        7, Mars X201, 65536, 826, 0, 36
        """

        devices = []
        reader = csv.reader(result.splitlines())
        for row in reader:
            index, name, memory_total, memory_used, utilization_gpu, temperature_gpu = (
                row
            )
            index = safe_int(index)
            name = name.strip()
            # Convert MiB to bytes
            memory_total = safe_int(memory_total.strip()) * 1024 * 1024
            # Convert MiB to bytes
            memory_used = safe_int(memory_used.strip()) * 1024 * 1024
            utilization_gpu = safe_float(utilization_gpu.strip())
            temperature_gpu = safe_float(temperature_gpu)
            device = GPUDeviceInfo(
                index=index,
                name=name,
                vendor=VendorEnum.Insi.value,
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
                    total=100,
                ),
                temperature=temperature_gpu,
                type=platform.DeviceTypeEnum.CUDA.value,
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        result = None
        try:
            process = subprocess.Popen(['ht-smi'], stdout=subprocess.PIPE, text=True)
            output, _ = process.communicate()
            result = self.__parse_gpu_info(output)
            return result
        except Exception as e:
            error_message = f"Failed to execute {command}: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            raise Exception(error_message)

    def _command_gather_gpu(self):
        executable_command = ["ht-smi"]
        return executable_command

    def __parse_gpu_info(self, output):
        # 解析输出
        result = []
        lines = output.splitlines()
        i = 0
        data_start = False
        # 查找数据表开始位置
        while i < len(lines):
            line = lines[i]
            # 检测数据表开始标记
            if '====' in line and 'System Management Interface' not in line:
                data_start = True
                # 跳过表头行，找到第一个 GPU 数据行
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('|'):
                    i += 1
                break
            i += 1
        # 处理数据行
        while i < len(lines) and data_start:
            line = lines[i]
            # 检查是否为 GPU 数据行 (以 | 开头且包含数字)
            if line.strip().startswith('|') and any(
                char.isdigit() for char in line.split('|')[1].strip()
            ):
                parts = [p.strip() for p in line.split('|')]
                # 确保有足够的字段
                if len(parts) < 4:
                    i += 1
                    continue
                # 提取GPU索引和名称
                gpu_field = parts[1].strip()
                gpu_parts = gpu_field.split()
                if not gpu_parts or not gpu_parts[0].isdigit():
                    i += 1
                    continue
                gpu_index = gpu_parts[0]
                # 处理 GPU 名称 - 可能需要处理不同格式
                gpu_name_parts = []
                for part in gpu_parts[1:]:
                    if part in ['Off', 'On']:
                        continue
                    gpu_name_parts.append(part)
                gpu_name = ' '.join(gpu_name_parts)
                # 提取GPU利用率 - 处理不同格式
                util_field = parts[3].strip()
                util_parts = util_field.split()
                gpu_util = '0'  # 默认值
                for part in util_parts:
                    if '%' in part:
                        gpu_util = ''.join(filter(str.isdigit, part))
                        break
                # 获取下一行（温度/内存行）
                i += 1
                if i >= len(lines):
                    break
                next_line = lines[i]
                next_parts = [p.strip() for p in next_line.split('|')]
                if len(next_parts) < 4:
                    i += 1
                    continue
                # 提取温度
                temp_field = next_parts[1].strip()
                temp_parts = temp_field.split()
                temp = 'N/A'
                for part in temp_parts:
                    if 'C' in part:
                        temp = ''.join(filter(str.isdigit, part))
                        break
                # 提取内存信息
                mem_field = next_parts[2].strip()
                mem_parts = mem_field.split('/')
                used_mem = 'N/A'
                total_mem = 'N/A'
                if len(mem_parts) >= 2:
                    used_mem = mem_parts[0].strip()
                    # 移除内存单位 (保留数字)
                    total_mem = ''.join(filter(str.isdigit, mem_parts[1]))
                else:
                    # 尝试直接从字段中提取内存信息
                    mem_info = mem_field.split()
                    for info in mem_info:
                        if '/' in info:
                            mem_parts = info.split('/')
                            if len(mem_parts) >= 2:
                                used_mem = mem_parts[0].strip()
                                total_mem = ''.join(filter(str.isdigit, mem_parts[1]))
                # 添加到结果
                result.append(
                    f"{gpu_index}, {gpu_name}, {total_mem}, {used_mem}, {gpu_util}, {temp}"
                )
                # 跳过可能的分隔行
                i += 1
                if i < len(lines) and lines[i].strip().startswith('+--'):
                    i += 1
            # 检查是否到达表格结束或进程信息开始
            elif 'Process:' in line or 'no process found' in line:
                break
            else:
                i += 1
        return "\n".join(result)
