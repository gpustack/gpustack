import json
import logging
import platform
import subprocess
from gpustack.detectors.base import GPUDetector
from gpustack.schemas.workers import (
    CPUInfo,
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    KernelInfo,
    MemoryInfo,
    MountPoint,
    OperatingSystemInfo,
    SwapInfo,
    SystemInfo,
    UptimeInfo,
    VendorEnum,
)
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)


class Fastfetch(GPUDetector):
    def is_available(self) -> bool:
        try:
            self._run_command(self._command_version(), parse_output=False)
            return True
        except Exception as e:
            logger.warning(f"Fastfetch is not available: {e}")
            return False

    def gather_gpu_info(self) -> GPUDevicesInfo:
        command = self._command_gather_gpu()
        results = self._run_command(command)

        for result in results:
            typ = result.get("type")
            r = result.get("result")

            if r is None:
                continue

            if typ == "GPU":
                gpu_devices = self._decode_gpu_devices(r)
                return gpu_devices

    def gather_system_info(self) -> SystemInfo:  # noqa: C901
        command = self._command_gather_system()
        results = self._run_command(command)

        system_info = SystemInfo()
        for result in results:
            typ = result.get("type")
            r = result.get("result")

            if r is None:
                continue

            if typ == "OS":
                system_info.os = OperatingSystemInfo(
                    name=self._get_value(r, "name"),
                    version=self._get_value(r, "version"),
                )
            elif typ == "Kernel":
                k = KernelInfo(
                    name=self._get_value(r, "name"),
                    release=self._get_value(r, "release"),
                    version=self._get_value(r, "version"),
                    architecure=self._get_value(r, "architecure"),
                )

                system_info.kernel = k
            elif typ == "Uptime":
                system_info.uptime = UptimeInfo(
                    uptime=self._get_value(r, "uptime"),
                    boot_time=self._get_value(r, "bootTime"),
                )
            elif typ == "CPU":
                total = self._get_value(r, "cores", "online")
                if system_info.cpu is None:
                    system_info.cpu = CPUInfo(
                        total=total,
                    )
                else:
                    system_info.cpu.total = total
            elif typ == "CPUUsage":
                core_count = len(r)
                sum = 0
                for usage_per_core in r:
                    sum += usage_per_core

                utilization_rate = sum / core_count if core_count > 0 else 0

                if system_info.cpu is None:
                    system_info.cpu = CPUInfo(
                        utilization_rate=utilization_rate,
                    )
                else:
                    system_info.cpu.utilization_rate = utilization_rate
            elif typ == "Memory":
                total = self._get_value(r, "total") or 0
                used = self._get_value(r, "used") or 0
                utilization_rate = used / total * 100 if total > 0 else 0

                system_info.memory = MemoryInfo(
                    total=total,
                    used=used,
                    utilization_rate=utilization_rate,
                )
            elif typ == "Swap":
                total = self._get_value(r, "total") or 0
                used = self._get_value(r, "used") or 0
                utilization_rate = used / total * 100 if total > 0 else 0

                system_info.swap = SwapInfo(
                    total=total,
                    used=used,
                    utilization_rate=utilization_rate,
                )
            elif typ == "Disk":
                mountpoints = []
                for disk in r:
                    mountpoints.append(
                        MountPoint(
                            name=self._get_value(disk, "name"),
                            mount_point=self._get_value(disk, "mountpoint"),
                            mount_from=self._get_value(disk, "mountFrom"),
                            total=self._get_value(disk, "bytes", "total") or 0,
                            used=self._get_value(disk, "bytes", "used") or 0,
                            free=self._get_value(disk, "bytes", "free") or 0,
                            available=self._get_value(disk, "bytes", "available") or 0,
                        )
                    )

                system_info.filesystem = mountpoints

        return system_info

    def _decode_gpu_devices(self, result: str) -> GPUDevicesInfo:
        devices = []
        list = sorted(result, key=lambda x: x["name"])
        key_set = set()
        for i, value in enumerate(list):
            # Metadatas.
            vender = self._get_value(value, "vendor")
            if vender is None or vender == "":
                continue

            name = self._get_value(value, "name")
            index = self._get_value(value, "index")

            if index is None:
                index = i

            key = f"{name}-{index}"
            if key in key_set:
                for offset in range(len(list)):
                    key = f"{name}-{offset}"
                    if key not in key_set:
                        index = offset
                        key_set.add(key)
                        break
            else:
                key_set.add(key)

            is_unified_memory = False
            if (
                vender == VendorEnum.Apple
                and self._get_value(value, "type") == "Integrated"
            ):
                is_unified_memory = True

            is_integrated = self._get_value(value, "type") == "Integrated"

            # Memory.
            memory_total = 0
            memory_used = 0
            if is_integrated:
                memory_total = self._get_value(value, "memory", "shared", "total") or 0
                memory_used = self._get_value(value, "memory", "shared", "used") or 0
            else:
                memory_total = (
                    self._get_value(value, "memory", "dedicated", "total") or 0
                )
                memory_used = self._get_value(value, "memory", "dedicated", "used") or 0
            memory_utilization_rate = (
                (memory_used / memory_total * 100) if memory_total > 0 else 0
            )
            memory = MemoryInfo(
                is_unified_memory=is_unified_memory,
                total=memory_total,
                used=memory_used,
                utilization_rate=memory_utilization_rate,
            )

            # Core.
            core_count = self._get_value(value, "coreCount") or 0
            core_utilization_rate = self._get_value(value, "coreUsage") or 0
            core = GPUCoreInfo(total=core_count, utilization_rate=core_utilization_rate)

            vendor = self._get_value(value, "vendor")
            vendor_all_values = [vendor.value for vendor in VendorEnum]
            vendor = next(
                (v for v in vendor_all_values if v.lower() in vendor.lower()), vendor
            )

            # Append.
            devices.append(
                GPUDeviceInfo(
                    name=name,
                    uuid=self._get_value(value, "uuid"),
                    vendor=vendor,
                    index=index,
                    core=core,
                    memory=memory,
                    temperature=self._get_value(value, "temperature"),
                )
            )

        return devices

    def _run_command(self, command, parse_output=True):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            output = result.stdout

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

        except Exception as e:
            raise Exception(
                f"Failed to execute {command}: {e},"
                f" stdout: {result.stdout}, stderr: {result.stderr}"
            )

        if not parse_output:
            return output

        try:
            parsed_json = json.loads(output)
            return parsed_json
        except Exception as e:
            raise Exception(
                f"Failed to parse the output of {command}: {e}, output: {output}"
            )

    def _command_executable_path(self):
        command = "fastfetch"
        if platform.system().lower() == "windows":
            command += ".exe"

        with pkg_resources.path(
            "gpustack.third_party.bin.fastfetch", command
        ) as executable_path:
            return str(executable_path)

    def _command_version(self):
        executable_path = self._command_executable_path()
        executable_command = [
            executable_path,
            "--version",
        ]
        return executable_command

    def _command_gather_gpu(self):
        with pkg_resources.path(
            "gpustack.detectors.fastfetch", "config_gpu.jsonc"
        ) as config_path:
            executable_path = self._command_executable_path()

            executable_command = [
                executable_path,
                "--config",
                str(config_path),
                "--gpu-driver-specific",
                "true",
                "--gpu-temp",
                "true",
                "--gpu-detection-method",
                "pci",
                "--format",
                "json",
            ]
            return executable_command

    def _command_gather_system(self):
        with pkg_resources.path(
            "gpustack.detectors.fastfetch", "config_system_info.jsonc"
        ) as config_path:
            executable_path = self._command_executable_path()
            executable_command = [
                executable_path,
                "--config",
                str(config_path),
                "--format",
                "json",
            ]
            return executable_command

    def _get_value(self, input: dict, *keys):
        current_value = input
        for key in keys:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                return None
        return current_value
