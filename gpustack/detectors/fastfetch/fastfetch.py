import json
import logging
import platform
import subprocess
from gpustack.detectors.base import SystemInfoDetector
from gpustack.schemas.workers import (
    CPUInfo,
    KernelInfo,
    MemoryInfo,
    MountPoint,
    OperatingSystemInfo,
    SwapInfo,
    SystemInfo,
    UptimeInfo,
)
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)


class Fastfetch(SystemInfoDetector):
    def is_available(self) -> bool:
        try:
            self._run_command(self._command_version(), parse_output=False)
            return True
        except Exception as e:
            logger.warning(f"Fastfetch is not available: {e}")
            return False

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
                    architecture=self._get_value(r, "architecture"),
                )

                system_info.kernel = k
            elif typ == "Uptime":
                uptime_value = self._get_value(r, "uptime")
                uptime = (
                    uptime_value / 1000
                    if uptime_value and isinstance(uptime_value, (int, float))
                    else 0
                )
                system_info.uptime = UptimeInfo(
                    uptime=uptime,
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

    @staticmethod
    def _run_command(command, parse_output=True):
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

        except Exception as e:
            error_message = f"Failed to execute {command}: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            raise Exception(error_message)

        if not parse_output:
            return output

        try:
            parsed_json = json.loads(output)
            return parsed_json
        except Exception as e:
            raise Exception(
                f"Failed to parse the output of {command}: {e}, output: {output}"
            )

    @staticmethod
    def _command_executable_path():
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

    @staticmethod
    def _get_value(input: dict, *keys):
        current_value = input
        for key in keys:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                return None
        return current_value
