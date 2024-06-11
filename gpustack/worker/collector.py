from gpustack.schemas.nodes import (
    CPUInfo,
    MemoryInfo,
    OperatingSystemInfo,
    KernelInfo,
    UptimeInfo,
    SwapInfo,
    GPUDevice,
    MountPoint,
)
import socket
import json
import os
import platform
import subprocess
from gpustack.schemas.nodes import NodeStatus, Node
import importlib.resources as pkg_resources


class NodeStatusCollector:
    def __init__(self, node_ip: str):
        self._hostname = socket.gethostname()
        self._node_ip = node_ip

    """A class for collecting node status information."""

    def collect(self) -> Node:
        """Collect node status information."""
        status = NodeStatus()

        results = self._run_fastfetch_and_parse_result()
        for result in results:
            typ = result.get("type")
            r = result.get("result")

            if r is None:
                continue

            match typ:
                case "OS":
                    status.os = OperatingSystemInfo(
                        name=self._get_value(r, "name"),
                        version=self._get_value(r, "version"),
                    )
                case "Kernel":
                    k = KernelInfo(
                        name=self._get_value(r, "name"),
                        release=self._get_value(r, "release"),
                        version=self._get_value(r, "version"),
                        architecure=self._get_value(r, "architecure"),
                    )
                    status.kernel = k
                case "Uptime":
                    status.uptime = UptimeInfo(
                        uptime=self._get_value(r, "uptime"),
                        boot_time=self._get_value(r, "bootTime"),
                    )
                case "CPU":
                    status.cpu = CPUInfo(
                        total=self._get_value(r, "cores", "online"),
                        utilization_rate=self._get_value(r, "cores", "utilizationRate"),
                    )
                case "GPU":
                    device = []
                    list = sorted(r, key=lambda x: x["name"])
                    for index, value in enumerate(list):
                        device.append(
                            GPUDevice(
                                uuid=self._get_value(value, "uuid"),
                                name=self._get_value(value, "name"),
                                vendor=self._get_value(value, "vendor"),
                                index=index,
                                core_total=self._get_value(value, "coreCount"),
                                core_utilization_rate=self._get_value(
                                    value, "coreUtilizationRate"
                                ),
                                temperature=self._get_value(value, "temperature"),
                            )
                        )
                    status.gpu = device
                case "Memory":
                    status.memory = MemoryInfo(
                        total=self._get_value(r, "total"),
                        used=self._get_value(r, "used"),
                    )
                case "Swap":
                    status.swap = SwapInfo(
                        total=self._get_value(r, "total"),
                        used=self._get_value(r, "used"),
                    )
                case "Disk":
                    mountpoints = []
                    for disk in r:
                        mountpoints.append(
                            MountPoint(
                                name=self._get_value(disk, "name"),
                                mount_point=self._get_value(disk, "mountpoint"),
                                mount_from=self._get_value(disk, "mountFrom"),
                                total=self._get_value(disk, "bytes", "total"),
                                used=self._get_value(disk, "bytes", "used"),
                                free=self._get_value(disk, "bytes", "free"),
                                available=self._get_value(disk, "bytes", "available"),
                            )
                        )
                    status.filesystem = mountpoints

        status.state = "active"

        return Node(
            name=self._hostname,
            hostname=self._hostname,
            address=self._node_ip,
            status=status,
        )

    def _run_fastfetch_and_parse_result(self):
        command = self._fastfetch_command()
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            parsed_json = json.loads(output)
            return parsed_json
        except subprocess.CalledProcessError as e:
            e.add_note(command + "execution failed")
            raise
        except json.JSONDecodeError as e:
            e.add_note("failed to parse the output of " + command)
            raise
        except Exception as e:
            e.add_note(
                "error occurred when trying execute and parse the output of " + command
            )
            raise e

    def _fastfetch_command(self):
        command = ""
        command_path = "gpustack.third_party.fastfetch"

        match platform.system():
            case "Windows":
                command = "fastfetch.exe"
            case "Darwin":
                command = "fastfetch-macos-universal"
            case "Linux":
                if "amd64" in platform.machine() or "x86_64" in platform.machine():
                    command = "fastfetch-linux-amd64"
                elif "arm" in platform.machine() or "aarch64" in platform.machine():
                    command = "fastfetch-linux-aarch64"

        if command == "":
            raise ValueError(
                "Unsupported platform: %s %s" % (platform.system(), platform.machine())
            )

        with pkg_resources.path(command_path, command) as executable_path:
            os.chmod(executable_path, 0o755)

        # ${path}/fastfetch --gpu-temp true --gpu-driver-specific true --format json
        executable_command = [
            str(executable_path),
            "--gpu-driver-specific",
            "true",
            "--gpu-temp",
            "true",
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
