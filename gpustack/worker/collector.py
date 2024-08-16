import logging
from gpustack.client.generated_clientset import ClientSet
from gpustack.scheduler.policy import Allocated
from gpustack.schemas.workers import (
    CPUInfo,
    GPUCoreInfo,
    MemoryInfo,
    OperatingSystemInfo,
    KernelInfo,
    UptimeInfo,
    SwapInfo,
    GPUDeviceInfo,
    MountPoint,
    VendorEnum,
    WorkerStateEnum,
)
import socket
import json
import os
import platform
import subprocess
from gpustack.schemas.workers import WorkerStatus, Worker
from gpustack.utils.command import get_platform_command
from gpustack.utils.compat_importlib import pkg_resources


logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    def __init__(self, worker_ip: str, worker_name: str, clientset: ClientSet):
        self._worker_name = worker_name
        self._hostname = socket.gethostname()
        self._worker_ip = worker_ip
        self._clientset = clientset

    """A class for collecting worker status information."""

    def collect(self) -> Worker:  # noqa: C901
        """Collect worker status information."""
        status = WorkerStatus()

        results = self._run_fastfetch_and_parse_result()
        for result in results:
            typ = result.get("type")
            r = result.get("result")

            if r is None:
                continue

            if typ == "OS":
                status.os = OperatingSystemInfo(
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

                status.kernel = k
            elif typ == "Uptime":
                status.uptime = UptimeInfo(
                    uptime=self._get_value(r, "uptime"),
                    boot_time=self._get_value(r, "bootTime"),
                )
            elif typ == "CPU":
                total = self._get_value(r, "cores", "online")
                if status.cpu is None:
                    status.cpu = CPUInfo(
                        total=total,
                    )
                else:
                    status.cpu.total = total
            elif typ == "CPUUsage":
                core_count = len(r)
                sum = 0
                for usage_per_core in r:
                    sum += usage_per_core

                utilization_rate = sum / core_count if core_count > 0 else 0

                if status.cpu is None:
                    status.cpu = CPUInfo(
                        utilization_rate=utilization_rate,
                    )
                else:
                    status.cpu.utilization_rate = utilization_rate
            elif typ == "GPU":
                device = []
                list = sorted(r, key=lambda x: x["name"])
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
                        memory_total = (
                            self._get_value(value, "memory", "shared", "total") or 0
                        )
                        memory_used = (
                            self._get_value(value, "memory", "shared", "used") or 0
                        )
                    else:
                        memory_total = (
                            self._get_value(value, "memory", "dedicated", "total") or 0
                        )
                        memory_used = (
                            self._get_value(value, "memory", "dedicated", "used") or 0
                        )
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
                    core_utilization_rate = (
                        self._get_value(value, "coreUtilizationRate") or 0
                    )
                    core = GPUCoreInfo(
                        total=core_count, utilization_rate=core_utilization_rate
                    )

                    # Append.
                    device.append(
                        GPUDeviceInfo(
                            name=name,
                            uuid=self._get_value(value, "uuid"),
                            vendor=self._get_value(value, "vendor"),
                            index=index,
                            core=core,
                            memory=memory,
                            temperature=self._get_value(value, "temperature"),
                        )
                    )

                # Set to status.
                status.gpu_devices = device
            elif typ == "Memory":
                total = self._get_value(r, "total") or 0
                used = self._get_value(r, "used") or 0
                utilization_rate = used / total * 100 if total > 0 else 0

                status.memory = MemoryInfo(
                    total=total,
                    used=used,
                    utilization_rate=utilization_rate,
                )
            elif typ == "Swap":
                total = self._get_value(r, "total") or 0
                used = self._get_value(r, "used") or 0
                utilization_rate = used / total * 100 if total > 0 else 0

                status.swap = SwapInfo(
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

                status.filesystem = mountpoints

        self._inject_unified_memory(status)
        self._inject_computed_filesystem_usage(status)
        self._inject_allocated_resource(status)

        return Worker(
            name=self._worker_name,
            hostname=self._hostname,
            ip=self._worker_ip,
            state=WorkerStateEnum.READY,
            status=status,
        )

    def _inject_unified_memory(self, status: WorkerStatus):
        is_unified_memory = False
        if status.gpu_devices is not None and len(status.gpu_devices) != 0:
            is_unified_memory = status.gpu_devices[0].memory.is_unified_memory

        status.memory.is_unified_memory = is_unified_memory

    def _inject_computed_filesystem_usage(self, status: WorkerStatus):
        if (
            status.os is None
            or "Windows" not in status.os.name
            or status.filesystem is None
        ):
            return

        try:
            computed = MountPoint(
                name="computed",
                mount_point="/",
                total=0,
                used=0,
                free=0,
                available=0,
            )
            for mountpoint in status.filesystem:
                computed.total = computed.total + mountpoint.total
                computed.used = computed.used + mountpoint.used
                computed.free = computed.free + mountpoint.free
                computed.available = computed.available + mountpoint.available

            # inject computed filesystem usage
            status.filesystem.append(computed)
        except Exception as e:
            logger.error(f"Failed to inject filesystem usage: {e}")

    def _inject_allocated_resource(self, status: WorkerStatus) -> Allocated:
        allocated = Allocated(memory=0, gpu_memory={})
        try:
            model_instances = self._clientset.model_instances.list()
            for model_instance in model_instances.items:
                if model_instance.worker_ip != self._worker_ip:
                    continue

                if model_instance.computed_resource_claim is None:
                    continue

                memory = model_instance.computed_resource_claim.memory or 0
                gpu_memory = model_instance.computed_resource_claim.gpu_memory or 0

                allocated.memory += memory
                if model_instance.gpu_index is not None:
                    allocated.gpu_memory[model_instance.gpu_index] = (
                        allocated.gpu_memory.get(model_instance.gpu_index) or 0
                    ) + gpu_memory

            # inject allocated resources
            status.memory.allocated = allocated.memory
            for ag, agv in allocated.gpu_memory.items():
                status.gpu_devices[ag].memory.allocated = agv
        except Exception as e:
            logger.error(f"Failed to inject allocated resources: {e}")

    def _run_fastfetch_and_parse_result(self):
        command = self._fastfetch_command()
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
                f"Failed to execute {command.__str__()}: {e},"
                f" stdout: {result.stdout}, stderr: {result.stderr}"
            )

        try:
            parsed_json = json.loads(output)
            return parsed_json
        except Exception as e:
            raise Exception(
                f"Failed to parse the output of {command.__str__()}: {e}, output: {output}"
            )

    def _fastfetch_command(self):
        command_map = {
            ("Windows", "amd64"): "fastfetch-windows-amd64.exe",
            ("Darwin", "amd64"): "fastfetch-macos-universal",
            ("Darwin", "arm64"): "fastfetch-macos-universal",
            ("Linux", "amd64"): "fastfetch-linux-amd64",
            ("Linux", "arm64"): "fastfetch-linux-aarch64",
        }

        command = get_platform_command(command_map)
        if command == "":
            raise Exception(
                f"No supported fastfetch command found "
                f"for {platform.system()} {platform.machine()}."
            )

        with pkg_resources.path(
            "gpustack.third_party.config.fastfetch", "config.jsonc"
        ) as config_path:
            config_file_path = str(config_path)

        with pkg_resources.path(
            "gpustack.third_party.bin.fastfetch", command
        ) as executable_path:

            if platform.system() not in ["Windows"]:
                os.chmod(executable_path, 0o755)

            # ${path}/fastfetch --gpu-temp true --gpu-driver-specific true \
            # --format json --config ${path}/config.jsonc
            executable_command = [
                str(executable_path),
                "--gpu-driver-specific",
                "true",
                "--gpu-temp",
                "true",
                "--format",
                "json",
                "--config",
                config_file_path,
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
