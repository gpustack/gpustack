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
    GPUDevice,
    MountPoint,
    WorkerStateEnum,
)
import socket
import json
import os
import platform
import subprocess
from gpustack.schemas.workers import WorkerStatus, Worker
import importlib.resources as pkg_resources

logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    def __init__(self, worker_ip: str, clientset: ClientSet):
        self._hostname = socket.gethostname()
        self._worker_ip = worker_ip
        self._clientset = clientset

    """A class for collecting worker status information."""

    def collect(self) -> Worker:
        """Collect worker status information."""
        status = WorkerStatus()
        is_unified_memory = False

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
                    total = self._get_value(r, "cores", "online")
                    utilization_rate = self._get_value(r, "cores", "utilizationRate")
                    status.cpu = CPUInfo(
                        total=total,
                        utilization_rate=utilization_rate,
                    )
                case "GPU":
                    device = []
                    list = sorted(r, key=lambda x: x["name"])
                    for index, value in enumerate(list):
                        name = self._get_value(value, "name")
                        if str.startswith(name, "Apple M"):
                            is_unified_memory = True

                        memory_total = (
                            self._get_value(value, "memory", "dedicated", "total") or 0
                        )
                        memory_used = (
                            self._get_value(value, "memory", "dedicated", "used") or 0
                        )
                        memory_utilization_rate = (
                            (memory_used / memory_total * 100)
                            if memory_total > 0
                            else 0
                        )
                        memory = MemoryInfo(
                            total=memory_total,
                            used=memory_used,
                            utilization_rate=memory_utilization_rate,
                        )

                        core_count = self._get_value(value, "coreCount") or 0
                        core_utilization_rate = (
                            self._get_value(value, "coreUtilizationRate") or 0
                        )
                        core = GPUCoreInfo(
                            total=core_count, utilization_rate=core_utilization_rate
                        )

                        device.append(
                            GPUDevice(
                                name=name,
                                uuid=self._get_value(value, "uuid"),
                                vendor=self._get_value(value, "vendor"),
                                index=index,
                                core=core,
                                memory=memory,
                                temperature=self._get_value(value, "temperature"),
                            )
                        )
                    status.gpu = device
                case "Memory":
                    total = self._get_value(r, "total") or 0
                    used = self._get_value(r, "used") or 0
                    utilization_rate = used / total * 100 if total > 0 else 0
                    status.memory = MemoryInfo(
                        total=total,
                        used=used,
                        utilization_rate=utilization_rate,
                    )
                case "Swap":
                    total = self._get_value(r, "total") or 0
                    used = self._get_value(r, "used") or 0
                    utilization_rate = used / total * 100 if total > 0 else 0
                    status.swap = SwapInfo(
                        total=total,
                        used=used,
                        utilization_rate=utilization_rate,
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

        status.memory.is_unified_memory = is_unified_memory
        if is_unified_memory:
            for index, _ in enumerate(status.gpu):
                status.gpu[index].memory = status.memory

        allocated = self._get_allocated_resource()
        status.memory.allocated = allocated.memory
        for ag, agv in allocated.gpu_memory.items():
            status.gpu[ag].memory.allocated = agv

        return Worker(
            name=self._hostname,
            hostname=self._hostname,
            ip=self._worker_ip,
            state=WorkerStateEnum.running,
            status=status,
        )

    def _get_allocated_resource(self) -> Allocated:
        allocated = Allocated(memory=0, gpu_memory={})
        try:
            model_instances = self._clientset.model_instances.list()
            for model_instance in model_instances.items:
                if model_instance.worker_ip != self._worker_ip:
                    continue

                if model_instance.computed_resource_claim is not None:
                    memory = model_instance.computed_resource_claim.memory or 0
                    gpu_memory = model_instance.computed_resource_claim.gpu_memory or 0

                    allocated.memory += memory

                    if model_instance.gpu_index is not None:
                        allocated.gpu_memory[model_instance.gpu_index] = (
                            allocated.gpu_memory.get(model_instance.gpu_index) or 0
                        ) + gpu_memory
        except Exception as e:
            logger.error(f"Failed to get allocated resources: {e}")
        return allocated

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
