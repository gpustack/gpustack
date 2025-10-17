import socket
import logging
from typing import Optional, Callable, Dict
from gpustack.config.config import Config
from gpustack.client.generated_clientset import ClientSet
from gpustack.detectors.base import GPUDetectExepction
from gpustack.detectors.custom.custom import Custom
from gpustack.detectors.detector_factory import DetectorFactory
from gpustack.policies.base import Allocated
from gpustack.schemas.models import ComputedResourceClaim
from gpustack.schemas.workers import (
    MountPoint,
    WorkerStatusPublic,
    WorkerStatus,
    SystemReserved,
    GPUDevicesInfo,
    SystemInfo,
)
from gpustack.utils.uuid import get_system_uuid, get_machine_id, get_legacy_uuid


logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    _cfg: Config
    _worker_id_getter: Callable[[], int]
    _worker_ip_getter: Callable[[], str]
    _system_uuid: str
    _machine_id: str
    _system_reserved: SystemReserved
    _gpu_devices: GPUDevicesInfo
    _system_info: SystemInfo

    @property
    def gpu_devices(self) -> GPUDevicesInfo:
        return self._gpu_devices

    @property
    def system_info(self) -> SystemInfo:
        return self._system_info

    def __init__(
        self,
        cfg: Config,
        worker_ip_getter: Callable[[], str],
        worker_id_getter: Callable[[], int],
    ):
        self._cfg = cfg
        self._worker_ip_getter = worker_ip_getter
        self._worker_id_getter = worker_id_getter
        self._system_uuid = get_legacy_uuid(cfg.data_dir) or get_system_uuid(
            cfg.data_dir
        )
        self._machine_id = get_machine_id()
        reserved_config: Dict[str, int] = {**(cfg.system_reserved or {})}
        reserved_config["ram"] = reserved_config.get(
            "ram", reserved_config.pop("memory", 2)
        )
        reserved_config["vram"] = reserved_config.get(
            "vram", reserved_config.pop("gpu_memory", 1)
        )
        # GB to Bytes
        self._system_reserved = SystemReserved(
            ram=reserved_config["ram"] << 30,
            vram=reserved_config["vram"] << 30,
        )

        self._gpu_devices = cfg.get_gpu_devices()
        self._system_info = cfg.get_system_info()
        if self._gpu_devices and self._system_info:
            self._detector_factory = DetectorFactory(
                device="custom",
                gpu_detectors={"custom": [Custom(gpu_devices=self._gpu_devices)]},
                system_info_detector=Custom(system_info=self._system_info),
            )
        elif self._gpu_devices:
            self._detector_factory = DetectorFactory(
                device="custom",
                gpu_detectors={"custom": [Custom(gpu_devices=self._gpu_devices)]},
            )
        elif self._system_info:
            self._detector_factory = DetectorFactory(
                system_info_detector=Custom(system_info=self._system_info)
            )
        else:
            self._detector_factory = DetectorFactory()

    """A class for collecting worker status information."""

    def collect(
        self, clientset: ClientSet = None, initial: bool = False
    ) -> WorkerStatusPublic:  # noqa: C901
        """Collect worker status information."""
        status = WorkerStatus()
        state_message = None
        try:
            system_info = self._detector_factory.detect_system_info()
            status = WorkerStatus.model_validate({**system_info.model_dump()})
        except Exception as e:
            logger.error(f"Failed to detect system info: {e}")

        if not initial:
            try:
                gpu_devices = self._detector_factory.detect_gpus()
                status.gpu_devices = gpu_devices
            except GPUDetectExepction as e:
                state_message = str(e)
            except Exception as e:
                logger.error(f"Failed to detect GPU devices: {e}")
        self._inject_unified_memory(status)
        self._inject_computed_filesystem_usage(status)
        self._inject_allocated_resource(clientset, status)

        # If disable_worker_metrics is set, set metrics_port to -1
        metrics_port = self._cfg.worker_metrics_port
        if self._cfg.disable_worker_metrics:
            metrics_port = -1

        return WorkerStatusPublic(
            hostname=socket.gethostname(),
            ip=self._worker_ip_getter(),
            port=self._cfg.worker_port,
            metrics_port=metrics_port,
            system_reserved=self._system_reserved,
            state_message=state_message,
            status=status,
            worker_uuid=self._system_uuid,
            machine_id=self._machine_id,
        )

    def _inject_unified_memory(self, status: WorkerStatus):
        is_unified_memory = False
        if status.gpu_devices is not None and len(status.gpu_devices) != 0:
            is_unified_memory = status.gpu_devices[0].memory.is_unified_memory

        if status.memory is not None:
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

    def _inject_allocated_resource(  # noqa: C901
        self, clientset: ClientSet, status: WorkerStatus
    ):
        if clientset is None:
            return
        worker_id = self._worker_id_getter()
        allocated = Allocated(ram=0, vram={})
        try:
            # TODO avoid listing model_instances with clientset.
            # The calculation might not be needed here.
            model_instances = clientset.model_instances.list()
            for model_instance in model_instances.items:
                if (
                    model_instance.distributed_servers
                    and model_instance.distributed_servers.subordinate_workers
                ):
                    for (
                        subworker
                    ) in model_instance.distributed_servers.subordinate_workers:
                        if subworker.worker_id != worker_id:
                            continue

                        aggregate_computed_resource_claim_allocated(
                            allocated, subworker.computed_resource_claim
                        )

                if model_instance.worker_id != worker_id:
                    continue

                aggregate_computed_resource_claim_allocated(
                    allocated, model_instance.computed_resource_claim
                )

            # inject allocated resources
            if status.memory is not None:
                status.memory.allocated = allocated.ram
            if status.gpu_devices is not None:
                for i, device in enumerate(status.gpu_devices):
                    if device.index in allocated.vram:
                        status.gpu_devices[i].memory.allocated = allocated.vram[
                            device.index
                        ]
                    else:
                        status.gpu_devices[i].memory.allocated = 0
        except Exception as e:
            logger.error(f"Failed to inject allocated resources: {e}")


def aggregate_computed_resource_claim_allocated(
    allocated: Allocated, computed_resource_claim: Optional[ComputedResourceClaim]
):
    """Aggregate allocated resources from a ComputedResourceClaim into Allocated."""
    if computed_resource_claim is None:
        return

    if computed_resource_claim.ram:
        allocated.ram += computed_resource_claim.ram

    for gpu_index, vram in (computed_resource_claim.vram or {}).items():
        allocated.vram[gpu_index] = (allocated.vram.get(gpu_index) or 0) + vram
