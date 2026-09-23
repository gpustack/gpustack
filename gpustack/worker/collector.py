import socket
import logging
from typing import Callable, List
from gpustack.config.config import Config
from gpustack.client.generated_clientset import ClientSet
from gpustack.detectors.base import GPUDetectExepction
from gpustack.detectors.custom.custom import Custom
from gpustack.detectors.detector_factory import DetectorFactory
from gpustack.envs import WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS
from gpustack.schemas.workers import (
    MountPoint,
    WorkerStatusPublic,
    WorkerStatus,
    SystemReserved,
    GPUDevicesStatus,
    SystemInfo,
)
from gpustack.utils.profiling import time_decorator
from gpustack.worker.topology_facts import facts_from_devices

logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    _cfg: Config
    _worker_id_getter: Callable[[], int]
    _worker_ifname_getter: Callable[[], str]
    _worker_ip_getter: Callable[[], str]
    _worker_uuid_getter: Callable[[], str]
    _gpu_devices: GPUDevicesStatus
    _system_info: SystemInfo

    @property
    def gpu_devices(self) -> GPUDevicesStatus:
        return self._gpu_devices

    @property
    def system_info(self) -> SystemInfo:
        return self._system_info

    def __init__(
        self,
        cfg: Config,
        worker_ip_getter: Callable[[], str],
        worker_ifname_getter: Callable[[], str],
        worker_id_getter: Callable[[], int],
        worker_uuid_getter: Callable[[], str],
    ):
        self._cfg = cfg
        self._worker_ip_getter = worker_ip_getter
        self._worker_ifname_getter = worker_ifname_getter
        self._worker_id_getter = worker_id_getter
        self._worker_uuid_getter = worker_uuid_getter
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

    @time_decorator(log_slow_seconds=WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS)
    def timed_collect(self, clientset: ClientSet = None, initial: bool = False):
        return self.collect(clientset=clientset, initial=initial)

    def collect(
        self, clientset: ClientSet = None, initial: bool = False
    ) -> WorkerStatusPublic:  # noqa: C901
        """Collect worker status information."""
        status = WorkerStatus.get_default_status()
        messages: List[str] = []
        try:
            system_info = self._detector_factory.detect_system_info()
            status = WorkerStatus.model_validate({**system_info.model_dump()})
        except Exception as e:
            # Said out loud, not only logged. The default status is all
            # zeros, and a worker that ships one is indistinguishable from a
            # host with no memory and no CPU — which is a measurement the
            # server then schedules against. Observed on a worker whose GPU
            # detection was fine and whose system detection was not: it stayed
            # READY with two idle cards while every group placement refused it,
            # because each role wants some RAM and the host appeared to have
            # none. The reason for that lived only in this worker's own log.
            logger.error(f"Failed to detect system info: {e}")
            messages.append(
                f"System information could not be detected, so this worker "
                f"reports no CPU, memory or disk and cannot be scheduled "
                f"onto: {e}"
            )

        if not initial:
            try:
                gpu_devices = self._detector_factory.detect_gpus()
                status.gpu_devices = gpu_devices
            except GPUDetectExepction as e:
                messages.append(str(e))
            except Exception as e:
                logger.error(f"Failed to detect GPU devices: {e}")
        # Both halves can fail independently — the case above had exactly one
        # of them fail — so the message carries whichever did.
        state_message = "\n".join(messages) or None
        self._inject_unified_memory(status)
        self._inject_computed_filesystem_usage(status)
        self._inject_topology_facts(status)
        self._inject_kv_ifname(status)

        # If disable_worker_metrics is set, set metrics_port to -1
        metrics_port = self._cfg.worker_metrics_port
        if self._cfg.disable_worker_metrics:
            metrics_port = -1

        return WorkerStatusPublic(
            advertise_address=self._cfg.advertise_address or self._worker_ip_getter(),
            hostname=socket.gethostname(),
            ip=self._worker_ip_getter(),
            ifname=self._worker_ifname_getter(),
            port=self._cfg.worker_port,
            metrics_port=metrics_port,
            system_reserved=SystemReserved(**self._cfg.get_system_reserved()),
            state_message=state_message,
            status=status,
            worker_uuid=self._worker_uuid_getter(),
            proxy_mode=self._cfg.proxy_mode,
        )

    def _inject_topology_facts(self, status: WorkerStatus):
        """Where this worker is, as its devices report it.

        Left as None when nothing is known, so a worker whose runtime reports
        no fabric at all looks exactly as it did before.
        """
        facts = facts_from_devices(status.gpu_devices or [])
        status.topology_facts = facts or None

    def _inject_kv_ifname(self, status: WorkerStatus):
        """The KV-transfer NIC this worker was told to use, if it was told one.

        Forwarded verbatim rather than normalised. The field is a record of
        what was configured, and a whitespace-only value -- which
        `derive_net_device` deliberately steps over and derives anyway -- is a
        mistake someone wants to see, not one to launder into the `None` that
        here means "nobody set this".
        """
        status.kv_ifname = self._cfg.kv_ifname

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
