from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from threading import Lock
import time
from typing import Callable
from prometheus_client.registry import Collector
from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client.core import (
    GaugeMetricFamily,
    InfoMetricFamily,
)
from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.policies.resource_view import compute_worker_allocated
from gpustack.utils.name import metric_name
from gpustack.utils.hardware_metrics import (
    gpu_identity,
    is_gpu_partition,
    verified_gpu_devices,
)
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.worker.runtime_metrics_aggregator import runtime_metrics_values
import uvicorn
import logging
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)
HARDWARE_IDENTITY_TTL_SECONDS = 30.0

unified_registry = CollectorRegistry()
raw_registry = CollectorRegistry()


def _safe_label(value, default: str = "unknown") -> str:
    return default if value is None else str(value)


def _add_metric(
    metric: GaugeMetricFamily,
    labels: list[str],
    value: float | int | None,
):
    if value is None:
        return
    metric.add_metric(labels, value)


def _drain(future, kind: str):
    """
    Collected metrics, or an empty list if that half of the collection failed.
    """
    try:
        return future.result()
    except Exception as e:
        logger.error(
            f"Failed to collect {kind} metrics: {e}",
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return []


def _stable_gauge(name, description, labels, values, value, bounds=(None, None)):
    if value is None or not isfinite(value):
        return None
    minimum, maximum = bounds
    if minimum is not None and value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    family = GaugeMetricFamily(metric_name(name), description, labels=labels)
    family.add_metric(values, value)
    return family


def _stable_filesystems(mounts, labels, values):
    unique = {mount.mount_point for mount in mounts}
    for name in ('used_bytes', 'total_bytes', 'utilization_percent'):
        family = None
        for mountpoint in sorted(unique):
            rows = [m for m in mounts if m.mount_point == mountpoint]
            if len(rows) != 1 or not mountpoint:
                continue
            mount = rows[0]
            if name == 'utilization_percent':
                value = (
                    100 * mount.used / mount.total
                    if mount.used is not None and mount.total and mount.total > 0
                    else None
                )
            else:
                value = mount.used if name == 'used_bytes' else mount.total
            metric = _stable_gauge(
                'worker_filesystem_' + name,
                'Owner-qualified filesystem ' + name,
                labels + ['mountpoint'],
                values + [mountpoint],
                value,
                (0, 100 if name == 'utilization_percent' else None),
            )
            if metric is not None:
                if family is None:
                    family = metric
                else:
                    family.samples.extend(metric.samples)
        if family is not None:
            yield family


class MetricExporter(Collector):
    _worker_ip_getter: Callable[[], str]
    _worker_name_getter: Callable[[], str]
    _worker_id_getter: Callable[[], int]
    _collector: WorkerStatusCollector

    def __init__(
        self,
        cfg: Config,
        collector: WorkerStatusCollector,
        worker_name_getter: Callable[[], str],
        worker_ip_getter: Callable[[], str],
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet] = None,
        cache: dict = None,
        worker_uuid_getter: Callable[[], str] = None,
    ):
        self._config = cfg
        self._collector = collector
        self._worker_name_getter = worker_name_getter
        self._worker_id_getter = worker_id_getter
        self._worker_ip_getter = worker_ip_getter
        self._port = cfg.worker_metrics_port
        self._cache = cache
        self._clientset_getter = clientset_getter
        self._worker_uuid_getter = worker_uuid_getter
        self._hardware_identity = None
        self._hardware_identity_lock = Lock()

    def collect(self):
        with ThreadPoolExecutor() as executor:
            worker_future = executor.submit(list, self.collect_worker_metrics())
            runtime_future = executor.submit(list, self.collect_runtime_metrics())
            # Collected independently: worker metrics failing should not take the
            # runtime metrics down with them, nor the other way round.
            yield from _drain(worker_future, "worker")
            yield from _drain(runtime_future, "runtime")

    def collect_worker_metrics(self):  # noqa: C901
        labels = ["worker_id", "worker_name", "instance"]
        filesystem_labels = labels + ["mountpoint"]
        gpu_labels = labels + ["gpu_index", "gpu_name", "gpu_chip_index"]

        # metrics
        os_info = InfoMetricFamily(
            metric_name("worker_node_os"), "Operating system information"
        )
        kernel_info = InfoMetricFamily(
            metric_name("worker_node_kernel"), "Kernel information"
        )
        uptime = GaugeMetricFamily(
            metric_name("worker_node_uptime_seconds"),
            "Uptime in seconds of the worker node",
            labels=labels,
        )
        cpu_cores = GaugeMetricFamily(
            metric_name("worker_node_cpu_cores"),
            "Total CPUs cores of the worker node",
            labels=labels,
        )
        cpu_utilization_rate = GaugeMetricFamily(
            metric_name("worker_node_cpu_utilization_rate"),
            "Rate of CPU utilization on the worker node",
            labels=labels,
        )
        memory_total = GaugeMetricFamily(
            metric_name("worker_node_memory_total_bytes"),
            "Total memory in bytes of the worker node",
            labels=labels,
        )
        memory_used = GaugeMetricFamily(
            metric_name("worker_node_memory_used_bytes"),
            "Memory used in bytes of the worker node",
            labels=labels,
        )
        memory_utilization_rate = GaugeMetricFamily(
            metric_name("worker_node_memory_utilization_rate"),
            "Rate of memory utilization on the worker node",
            labels=labels,
        )
        gpu_info = InfoMetricFamily("worker_node_gpu", "GPU information")
        gpu_cores = GaugeMetricFamily(
            metric_name("worker_node_gpu_cores"),
            "Total GPUs cores of the worker node",
            labels=gpu_labels,
        )
        gpu_utilization_rate = GaugeMetricFamily(
            metric_name("worker_node_gpu_utilization_rate"),
            "Rate of GPU utilization on the worker node",
            labels=gpu_labels,
        )
        gpu_temperature = GaugeMetricFamily(
            metric_name("worker_node_gpu_temperature_celsius"),
            "GPU temperature in celsius of the worker node",
            labels=gpu_labels,
        )
        gpu_power_limit = GaugeMetricFamily(
            metric_name("worker_node_gpu_power_limit_watts"),
            "GPU power management limit in watts of the worker node",
            labels=gpu_labels,
        )
        gpu_power_used = GaugeMetricFamily(
            metric_name("worker_node_gpu_power_used_watts"),
            "GPU power usage in watts of the worker node",
            labels=gpu_labels,
        )
        gram_total = GaugeMetricFamily(
            metric_name("worker_node_gram_total_bytes"),
            "Total GPU RAM in bytes of the worker node",
            labels=gpu_labels,
        )
        gram_allocated = GaugeMetricFamily(
            metric_name("worker_node_gram_allocated_bytes"),
            "Allocated GPU RAM in bytes of the worker node",
            labels=gpu_labels,
        )
        gram_used = GaugeMetricFamily(
            metric_name("worker_node_gram_used_bytes"),
            "GPU RAM used in bytes of the worker node",
            labels=gpu_labels,
        )
        gram_utilization_rate = GaugeMetricFamily(
            metric_name("worker_node_gram_utilization_rate"),
            "Rate of GPU RAM utilization on the worker node",
            labels=gpu_labels,
        )
        filesystem_total = GaugeMetricFamily(
            metric_name("worker_node_filesystem_total_bytes"),
            "Total filesystem in bytes of the worker node",
            labels=filesystem_labels,
        )
        filesystem_used = GaugeMetricFamily(
            metric_name("worker_node_filesystem_used_bytes"),
            "Total filesystem used in bytes of the worker node",
            labels=filesystem_labels,
        )
        filesystem_utilization_rate = GaugeMetricFamily(
            metric_name("worker_node_filesystem_utilization_rate"),
            "Rate of filesystem utilization on the worker node",
            labels=filesystem_labels,
        )
        worker_ip = _safe_label(self._worker_ip_getter())
        worker_id = _safe_label(self._worker_id_getter())
        worker_name = _safe_label(self._worker_name_getter())
        worker_label_values = [worker_id, worker_name, worker_ip]
        try:
            worker = self._collector.timed_collect(clientset=self._clientset_getter())
            status = worker.status
            if status is None:
                logger.error("Empty worker node status from collector.")
                return
        except Exception as e:
            logger.error(f"Failed to get worker node status for metrics exporter: {e}")
            return

        # system
        if status.os is not None:
            os_info.add_metric(
                ["worker_id", "worker_name", "instance", "name", "version"],
                {
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "instance": worker_ip,
                    "name": _safe_label(status.os.name),
                    "version": _safe_label(status.os.version),
                },
            )

        # kernel
        if status.kernel is not None:
            kernel_info.add_metric(
                ["worker_id", "worker_name", "instance", "name", "version"],
                {
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "instance": worker_ip,
                    "name": _safe_label(status.kernel.name),
                    "release": _safe_label(status.kernel.release),
                    "version": _safe_label(status.kernel.version),
                    "architecture": _safe_label(status.kernel.architecture),
                },
            )

        # uptime
        if status.uptime is not None:
            _add_metric(uptime, worker_label_values, status.uptime.uptime)

        # cpu
        if status.cpu is not None:
            _add_metric(cpu_cores, worker_label_values, status.cpu.total)
            _add_metric(
                cpu_utilization_rate,
                worker_label_values,
                status.cpu.utilization_rate,
            )

        # memory
        if status.memory is not None:
            _add_metric(memory_total, worker_label_values, status.memory.total)
            _add_metric(memory_used, worker_label_values, status.memory.used)

            if (
                status.memory.total is not None
                and status.memory.used is not None
                and status.memory.total != 0
            ):
                _add_metric(
                    memory_utilization_rate,
                    worker_label_values,
                    _rate(status.memory.used, status.memory.total),
                )

        # gpu
        # Compute per-GPU allocated locally for the gram_allocated metric.
        # The collector no longer writes allocated into status — the
        # authoritative view is the server-side aggregation in /v2/workers
        # (compute_worker_allocated against the ModelInstance table). We
        # recompute here purely to keep the Prometheus metric working,
        # using the same helper to guarantee identical math.
        allocated_vram: dict[int, int] = {}
        try:
            clientset = self._clientset_getter()
            if clientset is not None:
                model_instances = clientset.model_instances.list()
                allocated_vram = compute_worker_allocated(
                    model_instances.items, self._worker_id_getter()
                ).vram
        except Exception as e:
            logger.error(f"Failed to compute allocated GPU memory for metrics: {e}")

        if status.gpu_devices is not None:
            for i, d in enumerate(status.gpu_devices):
                # Node metrics retain their inventory-slot and single-chip labels.
                gpu_chip_index = "0"
                gpu_label_values = worker_label_values + [
                    str(i),
                    _safe_label(d.name),
                    _safe_label(gpu_chip_index),
                ]
                gpu_info.add_metric(
                    gpu_labels,
                    {
                        "worker_id": worker_id,
                        "worker_name": worker_name,
                        "instance": worker_ip,
                        "gpu_index": str(i),
                        "gpu_chip_index": _safe_label(gpu_chip_index),
                        "gpu_name": _safe_label(d.name),
                    },
                )
                if d.core is not None:
                    _add_metric(gpu_cores, gpu_label_values, d.core.total)
                    _add_metric(
                        gpu_utilization_rate,
                        gpu_label_values,
                        d.core.utilization_rate,
                    )

                _add_metric(gpu_temperature, gpu_label_values, d.temperature)

                _add_metric(gpu_power_limit, gpu_label_values, d.power)
                _add_metric(gpu_power_used, gpu_label_values, d.power_used)

                if d.memory is not None:
                    _add_metric(gram_total, gpu_label_values, d.memory.total)
                    _add_metric(
                        gram_allocated,
                        gpu_label_values,
                        allocated_vram.get(d.index) if d.index is not None else None,
                    )
                    _add_metric(gram_used, gpu_label_values, d.memory.used)

                    if (
                        d.memory.total is not None
                        and d.memory.used is not None
                        and d.memory.total != 0
                    ):
                        _add_metric(
                            gram_utilization_rate,
                            gpu_label_values,
                            _rate(d.memory.used, d.memory.total),
                        )

        yield from self._collect_stable_hardware(status)

        # filesystem
        if status.filesystem is not None:
            for _, d in enumerate(status.filesystem):
                _add_metric(
                    filesystem_total,
                    worker_label_values + [_safe_label(d.mount_point)],
                    d.total,
                )

                _add_metric(
                    filesystem_used,
                    worker_label_values + [_safe_label(d.mount_point)],
                    d.used,
                )

                if d.total is not None and d.used is not None and d.total != 0:
                    _add_metric(
                        filesystem_utilization_rate,
                        worker_label_values + [_safe_label(d.mount_point)],
                        _rate(d.used, d.total),
                    )

        # system
        yield os_info
        yield kernel_info
        yield uptime
        yield cpu_cores
        yield cpu_utilization_rate
        yield memory_total
        yield memory_used
        yield memory_utilization_rate
        yield gpu_info
        yield gpu_cores
        yield gpu_utilization_rate
        yield gpu_temperature
        yield gpu_power_limit
        yield gpu_power_used
        yield gram_total
        yield gram_allocated
        yield gram_used
        yield gram_utilization_rate
        yield filesystem_total
        yield filesystem_used
        yield filesystem_utilization_rate

    def _load_worker_identity(self):
        if self._worker_uuid_getter is None or self._clientset_getter is None:
            return None
        worker_id = self._worker_id_getter()
        worker_uuid = self._worker_uuid_getter()
        if (
            type(worker_id) is not int
            or worker_id <= 0
            or not isinstance(worker_uuid, str)
            or not worker_uuid.strip()
        ):
            return None
        clientset = self._clientset_getter()
        if clientset is None:
            return None
        response = clientset.http_client.get_httpx_client().get(
            f'/workers/{worker_id}', timeout=2.0
        )
        response.raise_for_status()
        worker = response.json()
        if (
            worker.get('id') != worker_id
            or worker.get('me') is not True
            or 'owner_principal_id' not in worker
        ):
            return None
        owner = worker['owner_principal_id']
        if owner is not None and (type(owner) is not int or owner <= 0):
            return None
        return worker_id, worker_uuid, owner

    def refresh_worker_identity(self) -> None:
        """Refresh the server-verified owner outside the Prometheus scrape path."""
        snapshot = None
        try:
            identity = self._load_worker_identity()
            if identity is not None:
                snapshot = (*identity, time.monotonic() + HARDWARE_IDENTITY_TTL_SECONDS)
        except Exception as exc:
            logger.debug(
                'Stable hardware identity refresh failed (%s)', type(exc).__name__
            )
        with self._hardware_identity_lock:
            lost_identity = self._hardware_identity is not None and snapshot is None
            self._hardware_identity = snapshot
        if lost_identity:
            logger.warning('Stable hardware identity became unavailable')

    def _cached_worker_identity(self):
        current = None
        try:
            current = (self._worker_id_getter(), self._worker_uuid_getter())
        except Exception:
            pass
        with self._hardware_identity_lock:
            snapshot = self._hardware_identity
            if snapshot is None:
                return None
            if snapshot[:2] == current and snapshot[3] > time.monotonic():
                return snapshot[:3]
            self._hardware_identity = None
        logger.warning('Stable hardware identity became unavailable')
        return None

    def _collect_stable_hardware(self, status):
        """Qualify samples only with an unexpired server ownership snapshot."""
        identity = self._cached_worker_identity()
        if identity is None:
            return
        worker_id, worker_uuid, owner = identity

        labels = ['worker_id', 'worker_uuid', 'owner_principal_id']
        values = [str(worker_id), worker_uuid, str(owner or '')]
        host = (
            ('cpu_utilization_percent', status.cpu, 'utilization_rate', (0, 100)),
            ('memory_used_bytes', status.memory, 'used', (0, None)),
            ('memory_total_bytes', status.memory, 'total', (0, None)),
        )
        for name, obj, attribute, bounds in host:
            metric = _stable_gauge(
                'worker_host_' + name,
                'Owner-qualified whole worker ' + name,
                labels,
                values,
                getattr(obj, attribute, None),
                bounds,
            )
            if metric is not None:
                yield metric

        yield from _stable_filesystems(status.filesystem or [], labels, values)

        gpu_labels = labels + ['gpu_uuid', 'gpu_chip_index']
        identity_info = InfoMetricFamily(
            metric_name('worker_gpu_identity'),
            'Owner-qualified UUID/chip to reported device index mapping',
            labels=gpu_labels,
        )
        families = {}
        for device in verified_gpu_devices(status.gpu_devices or []):
            uuid, chip = gpu_identity(device)
            gpu_values = values + [uuid, str(chip)]
            identity_info.add_metric(gpu_values, {'gpu_index': str(device.index)})
            observations = [
                ('memory_used_bytes', device.memory, 'used', (0, None)),
                ('memory_total_bytes', device.memory, 'total', (0, None)),
            ]
            if not is_gpu_partition(device):
                observations.extend(
                    [
                        (
                            'utilization_percent',
                            device.core,
                            'utilization_rate',
                            (0, 100),
                        ),
                        ('power_watts', device, 'power_used', (0, None)),
                        ('temperature_celsius', device, 'temperature', (None, None)),
                    ]
                )
            for name, obj, attribute, bounds in observations:
                metric = _stable_gauge(
                    'worker_gpu_' + name,
                    'Owner-qualified device ' + name,
                    gpu_labels,
                    gpu_values,
                    getattr(obj, attribute, None),
                    bounds,
                )
                if metric is not None:
                    if name not in families:
                        families[name] = metric
                    else:
                        families[name].samples.extend(metric.samples)
        yield identity_info
        yield from families.values()

    def collect_runtime_metrics(self):
        yield from runtime_metrics_values(self._cache, "unified")

    def start(self):
        try:
            raw_collector = RawCollector(
                cache=self._cache,
            )
            raw_registry.register(raw_collector)
            unified_registry.register(self)

            # Start FastAPI server
            app = FastAPI(
                title="GPUStack Worker Metrics Exporter",
                response_model_exclude_unset=True,
                docs_url=None if self._config.disable_openapi_docs else "/docs",
                redoc_url=None if self._config.disable_openapi_docs else "/redoc",
                openapi_url=(
                    None if self._config.disable_openapi_docs else "/openapi.json"
                ),
            )

            @app.get("/metrics")
            def metrics():
                data = generate_latest(unified_registry)
                return Response(content=data, media_type=CONTENT_TYPE_LATEST)

            @app.get("/metrics/raw")
            def metrics_raw():
                data = generate_latest(raw_registry)
                return Response(content=data, media_type=CONTENT_TYPE_LATEST)

            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self._port,
                access_log=False,
                log_level="error",
            )

            setup_logging()
            logger.info(f"Serving metric exporter on {config.host}:{config.port}.")
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            logger.error(f"Failed to start metric exporter: {e}")


class RawCollector(Collector):
    def __init__(
        self,
        cache: dict = None,
    ):
        self._cache = cache

    def collect(self):
        # passthrough raw metrics from runtime and add gpustack related labels.
        yield from runtime_metrics_values(self._cache, "raw")


def _rate(used, total):
    return round(used / total, 6) * 100 if total != 0 else 0
