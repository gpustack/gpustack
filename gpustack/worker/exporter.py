from concurrent.futures import ThreadPoolExecutor
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
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.utils.name import metric_name
from gpustack.worker.collector import WorkerStatusCollector
import uvicorn
import logging
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)

unified_registry = CollectorRegistry()
raw_registry = CollectorRegistry()


class MetricExporter(Collector):
    _worker_ip_getter: Callable[[], str]
    _collector: WorkerStatusCollector

    def __init__(
        self,
        cfg: Config,
        worker_name: str,
        collector: WorkerStatusCollector,
        worker_ip_getter: Callable[[], str],
        worker_id_getter: Callable[[], str],
        cache: dict = None,
    ):
        self._collector = collector
        self._worker_name = worker_name
        self._worker_id_getter = worker_id_getter
        self._worker_ip_getter = worker_ip_getter
        self._port = cfg.worker_metrics_port
        self._cache = cache

    def collect(self):
        with ThreadPoolExecutor() as executor:
            worker_future = executor.submit(list, self.collect_worker_metrics())
            runtime_future = executor.submit(list, self.collect_runtime_metrics())
            for metric in worker_future.result():
                yield metric
            for metric in runtime_future.result():
                yield metric

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
        worker_ip = self._worker_ip_getter()
        worker_id = str(self._worker_id_getter())
        worker_name = self._worker_name
        worker_label_values = [worker_id, worker_name, worker_ip]
        try:
            worker = self._collector.collect()
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
                    "name": status.os.name,
                    "version": status.os.version,
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
                    "name": status.kernel.name,
                    "release": status.kernel.release,
                    "version": status.os.version,
                    "architecture": status.kernel.architecture,
                },
            )

        # uptime
        if status.uptime is not None:
            uptime.add_metric(worker_label_values, status.uptime.uptime)

        # cpu
        if status.cpu is not None:
            cpu_cores.add_metric(worker_label_values, status.cpu.total)
            cpu_utilization_rate.add_metric(
                worker_label_values, status.cpu.utilization_rate
            )

        # memory
        if status.memory is not None:
            memory_total.add_metric(worker_label_values, status.memory.total)
            memory_used.add_metric(worker_label_values, status.memory.used)

            if status.memory.total != 0 and status.memory.used is not None:
                memory_utilization_rate.add_metric(
                    worker_label_values,
                    _rate(status.memory.used, status.memory.total),
                )

        # gpu
        if status.gpu_devices is not None:
            for i, d in enumerate(status.gpu_devices):
                gpu_chip_index = "0"  # TODO(michelia): Placeholder, replace with actual chip index if available
                gpu_label_values = worker_label_values + [
                    str(i),
                    d.name,
                    gpu_chip_index,
                ]
                gpu_info.add_metric(
                    gpu_labels,
                    {
                        "worker_id": worker_id,
                        "worker_name": worker_name,
                        "instance": worker_ip,
                        "gpu_index": str(i),
                        "gpu_chip_index": gpu_chip_index,
                        "gpu_name": d.name,
                    },
                )
                if d.core is not None:
                    gpu_cores.add_metric(gpu_label_values, d.core.total)
                    gpu_utilization_rate.add_metric(
                        gpu_label_values,
                        d.core.utilization_rate,
                    )

                if d.temperature is not None:
                    gpu_temperature.add_metric(gpu_label_values, d.temperature)

                if d.memory is not None:
                    gram_total.add_metric(gpu_label_values, d.memory.total)
                    gram_allocated.add_metric(gpu_label_values, d.memory.allocated or 0)
                    gram_used.add_metric(gpu_label_values, d.memory.used)

                    if d.memory.total != 0 and d.memory.used is not None:
                        gram_utilization_rate.add_metric(
                            gpu_label_values,
                            _rate(d.memory.used, d.memory.total),
                        )

        # filesystem
        if status.filesystem is not None:
            for _, d in enumerate(status.filesystem):
                filesystem_total.add_metric(gpu_label_values + [d.mount_point], d.total)
                filesystem_used.add_metric(gpu_label_values + [d.mount_point], d.used)

                if d.total != 0 and d.used is not None:
                    filesystem_utilization_rate.add_metric(
                        gpu_label_values + [d.mount_point],
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
        yield gram_total
        yield gram_allocated
        yield gram_used
        yield gram_utilization_rate
        yield filesystem_total
        yield filesystem_used
        yield filesystem_utilization_rate

    def collect_runtime_metrics(self):
        if not self._cache or self._cache.get("unified") is None:
            return

        for _, prom_metric in self._cache["unified"].items():
            yield prom_metric

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
        if not self._cache or self._cache.get("raw") is None:
            return

        for _, prom_metric in self._cache["raw"].items():
            yield prom_metric


def _rate(used, total):
    return round(used / total, 6) * 100 if total != 0 else 0
