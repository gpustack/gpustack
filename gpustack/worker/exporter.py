from prometheus_client.registry import Collector
from prometheus_client import make_asgi_app, REGISTRY
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from gpustack.client.generated_clientset import ClientSet
from gpustack.logging import setup_logging
from gpustack.worker.collector import WorkerStatusCollector
import uvicorn
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class MetricExporter(Collector):
    _provider = "gpustack"

    def __init__(
        self, worker_ip: str, worker_name: str, port: int, clientset: ClientSet
    ):
        self._worker_ip = worker_ip
        self._worker_name = worker_name
        self._collector = WorkerStatusCollector(
            worker_ip, worker_name=worker_name, clientset=clientset
        )
        self._port = port

    def collect(self):  # noqa: C901
        labels = ["instance", "provider"]
        filesystem_labels = ["instance", "provider", "mountpoint"]
        gpu_labels = ["instance", "provider", "index"]

        # metrics
        os_info = InfoMetricFamily("worker_node_os", "Operating system information")
        kernel_info = InfoMetricFamily("worker_node_kernel", "Kernel information")
        uptime = GaugeMetricFamily(
            "worker_node_uptime_seconds",
            "Uptime in seconds of the worker node",
            labels=labels,
        )
        cpu_cores = GaugeMetricFamily(
            "worker_node_cpu_cores",
            "Total CPUs cores of the worker node",
            labels=labels,
        )
        cpu_utilization_rate = GaugeMetricFamily(
            "worker_node_cpu_utilization_rate",
            "Rate of CPU utilization on the worker node",
            labels=labels,
        )
        memory_total = GaugeMetricFamily(
            "worker_memory_total_bytes",
            "Total memory in bytes of the worker node",
            labels=labels,
        )
        memory_used = GaugeMetricFamily(
            "worker_node_memory_used_bytes",
            "Memory used in bytes of the worker node",
            labels=labels,
        )
        memory_utilization_rate = GaugeMetricFamily(
            "worker_node_memory_utilization_rate",
            "Rate of memory utilization on the worker node",
            labels=labels,
        )
        gpu_info = InfoMetricFamily("worker_node_gpu", "GPU information")
        gpu_cores = GaugeMetricFamily(
            "worker_node_gpu_cores",
            "Total GPUs cores of the worker node",
            labels=gpu_labels,
        )
        gpu_utilization_rate = GaugeMetricFamily(
            "worker_node_gpu_utilization_rate",
            "Rate of GPU utilization on the worker node",
            labels=gpu_labels,
        )
        gpu_temperature = GaugeMetricFamily(
            "worker_node_gpu_temperature_celsius",
            "GPU temperature in celsius of the worker node",
            labels=gpu_labels,
        )
        gram_total = GaugeMetricFamily(
            "worker_node_gram_total_bytes",
            "Total GPU RAM in bytes of the worker node",
            labels=gpu_labels,
        )
        gram_used = GaugeMetricFamily(
            "worker_node_gram_used_bytes",
            "GPU RAM used in bytes of the worker node",
            labels=gpu_labels,
        )
        gram_utilization_rate = GaugeMetricFamily(
            "worker_node_gram_utilization_rate",
            "Rate of GPU RAM utilization on the worker node",
            labels=gpu_labels,
        )
        filesystem_total = GaugeMetricFamily(
            "worker_node_filesystem_total_bytes",
            "Total filesystem in bytes of the worker node",
            labels=filesystem_labels,
        )
        filesystem_used = GaugeMetricFamily(
            "worker_node_filesystem_used_bytes",
            "Total filesystem used in bytes of the worker node",
            labels=filesystem_labels,
        )
        filesystem_utilization_rate = GaugeMetricFamily(
            "worker_node_filesystem_utilization_rate",
            "Rate of filesystem utilization on the worker node",
            labels=filesystem_labels,
        )

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
                ["instance", "provider", "name", "version"],
                {
                    "instance": self._worker_ip,
                    "provider": self._provider,
                    "name": status.os.name,
                    "version": status.os.version,
                },
            )

        # kernel
        if status.kernel is not None:
            kernel_info.add_metric(
                ["instance", "provider", "name", "version"],
                {
                    "instance": self._worker_ip,
                    "provider": self._provider,
                    "name": status.kernel.name,
                    "release": status.kernel.release,
                    "version": status.os.version,
                    "architecture": status.kernel.architecture,
                },
            )

        # uptime
        if status.uptime is not None:
            uptime.add_metric([self._worker_ip, self._provider], status.uptime.uptime)

        # cpu
        if status.cpu is not None:
            cpu_cores.add_metric([self._worker_ip, self._provider], status.cpu.total)
            cpu_utilization_rate.add_metric(
                [self._worker_ip, self._provider], status.cpu.utilization_rate
            )

        # memory
        if status.memory is not None:
            memory_total.add_metric(
                [self._worker_ip, self._provider], status.memory.total
            )
            memory_used.add_metric(
                [self._worker_ip, self._provider], status.memory.used
            )
            memory_utilization_rate.add_metric(
                [self._worker_ip, self._provider],
                _rate(status.memory.used, status.memory.total),
            )

        # gpu
        if status.gpu_devices is not None:
            for i, d in enumerate(status.gpu_devices):
                gpu_info.add_metric(
                    ["instance", "provider", "index", "name"],
                    {
                        "instance": self._worker_ip,
                        "provider": self._provider,
                        "index": str(i),
                        "name": d.name,
                    },
                )
                gpu_cores.add_metric(
                    [self._worker_ip, self._provider, str(i)], d.core.total
                )
                gpu_utilization_rate.add_metric(
                    [self._worker_ip, self._provider, str(i)], d.core.utilization_rate
                )
                gpu_temperature.add_metric(
                    [self._worker_ip, self._provider, str(i)], d.temperature
                )
                gram_total.add_metric(
                    [self._worker_ip, self._provider, str(i)], d.memory.total
                )
                gram_used.add_metric(
                    [self._worker_ip, self._provider, str(i)], d.memory.used
                )
                gram_utilization_rate.add_metric(
                    [self._worker_ip, self._provider, str(i)],
                    _rate(d.memory.used, d.memory.total),
                )

        # filesystem
        if status.filesystem is not None:
            for _, d in enumerate(status.filesystem):
                filesystem_total.add_metric(
                    [self._worker_ip, self._provider, d.mount_point], d.total
                )
                filesystem_used.add_metric(
                    [self._worker_ip, self._provider, d.mount_point], d.used
                )
                filesystem_utilization_rate.add_metric(
                    [self._worker_ip, self._provider, d.mount_point],
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
        yield gram_used
        yield gram_utilization_rate
        yield filesystem_total
        yield filesystem_used
        yield filesystem_utilization_rate

    def start(self):
        try:
            REGISTRY.register(self)

            # Start FastAPI server
            metrics_app = make_asgi_app()

            app = FastAPI(title="GPUStack Worker", response_model_exclude_unset=True)
            app.mount("/metrics", metrics_app)

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


def _rate(used, total):
    return round(used / total, 6) * 100 if total != 0 else 0
