import asyncio
import re
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client.registry import Collector
from prometheus_client.core import (
    GaugeMetricFamily,
    InfoMetricFamily,
)
import uvicorn
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine
from gpustack.server.deps import SessionDep
from gpustack.utils.name import metric_name
import logging
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import FastAPI, Response


logger = logging.getLogger(__name__)

# Prometheus label name pattern
# https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
label_name_pattern = r'^[a-zA-Z_:][a-zA-Z0-9_:]*$'


class MetricExporter(Collector):

    def __init__(self, cfg: Config):
        self._engine = get_engine()
        self._cache_metrics = []
        self._port = cfg.metrics_port

    def collect(self):
        for metric in self._cache_metrics:
            yield metric

    async def generate_metrics_cache(self):
        while True:
            async with AsyncSession(self._engine) as session:
                self._cache_metrics = await self._collect_metrics(session)
            await asyncio.sleep(3)

    async def _collect_metrics(self, session: AsyncSession):
        cluster_labels = ["cluster_id", "cluster_name"]
        worker_labels = cluster_labels + ["worker_id", "worker_name"]
        model_labels = cluster_labels + ["model_id", "model_name"]
        model_instance_labels = worker_labels + [
            "model_id",
            "model_name",
            "model_instance_name",
        ]

        # cluster metrics
        cluster_info = InfoMetricFamily(metric_name("cluster"), "Cluster information")
        cluster_status = GaugeMetricFamily(
            metric_name("cluster_status"),
            "Cluster status",
            labels=cluster_labels + ["state"],
        )

        # worker metrics
        worker_info = InfoMetricFamily(metric_name("worker"), "Worker information")
        worker_status = GaugeMetricFamily(
            metric_name("worker_status"),
            "Worker status",
            labels=worker_labels + ["state"],
        )

        # model metrics
        model_info = InfoMetricFamily(metric_name("model"), "Model information")
        model_desired_instances = GaugeMetricFamily(
            metric_name("model_desired_instances"),
            "Desired instances of the model",
            labels=model_labels,
        )
        model_running_instances = GaugeMetricFamily(
            metric_name("model_running_instances"),
            "Running instances of the model",
            labels=model_labels,
        )
        model_instance_status = GaugeMetricFamily(
            metric_name("model_instance_status"),
            "Model instance status",
            labels=model_instance_labels + ["state"],
        )

        metrics = [
            cluster_info,
            cluster_status,
            worker_info,
            worker_status,
            model_info,
            model_desired_instances,
            model_running_instances,
            model_instance_status,
        ]

        # cluster metrics
        cluster_id_to_name = {}
        model_id_to_name = {}
        model_id_to_cluster_id = {}
        clusters = await Cluster.all(session)

        for cluster in clusters:
            cluster_id_to_name[str(cluster.id)] = cluster.name
            cluster_label_values = [str(cluster.id), cluster.name]

            cluster_info.add_metric(
                cluster_labels + ["provider"],
                {
                    "cluster_id": str(cluster.id),
                    "cluster_name": cluster.name,
                    "provider": str(cluster.provider),
                },
            )

            cluster_status.add_metric(
                cluster_label_values + [cluster.state],
                1,
            )

            # worker metrics
            workers = cluster.cluster_workers
            for worker in workers:
                worker_label_values = cluster_label_values + [
                    str(worker.id),
                    worker.name,
                    worker.state,
                ]

                worker_dynamic_label_keys = []
                worker_info_metric_values = {
                    "cluster_id": str(cluster.id),
                    "cluster_name": cluster.name,
                    "worker_id": str(worker.id),
                    "worker_name": worker.name,
                }
                for k, v in (worker.labels or {}).items():
                    if not re.match(label_name_pattern, k):
                        continue
                    worker_dynamic_label_keys.append(k)
                    worker_info_metric_values[k] = v

                worker_info.add_metric(
                    worker_labels + worker_dynamic_label_keys,
                    worker_info_metric_values,
                )

                worker_status.add_metric(
                    worker_label_values,
                    1,
                )

            # model metrics
            models = cluster.cluster_models
            for model in models:
                model_id_to_name[str(model.id)] = model.name
                model_id_to_cluster_id[str(model.id)] = str(cluster.id)

                model_label_values = cluster_label_values + [
                    str(model.id),
                    model.name,
                ]

                model_info.add_metric(
                    model_labels
                    + ["runtime", "runtime_version", "source", "source_key"],
                    {
                        "cluster_id": str(cluster.id),
                        "cluster_name": cluster.name,
                        "model_id": str(model.id),
                        "model_name": model.name,
                        "runtime": model.backend,
                        "runtime_version": model.backend_version or "unknown",
                        "source": model.source,
                        "source_key": model.model_source_key,
                    },
                )

                model_desired_instances.add_metric(
                    model_label_values,
                    model.replicas,
                )

                model_running_instances.add_metric(
                    model_label_values,
                    model.ready_replicas,
                )

                # instance metrics
                instances = model.instances
                for mi in instances:
                    worker_id = str(mi.worker_id) if mi.worker_id else "unknown"
                    worker_name = mi.worker_name if mi.worker_name else "unknown"
                    mi_label_values = cluster_label_values + [
                        worker_id,
                        worker_name,
                        str(model.id),
                        model.name,
                        mi.name,
                        mi.state,
                    ]
                    model_instance_status.add_metric(
                        mi_label_values,
                        1,
                    )

        # return all metrics
        return metrics

    async def start(self):
        try:
            REGISTRY.register(self)

            # Start FastAPI server
            app = FastAPI(
                title="GPUStack Metrics Exporter", response_model_exclude_unset=True
            )

            @app.get("/metrics")
            def metrics():
                data = generate_latest(REGISTRY)
                return Response(content=data, media_type=CONTENT_TYPE_LATEST)

            @app.get("/metrics/targets")
            async def metrics_targets(session: SessionDep):

                targets = []
                worker_list = await Worker.all(session=session)
                cluster_workers = {}
                for worker in worker_list:
                    if (
                        worker.state == WorkerStateEnum.READY
                        and worker.metrics_port
                        and worker.metrics_port > 0
                    ):
                        key = (worker.cluster_id, worker.cluster.name)
                        if key not in cluster_workers:
                            cluster_workers[key] = []
                        cluster_workers[key].append(
                            f"{worker.advertise_address}:{worker.metrics_port}"
                        )
                for (cluster_id, cluster_name), endpoints in cluster_workers.items():
                    targets.append(
                        {
                            "labels": {
                                "cluster_id": str(cluster_id),
                                "cluster_name": cluster_name,
                            },
                            "targets": endpoints,
                        }
                    )

                return targets

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
            await server.serve()
        except Exception as e:
            logger.error(f"Failed to start metric exporter: {e}")
