import asyncio
import re
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client.registry import Collector
from prometheus_client.core import (
    GaugeMetricFamily,
    InfoMetricFamily,
)
import uvicorn
from gpustack.config.config import Config
from gpustack.exporter.bus_metrics import BusMetricsCollector
from gpustack.logging import setup_logging
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceEndpoint,
    CacheServiceInstance,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.schemas.config import ModelInstanceProxyModeEnum
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import CategoryEnum, Model
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep
from gpustack.utils.name import metric_name
import logging
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)

# Prometheus label name pattern
# https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
label_name_pattern = r'^[a-zA-Z_:][a-zA-Z0-9_:]*$'

DEFAULT_METRICS_PATH = "/metrics"


class MetricExporter(Collector):

    def __init__(self, cfg: Config):
        self._cache_metrics = []
        self._port = cfg.metrics_port

    def collect(self):
        for metric in self._cache_metrics:
            yield metric

    async def generate_metrics_cache(self):
        while True:
            try:
                async with async_session() as session:
                    self._cache_metrics = await self._collect_metrics(session)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient DB error here (e.g. a pool-exhaustion timeout)
                # must not escape the loop -- an unhandled exception propagates
                # through the server's asyncio.gather and takes the whole
                # process down. Keep the last cache, log, and retry next tick.
                logger.exception("Failed to refresh metrics cache")
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
        model_instance_restart_count = GaugeMetricFamily(
            metric_name("model_instance_restart_count"),
            "Model instance restart count",
            labels=model_instance_labels,
        )
        model_instance_latest_restart_time = GaugeMetricFamily(
            metric_name("model_instance_latest_restart_time"),
            "Model instance latest restart time as Unix timestamp seconds",
            labels=model_instance_labels,
        )

        # Which deployments attach to which shared cache service: the
        # relation joins engine-side series (labeled by model) with
        # cache-service series (labeled by service) in dashboards.
        cache_service_attached_model = GaugeMetricFamily(
            metric_name("cache_service_attached_model"),
            "Model attached to a shared cache service",
            labels=cluster_labels
            + ["cache_service_id", "cache_service_name", "model_id", "model_name"],
        )
        # Degraded attachment is the expensive silent failure of shared
        # caching (the engine runs, hit rate is just forever zero), so
        # it gets a first-class signal: 0 = running without the cache.
        model_instance_cache_attached = GaugeMetricFamily(
            metric_name("model_instance_cache_attached"),
            "Whether a shared-cache instance runs with its cache injected "
            "and the recorded endpoint still served (0 = degraded: "
            "started without the shared KV cache, or the cache has since "
            "gone away)",
            labels=model_instance_labels,
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
            model_instance_restart_count,
            model_instance_latest_restart_time,
            cache_service_attached_model,
            model_instance_cache_attached,
        ]

        cache_service_names = {
            service.id: service.name
            for service in await CacheService.all_by_fields(
                session,
                fields={},
                extra_conditions=[CacheService.deleted_at.is_(None)],
            )
        }

        # cluster metrics
        cluster_id_to_name = {}
        model_id_to_name = {}
        model_id_to_cluster_id = {}
        clusters = await Cluster.all(
            session,
            options=[
                selectinload(Cluster.cluster_workers),
                selectinload(Cluster.cluster_models).selectinload(Model.instances),
            ],
        )

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

                # NOTE: Model.categories is a list, but Prometheus labels are
                # scalar. GPUStack currently treats the first entry as the
                # primary category for metrics, so secondary categories are not
                # exposed in gpustack:model_info. This keeps one model_info
                # series per model for model_id joins.
                category = (
                    model.categories[0]
                    if model.categories
                    else CategoryEnum.UNKNOWN.value
                )

                model_info.add_metric(
                    model_labels
                    + [
                        "runtime",
                        "runtime_version",
                        "source",
                        "source_key",
                        "category",
                    ],
                    {
                        "cluster_id": str(cluster.id),
                        "cluster_name": cluster.name,
                        "model_id": str(model.id),
                        "model_name": model.name,
                        "runtime": model.backend,
                        "runtime_version": model.backend_version or "unknown",
                        "source": model.source,
                        "source_key": model.model_source_key,
                        "category": category,
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

                kv_cache = model.extended_kv_cache
                if (
                    kv_cache
                    and kv_cache.is_shared()
                    and kv_cache.cache_service_id in cache_service_names
                ):
                    cache_service_attached_model.add_metric(
                        cluster_label_values
                        + [
                            str(kv_cache.cache_service_id),
                            cache_service_names[kv_cache.cache_service_id],
                            str(model.id),
                            model.name,
                        ],
                        1,
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
                    ]
                    model_instance_status.add_metric(
                        mi_label_values + [mi.state],
                        1,
                    )
                    model_instance_restart_count.add_metric(
                        mi_label_values,
                        mi.restart_count or 0,
                    )
                    model_instance_latest_restart_time.add_metric(
                        mi_label_values,
                        (
                            mi.last_restart_time.timestamp()
                            if mi.last_restart_time
                            else 0
                        ),
                    )
                    if kv_cache and kv_cache.is_shared():
                        cache_config = getattr(mi, "cache_config", None)
                        attached = bool(
                            cache_config
                            and cache_config.injected
                            # endpoint_live tracks the present; None means
                            # never evaluated and reads as live
                            and cache_config.endpoint_live is not False
                        )
                        model_instance_cache_attached.add_metric(
                            mi_label_values,
                            1 if attached else 0,
                        )

        # return all metrics
        return metrics

    async def start(self):
        try:
            REGISTRY.register(self)
            REGISTRY.register(BusMetricsCollector())

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
                return await _metrics_targets(session, is_proxy=False)

            @app.get("/metrics/proxy-targets")
            async def metrics_proxy_targets(session: SessionDep):
                return await _metrics_targets(session, is_proxy=True)

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


async def _metrics_targets(session: AsyncSession, is_proxy: bool):
    """Prometheus HTTP SD target list. The direct list (is_proxy=False)
    holds targets the Prometheus server reaches on its own network; the
    proxy list holds targets behind the server's tunnel proxy."""
    targets = []
    worker_list = await Worker.all(
        session=session, options=[selectinload(Worker.cluster)]
    )
    cluster_workers = {}
    for worker in worker_list:
        preferred_address = worker.advertise_address if not is_proxy else worker.ip
        if (
            worker.state == WorkerStateEnum.READY
            and worker.metrics_port
            and worker.metrics_port > 0
            and (is_proxy == (worker.proxy_mode == ModelInstanceProxyModeEnum.TUNNEL))
        ):
            key = (worker.cluster_id, worker.cluster.name)
            if key not in cluster_workers:
                cluster_workers[key] = []
            cluster_workers[key].append(f"{preferred_address}:{worker.metrics_port}")
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

    targets.extend(await _cache_service_targets(session, worker_list, is_proxy))

    return targets


def _normalize_metrics_path(path: Optional[str]) -> str:
    path = path or DEFAULT_METRICS_PATH
    if not path.startswith("/"):
        path = "/" + path
    return path


def _external_metrics_address(
    endpoint: Optional[CacheServiceEndpoint], provider_path: str
) -> Optional[Tuple[str, str, str]]:
    """(host:port, metrics path, scheme) of an external cache service's
    metrics endpoint. metrics_url takes precedence over host+metrics_port,
    mirroring the server-side metrics collector."""
    if endpoint is None:
        return None
    if endpoint.metrics_url:
        parsed = urlsplit(endpoint.metrics_url)
        if not parsed.hostname:
            return None
        scheme = parsed.scheme or "http"
        port = parsed.port or (443 if scheme == "https" else 80)
        return (
            f"{parsed.hostname}:{port}",
            _normalize_metrics_path(parsed.path),
            scheme,
        )
    if endpoint.host and endpoint.metrics_port:
        return f"{endpoint.host}:{endpoint.metrics_port}", provider_path, "http"
    return None


async def _cache_service_targets(
    session: AsyncSession, workers: List[Worker], is_proxy: bool
) -> List[dict]:
    """HTTP SD target groups for running cache services whose provider
    declares a metrics endpoint. Managed services are scraped per running
    instance on the instance's worker, so each instance follows its own
    worker's proxy split; external services are reachable from the server
    network and only appear on the direct target list."""
    services = await CacheService.all_by_fields(
        session,
        fields={"state": CacheServiceStateEnum.RUNNING},
        extra_conditions=[CacheService.deleted_at.is_(None)],
    )
    if not services:
        return []

    workers_by_id = {worker.id: worker for worker in workers}
    clusters = await Cluster.all(session)
    cluster_names = {cluster.id: cluster.name for cluster in clusters}

    instances_by_service: dict = {}
    if any(service.mode == CacheServiceModeEnum.MANAGED for service in services):
        instances = await CacheServiceInstance.all_by_fields(
            session, fields={"state": CacheServiceStateEnum.RUNNING}
        )
        for instance in instances:
            instances_by_service.setdefault(instance.cache_service_id, []).append(
                instance
            )

    groups = []
    for service in services:
        provider = get_cache_provider(service.provider_name)
        # Fields flagged metrics_target carry extra scrape endpoints (e.g.
        # an L2 storage cluster's exporter); they are independent of the
        # provider's own metrics declaration.
        if not is_proxy:
            groups.extend(
                _extra_metrics_target_groups(service, provider, cluster_names)
            )
        metrics = provider.metrics_for(service.provider_version) if provider else None
        if metrics is None:
            continue
        provider_path = _normalize_metrics_path(metrics.path)

        if service.mode == CacheServiceModeEnum.MANAGED:
            groups.extend(
                _managed_cache_service_groups(
                    service,
                    instances_by_service.get(service.id, []),
                    workers_by_id,
                    is_proxy,
                    provider_path,
                    cluster_names,
                )
            )
            continue

        if is_proxy:
            continue
        resolved = _external_metrics_address(service.endpoint, provider_path)
        if resolved is None:
            continue
        target, path, scheme = resolved

        labels = _cache_service_labels(service, cluster_names)
        if path != DEFAULT_METRICS_PATH:
            labels["__metrics_path__"] = path
        if scheme != "http":
            labels["__scheme__"] = scheme
        groups.append({"labels": labels, "targets": [target]})
    return groups


def _parse_scrape_address(value) -> Optional[Tuple[str, str, str]]:
    """(host:port, metrics path, scheme) parsed from a metrics_target
    field value: a full URL, or host:port with an optional path. The port
    defaults per scheme when omitted, mirroring _external_metrics_address.
    Non-HTTP schemes yield None — Prometheus cannot scrape them."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if "//" not in raw:
        raw = f"http://{raw}"
    parsed = urlsplit(raw)
    if not parsed.hostname:
        return None
    scheme = parsed.scheme or "http"
    if scheme not in ("http", "https"):
        return None
    port = parsed.port or (443 if scheme == "https" else 80)
    return (
        f"{parsed.hostname}:{port}",
        _normalize_metrics_path(parsed.path),
        scheme,
    )


def _extra_metrics_target_groups(
    service: CacheService,
    provider,
    cluster_names: dict,
) -> List[dict]:
    """Target groups from provider fields flagged metrics_target: extra
    Prometheus endpoints (e.g. an L2 storage cluster's exporter) the user
    filled into the service config. They live outside the workers'
    networks, so they only appear on the direct target list. Targets fed
    by an L2 backend field carry a cache_l2_backend label."""
    if provider is None:
        return []
    groups: List[dict] = []

    def add_target(value, extra_labels: dict):
        resolved = _parse_scrape_address(value)
        if resolved is None:
            return
        target, path, scheme = resolved
        labels = _cache_service_labels(service, cluster_names)
        labels.update(extra_labels)
        if path != DEFAULT_METRICS_PATH:
            labels["__metrics_path__"] = path
        if scheme != "http":
            labels["__scheme__"] = scheme
        groups.append({"labels": labels, "targets": [target]})

    endpoint_params = service.endpoint.params if service.endpoint else {}
    for field in provider.external_fields:
        if field.metrics_target and endpoint_params.get(field.name):
            add_target(endpoint_params[field.name], {})

    storages = (service.config.l2_storages if service.config else None) or []
    for storage in storages:
        backend = provider.l2_backends.get(storage.backend)
        if backend is None:
            continue
        for field in backend.fields:
            if field.metrics_target and (storage.params or {}).get(field.name):
                add_target(
                    storage.params[field.name],
                    {"cache_l2_backend": storage.backend},
                )

    return groups


def _managed_cache_service_groups(
    service: CacheService,
    instances: List[CacheServiceInstance],
    workers_by_id: dict,
    is_proxy: bool,
    provider_path: str,
    cluster_names: dict,
) -> List[dict]:
    """One target group per running instance with an allocated metrics
    port. Each group carries worker_name and cache_service_instance_id
    labels so per-instance series stay distinguishable."""
    groups = []
    for instance in instances:
        if not instance.metrics_port or instance.metrics_port <= 0:
            continue
        worker = workers_by_id.get(instance.worker_id)
        if worker is None:
            continue
        if is_proxy != (worker.proxy_mode == ModelInstanceProxyModeEnum.TUNNEL):
            continue
        address = worker.advertise_address if not is_proxy else worker.ip
        if not address:
            continue
        labels = _cache_service_labels(service, cluster_names)
        labels["worker_name"] = worker.name
        labels["cache_service_instance_id"] = str(instance.id)
        if provider_path != DEFAULT_METRICS_PATH:
            labels["__metrics_path__"] = provider_path
        groups.append(
            {
                "labels": labels,
                "targets": [f"{address}:{instance.metrics_port}"],
            }
        )
    return groups


def _cache_service_labels(service: CacheService, cluster_names: dict) -> dict:
    return {
        "cluster_id": str(service.cluster_id),
        "cluster_name": cluster_names.get(service.cluster_id, "unknown"),
        "cache_service_id": str(service.id),
        "cache_service_name": service.name,
        "provider": service.provider_name,
        "gpustack_target_type": "cache-service",
    }
