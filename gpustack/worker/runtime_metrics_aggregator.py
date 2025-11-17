from typing import Callable
from prometheus_client.core import (  # noqa: F401
    GaugeMetricFamily,
    InfoMetricFamily,
    HistogramMetricFamily,
    CounterMetricFamily,
    SummaryMetricFamily,
)
from prometheus_client import CollectorRegistry
from gpustack.client.generated_clientset import ClientSet
from gpustack.utils.command import find_parameter
from gpustack.utils.metrics import (
    get_builtin_metrics_config,
    get_runtime_metrics_config,
)
from gpustack.worker.runtime_metrics_client import (
    Config as RunTimeMetricsClientConfig,
)
from gpustack.worker.runtime_metrics_client import Client as RuntimeMetricsClient
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    get_backend,
    is_audio_model,
    is_image_model,
)
import logging
import uuid
from typing import Optional
from gpustack.utils import version


logger = logging.getLogger(__name__)

# unified registry
unified_registry = CollectorRegistry()

# raw metrics registry
raw_registry = CollectorRegistry()


class RuntimeMetricsAggregator:
    def __init__(
        self,
        cache: dict = None,
        worker_id_getter=Callable[[], int],
        clientset: ClientSet = None,
    ):
        self._cache = cache
        self._metrics_client_config = RunTimeMetricsClientConfig(
            timeout=5, max_retries=2, insecure_tls=True
        )
        self._metrics_client = RuntimeMetricsClient(self._metrics_client_config)
        self._worker_id_getter = worker_id_getter
        self._clientset = clientset

    def aggregate(self):
        """
        Fetch metrics from all model instances, normalize and aggregate both unified and raw metrics, and write results to cache.
        """
        worker_id = self._worker_id_getter()
        if not worker_id:
            logger.trace("Worker ID is not set. Skipping runtime metrics fetch.")
            return

        # 1. Get metrics config
        metrics_config = self._get_metrics_config()

        # 2. Get active model endpoints
        endpoints, endpoint_to_instance, instance_id_to_model = (
            self._find_active_model_endpoints(worker_id, metrics_config)
        )
        if not endpoints:
            logger.trace(
                "No valid endpoints found for model instances. Skipping runtime metrics fetch."
            )
            return

        trace_id = uuid.uuid4().hex[:8]
        logger.trace(
            f"trace_id: {trace_id}, fetching runtime metrics from {len(endpoints)} endpoints"
        )

        # 3. Batch fetch metrics from all endpoints
        endpoint_metrics = self._metrics_client.fetch_metrics_from_endpoints(endpoints)

        # 4. Unified and raw aggregation
        unified_metrics = {}
        raw_metrics = {}
        model_runtime_version = {}
        for ep, metrics in endpoint_metrics.items():
            if not metrics:
                continue
            mi = endpoint_to_instance[ep]
            m = instance_id_to_model.get(mi.id)
            runtime = get_backend(m)
            runtime_version = self._get_model_runtime_version(
                m, ep, model_runtime_version
            )
            base_labels = self._build_base_labels(mi, m, runtime)
            self._process_endpoint_metrics(
                metrics,
                base_labels,
                runtime,
                runtime_version,
                unified_metrics,
                raw_metrics,
                metrics_config,
            )

        self._cache["unified"] = unified_metrics
        self._cache["raw"] = raw_metrics
        logger.trace(f"trace_id: {trace_id}, completed fetching runtime metrics.")

    def _get_model_runtime_version(
        self, model: Model, endpoint: str, model_runtime_version: dict
    ) -> Optional[str]:
        if model.id in model_runtime_version:
            return model_runtime_version[model.id]

        version = self._metrics_client.fetch_runtime_version_from_endpoint(
            endpoint, model.backend
        )
        if version is not None:
            model_runtime_version[model.id] = version
            return version
        elif model.backend_version is not None:
            model_runtime_version[model.id] = model.backend_version
            return model.backend_version
        return None

    def _find_active_model_endpoints(self, worker_id: int, metrics_config: dict):
        """
        Get all endpoints and related mappings for RUNNING model instances on this worker.
        Returns: (endpoints, endpoint->instance, instance_id->model)
        """
        model_instances, models = self._list_worker_models(worker_id)
        if not model_instances or not models:
            return set(), {}, {}

        model_id_to_model = {m.id: m for m in models.items}
        endpoints = set()
        endpoint_to_instance = {}
        instance_id_to_model = {}
        for mi in model_instances.items:
            model = model_id_to_model.get(mi.model_id)

            if self._should_skip_endpoint(
                model=model,
                model_instance=mi,
                metrics_config=metrics_config,
            ):
                logger.trace(f"Skipping model instance {mi.id} in metrics aggregation.")
                continue

            endpoint = f"{mi.worker_ip}:{mi.ports[0]}"
            endpoints.add(endpoint)
            endpoint_to_instance[endpoint] = mi
            instance_id_to_model[mi.id] = model

        return endpoints, endpoint_to_instance, instance_id_to_model

    def _list_worker_models(self, worker_id: int):
        """
        Query all model instances and model objects on this worker.
        """
        model_instances = self._clientset.model_instances.list(
            params={"worker_id": str(worker_id)}
        )
        models = self._clientset.models.list()
        return model_instances, models

    def _build_base_labels(self, mi, m, runtime):
        """
        Build base labels for each metric.
        """
        return {
            "worker_id": str(mi.worker_id) if mi.worker_id else "",
            "worker_name": mi.worker_name if mi.worker_name else "",
            "model_id": str(m.id) if m else "",
            "model_name": m.name if m else "",
            "model_instance_id": str(mi.id),
            "model_instance_name": mi.name,
            "runtime": runtime,
        }

    def _process_endpoint_metrics(
        self,
        metrics,
        base_labels,
        runtime,
        runtime_version,
        unified_metrics,
        raw_metrics,
        metrics_config,
    ):
        """
        Process metrics for a single endpoint, aggregate to unified and raw.
        """
        for source_family_name, family in metrics.items():
            first_sample = family.samples[0] if family.samples else None
            if not first_sample:
                continue

            label_keys = list(base_labels.keys())
            for k in first_sample.labels.keys():
                if k not in label_keys:
                    label_keys.append(k)

            # raw metrics
            if source_family_name not in raw_metrics:
                raw_metrics[source_family_name] = create_prom_metric_family(
                    name=source_family_name,
                    type=family.type,
                    description=family.documentation,
                    labels=label_keys,
                )
            raw_family = raw_metrics[source_family_name]

            # unified metrics
            unified_family = None
            unified_metric_family_name = get_unified_metric_family_name(
                metrics_config, source_family_name, runtime, runtime_version
            )
            if unified_metric_family_name:
                cfg = get_unified_metric_family_config(
                    metrics_config, unified_metric_family_name
                )
                if cfg:
                    if unified_metric_family_name not in unified_metrics:
                        unified_metrics[unified_metric_family_name] = (
                            create_prom_metric_family(
                                name=unified_metric_family_name,
                                type=cfg.get("type"),
                                description=cfg.get("description"),
                                labels=label_keys,
                            )
                        )
                    unified_family = unified_metrics[unified_metric_family_name]

            for sample in family.samples:
                label_values = [
                    (
                        base_labels.get(k, sample.labels.get(k, ""))
                        if k in base_labels
                        else sample.labels.get(k, "")
                    )
                    for k in label_keys
                ]
                labels = sample.labels.copy()
                labels.update(base_labels)

                if family.type in ("histogram", "summary"):
                    raw_family.add_sample(
                        name=sample.name,
                        labels=labels,
                        value=sample.value,
                        timestamp=sample.timestamp,
                    )
                    if unified_family:
                        new_name = sample.name.replace(
                            source_family_name, unified_metric_family_name
                        )
                        unified_family.add_sample(
                            name=new_name,
                            labels=labels,
                            value=sample.value,
                            timestamp=sample.timestamp,
                        )
                else:
                    raw_family.add_metric(
                        labels=label_values,
                        value=sample.value,
                        timestamp=sample.timestamp,
                    )
                    if unified_family:
                        unified_family.add_metric(
                            labels=label_values,
                            value=sample.value,
                            timestamp=sample.timestamp,
                        )

    def _should_skip_endpoint(
        self, model: Model, model_instance: ModelInstance, metrics_config: dict
    ) -> bool:
        # skip image and audio models
        if is_image_model(model) or is_audio_model(model):
            return True

        # model and model instance must be valid
        if (
            model_instance.state != ModelInstanceStateEnum.RUNNING
            or model_instance.worker_ip is None
            or not model_instance.ports
        ):
            return True

        if not model:
            return True

        runtime = model.backend
        if not runtime:
            return True

        # check runtime metrics config
        runtime_cfg = get_runtime_metrics_config(metrics_config, runtime)
        if not runtime_cfg:
            return True

        # check runtime-specific metrics flags
        if runtime == BackendEnum.VLLM:
            disable_metrics = find_parameter(
                model.backend_parameters, ["disable-log-stats"]
            )
            if disable_metrics:
                return True

        if model.env and model.env.get("GPUSTACK_DISABLE_METRICS"):
            return True

        return False

    def _get_online_metrics_config(self):
        try:
            resp = self._clientset.http_client.get_httpx_client().get(
                f"{self._clientset.base_url}/metrics/config", timeout=5
            )
            if resp.status_code == 404:
                return None
            elif resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch online metrics config, status: {resp.status_code}"
                )
                return None

            data = resp.json()
            if not isinstance(data, dict):
                logger.warning(
                    "Online metrics config is not a dict, fallback to builtin config."
                )
                return None

            return data
        except Exception as e:
            logger.error(f"Error fetching online metrics config: {e}")
            return None

    def _get_metrics_config(self):
        online_config = self._get_online_metrics_config()
        if online_config:
            return online_config
        else:
            return get_builtin_metrics_config()


_METRIC_FAMILY_CLASS = {
    "gauge": GaugeMetricFamily,
    "info": InfoMetricFamily,
    "histogram": HistogramMetricFamily,
    "counter": CounterMetricFamily,
    "summary": SummaryMetricFamily,
}


def create_prom_metric_family(type: str, name: str, description: str, labels=None):
    cls = _METRIC_FAMILY_CLASS.get(str(type).lower())
    if not cls:
        raise ValueError(f"Unknown metric family type: {type}")
    if labels is not None:
        return cls(name, description, labels=labels)
    else:
        return cls(name, description)


def get_unified_metric_family_name(
    config: dict,
    source_metric_family_name: str,
    runtime: str,
    runtime_version: Optional[str],
) -> Optional[str]:
    """
    Return the unified (normalized) metric family name as a string. If not found, return an empty string.
    Prefer version-specific mapping if matched, otherwise use the default '*'.
    """
    runtime_cfg = get_runtime_metrics_config(config, runtime)
    if not runtime_cfg:
        return None

    name = runtime_cfg.get("*", {}).get(source_metric_family_name, None)
    if runtime_version:
        is_valid_version = version.is_valid_version_str(runtime_version)
        for ver_range, mapping in runtime_cfg.items():
            if ver_range == "*":
                continue
            if (is_valid_version and version.in_range(runtime_version, ver_range)) or (
                not is_valid_version and runtime_version == ver_range
            ):
                old_version_name = mapping.get(source_metric_family_name)
                if old_version_name is not None:
                    return old_version_name

    return name


def get_unified_metric_family_config(
    config: dict, unified_metric_family_name: str
) -> dict:
    return config.get("gpustack_metrics", {}).get(unified_metric_family_name, {})
