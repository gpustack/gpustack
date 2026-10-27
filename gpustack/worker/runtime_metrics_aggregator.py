from typing import Callable, Tuple
from cachetools import TTLCache
from prometheus_client.core import (  # noqa: F401
    GaugeMetricFamily,
    InfoMetricFamily,
    HistogramMetricFamily,
    CounterMetricFamily,
    SummaryMetricFamily,
    UnknownMetricFamily,
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
    ModelInstanceUpdate,
    get_backend,
    is_audio_model,
    is_image_model,
)
import logging
import uuid
from typing import Optional
from gpustack.utils import version

logger = logging.getLogger(__name__)

METRICS_CONFIG_FETCH_TIMEOUT_SECONDS = 30

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

        # Cache for metrics config (refresh every 300 seconds)
        self._metrics_config_cache = TTLCache(maxsize=1, ttl=300)

        # Failures already reported, so a persistent one is not logged every
        # pass. Bounded and expiring, since endpoints and families come and go
        # over the lifetime of a worker.
        self._warned = TTLCache(maxsize=1024, ttl=3600)

    def _warn_once(self, key, message: str):
        """
        Warn the first time a failure shows up and stay quiet while it persists.

        Aggregation runs every few seconds, so warning unconditionally would
        repeat the same line hundreds of times an hour for a single bad family.
        The key is dropped once that family or endpoint succeeds again, and
        expires on its own after an hour, so a lasting failure is still
        reported from time to time rather than going quiet for good.
        """
        if key in self._warned:
            logger.debug(message, exc_info=True)
            return
        self._warned[key] = True
        logger.warning(message, exc_info=logger.isEnabledFor(logging.DEBUG))

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
            # No active instances left on this worker (e.g. the last one was
            # removed). Clear the cached metrics: otherwise the exporter keeps
            # re-exporting the last batch of the removed instance, so dashboards
            # and alerts keep showing an instance that no longer exists.
            if self._cache is not None:
                self._cache["unified"] = {}
                self._cache["raw"] = {}
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
        for ep, metrics in endpoint_metrics.items():
            if not metrics:
                continue
            try:
                mi = endpoint_to_instance[ep]
                m = instance_id_to_model.get(mi.id)

                runtime = get_backend(m)
                runtime_version = self.fetch_and_update_api_backend_version(mi, ep)

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
                self._warned.pop(("endpoint", ep), None)
            except Exception as e:
                # Keep one endpoint from discarding the metrics collected from
                # the others on this worker.
                self._warn_once(
                    ("endpoint", ep), f"Skipping metrics from endpoint {ep}: {e}"
                )

        self._cache["unified"] = unified_metrics
        self._cache["raw"] = raw_metrics
        logger.trace(f"trace_id: {trace_id}, completed fetching runtime metrics.")

    def fetch_and_update_api_backend_version(
        self,
        model_instance: ModelInstance,
        endpoint: str,
    ) -> Optional[str]:
        if model_instance.api_detected_backend_version is not None:
            return model_instance.api_detected_backend_version

        api_endpoint = f"{model_instance.worker_ip}:{_api_port(model_instance)}"
        if api_endpoint != endpoint:
            # The caller hands us the EXPOSITION endpoint, which for a router is a
            # separate port band (see `_metrics_port`). `/version` does not live
            # there. Worse than a 404: the router's exposition listener answers
            # EVERY path with 200 + Prometheus text, so probing it returns a body
            # that is not JSON and the parse raises once per scrape — an ERROR
            # every few seconds that reads like metrics collection is broken when
            # only the version probe is.
            logger.trace(
                f"Instance {model_instance.id} exposes metrics on a separate port; "
                f"probing the API port {api_endpoint} for its runtime version."
            )
        version = self._metrics_client.fetch_runtime_version_from_endpoint(
            api_endpoint, model_instance.backend
        )
        if version is not None:
            self._update_model_instance(
                model_instance.id, api_detected_backend_version=version
            )
            return version

        return model_instance.backend_version

    def _find_active_model_endpoints(
        self, worker_id: int, metrics_config: dict
    ) -> tuple[set, dict[str, ModelInstance], dict[int, Model]]:
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

            endpoint = f"{mi.worker_ip}:{_metrics_port(mi)}"
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

    def _update_model_instance(self, id: int, **kwargs):
        try:
            mi_public = self._clientset.model_instances.get(id=id)

            mi = ModelInstanceUpdate(**mi_public.model_dump())
            for key, value in kwargs.items():
                setattr(mi, key, value)

            self._clientset.model_instances.update(id=id, model_update=mi)
        except Exception as e:
            logger.error(f"Failed to update model instance {id}: {e}")

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
            # Empty for a single-role deployment. Carried because under
            # disaggregation the engine-level latencies are not one
            # population: prefill owns TTFT and decode owns TPOT, and a
            # figure averaged over both roles describes neither.
            "role": mi.role or "",
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
        instance_id = base_labels.get("model_instance_id", "")
        for source_family_name, family in metrics.items():
            warn_key = ("family", instance_id, source_family_name)
            try:
                self._process_metric_family(
                    source_family_name,
                    family,
                    base_labels,
                    runtime,
                    runtime_version,
                    unified_metrics,
                    raw_metrics,
                    metrics_config,
                )
                self._warned.pop(warn_key, None)
            except Exception as e:
                # Keep a single unsupported family from discarding every metric
                # collected on this worker.
                self._warn_once(
                    warn_key, f"Skipping metric family {source_family_name}: {e}"
                )

    def _process_metric_family(
        self,
        source_family_name,
        family,
        base_labels,
        runtime,
        runtime_version,
        unified_metrics,
        raw_metrics,
        metrics_config,
    ):
        """
        Aggregate a single metric family into the unified and raw collections.
        """
        first_sample = family.samples[0] if family.samples else None
        if not first_sample:
            return

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
        resolved = get_unified_metric_family_name(
            metrics_config, source_family_name, runtime, runtime_version
        )
        unified_metric_family_name, unified_scale = resolved or (None, 1.0)
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

        # Histogram, summary and info samples are added under the names the
        # runtime gave them. Everything else goes through add_metric, which
        # appends the type's own suffix: OpenMetrics keeps a counter's _created
        # sample inside the family (the Prometheus text parser splits it off
        # into one of its own), and passing that through would emit a second
        # _total series carrying a creation timestamp under identical labels —
        # a duplicate Prometheus rejects an entire scrape over.
        keeps_sample_names = family.type in ("histogram", "summary", "info")
        value_sample_names = {source_family_name}
        if family.type == "counter":
            value_sample_names.add(f"{source_family_name}_total")

        for sample in family.samples:
            if not keeps_sample_names and sample.name not in value_sample_names:
                continue

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

            if keeps_sample_names:
                # The raw passthrough is never scaled: it exists to be the
                # engine's own numbers under the engine's own names.
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
                    unified_value, unified_labels = scale_sample(
                        sample.name,
                        source_family_name,
                        family.type,
                        labels,
                        sample.value,
                        unified_scale,
                    )
                    unified_family.add_sample(
                        name=new_name,
                        labels=unified_labels,
                        value=unified_value,
                        timestamp=sample.timestamp,
                    )
            else:
                raw_family.add_metric(
                    labels=label_values,
                    value=sample.value,
                    timestamp=sample.timestamp,
                )
                if unified_family:
                    unified_value, _ = scale_sample(
                        sample.name,
                        source_family_name,
                        family.type,
                        labels,
                        sample.value,
                        unified_scale,
                    )
                    unified_family.add_metric(
                        labels=label_values,
                        value=unified_value,
                        timestamp=sample.timestamp,
                    )

    def _should_skip_endpoint(
        self, model: Model, model_instance: ModelInstance, metrics_config: dict
    ) -> bool:
        # model and model instance must be valid
        if not model:
            return True

        if (
            model_instance.state != ModelInstanceStateEnum.RUNNING
            or model_instance.worker_ip is None
            or not model_instance.ports
        ):
            return True

        # skip image and audio models
        if is_image_model(model) or is_audio_model(model):
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
                f"{self._clientset.base_url}/v2/metrics/config",
                timeout=METRICS_CONFIG_FETCH_TIMEOUT_SECONDS,
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
        """Get metrics config with automatic caching (300 seconds TTL)."""
        try:
            return self._metrics_config_cache["config"]
        except KeyError:
            # Cache miss, fetch fresh config
            pass

        online_config = self._get_online_metrics_config()
        if online_config:
            logger.debug("Updated online metrics config cache")
            self._metrics_config_cache["config"] = online_config
            return online_config
        else:
            builtin_config = get_builtin_metrics_config()
            logger.debug("Using builtin metrics config")
            # Cache for 300 seconds
            self._metrics_config_cache["config"] = builtin_config
            return builtin_config


_METRIC_FAMILY_CLASS = {
    "gauge": GaugeMetricFamily,
    "info": InfoMetricFamily,
    "histogram": HistogramMetricFamily,
    "counter": CounterMetricFamily,
    "summary": SummaryMetricFamily,
    # Samples a runtime exposes without a TYPE declaration.
    "unknown": UnknownMetricFamily,
    "untyped": UnknownMetricFamily,
}


def _api_port(mi) -> int:
    """Where this instance serves its HTTP API.

    Separate from `_metrics_port` because the two coincide for an engine and
    diverge for a router. Anything asking a question of the SERVER (its version,
    its health) belongs here; only the Prometheus scrape belongs on the other.
    """
    return mi.port or mi.ports[0]


def _metrics_port(mi) -> int:
    """Where this instance serves its Prometheus exposition.

    Normally the serving port — an engine exposes `/metrics` on the same
    listener as its API. A PD router does not: its exposition is a separate
    listener on a band GPUStack allocates, because upstream's default port is
    fixed and two routers on one host would collide. Scraping its API port
    returns 404, and a 404 here is indistinguishable from "this instance has
    no metrics" — the router's request counters are the denominator of the
    PD-effectiveness ratio, so losing them silently costs the one signal that
    separates "PD works" from "PD stopped disaggregating".

    Keyed on the band's *name* rather than on the role, so this stays true for
    any instance that separates its metrics listener, not just today's router.
    """
    band = (mi.named_ports or {}).get("prometheus") if mi.named_ports else None
    if band is not None:
        base = getattr(band, "base", None)
        if base is None and isinstance(band, dict):
            base = band.get("base")
        if base:
            return int(base)
    return mi.ports[0]


def create_prom_metric_family(type: str, name: str, description: str, labels=None):
    cls = _METRIC_FAMILY_CLASS.get(str(type).lower())
    if not cls:
        raise ValueError(f"Unknown metric family type: {type}")
    if labels is not None:
        return cls(name, description, labels=labels)
    else:
        return cls(name, description)


def _parse_mapping_entry(entry) -> Optional[Tuple[str, float]]:
    """A mapping value -> (unified name, scale).

    Two accepted shapes, and the string one is the whole reason for this
    function: every existing entry is `raw: unified`, and they must keep
    meaning exactly what they meant (scale 1).

        vllm:num_requests_running: gpustack:num_requests_running
        sglang:kv_transfer_total_mb:
          name: gpustack:pd_kv_transfer_bytes
          scale: 1048576
    """
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry, 1.0
    if isinstance(entry, dict):
        name = entry.get("name")
        if not name:
            return None
        try:
            scale = float(entry.get("scale", 1.0))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring a non-numeric scale on metric mapping %r; using 1.0",
                name,
            )
            scale = 1.0
        return name, scale
    return None


def get_unified_metric_family_name(
    config: dict,
    source_metric_family_name: str,
    runtime: str,
    runtime_version: Optional[str],
) -> Optional[Tuple[str, float]]:
    """
    Return (unified metric family name, scale) or None.
    Prefer version-specific mapping if matched, otherwise use the default '*'.

    The scale converts the engine's unit into the unified metric's declared
    one. It exists because renaming alone is only half a normalization: two
    engines can report the same concept in different units, and a shared name
    over mixed units is worse than two names — a reader can no longer tell
    which they are looking at. Measured case: SGLang reports KV transfer
    volume in megabytes and duration in milliseconds where vLLM reports bytes
    and seconds.
    """
    runtime_cfg = get_runtime_metrics_config(config, runtime)
    if not runtime_cfg:
        return None

    entry = runtime_cfg.get("*", {}).get(source_metric_family_name, None)
    if runtime_version:
        is_valid_version = version.is_valid_version_str(runtime_version)
        for ver_range, mapping in runtime_cfg.items():
            if ver_range == "*":
                continue
            if (is_valid_version and version.in_range(runtime_version, ver_range)) or (
                not is_valid_version and runtime_version == ver_range
            ):
                old_version_entry = mapping.get(source_metric_family_name)
                if old_version_entry is not None:
                    return _parse_mapping_entry(old_version_entry)

    return _parse_mapping_entry(entry)


def scale_sample(
    sample_name: str,
    family_name: str,
    family_type: str,
    labels: dict,
    value: float,
    scale: float,
) -> Tuple[float, dict]:
    """One sample's value and labels, converted into the unified unit.

    A histogram cannot be multiplied through, and getting this wrong is
    silent. The four sample kinds carry different dimensions:

    - `_sum`      the sum of the observed values -> **scaled**
    - `_count`    how many observations -> **never scaled**, it is a count
    - `_bucket`   its value is also a count -> not scaled, but its `le` label
                  is a bucket *boundary* in the observed unit -> **scaled**
    - `_created`  a unix timestamp -> never scaled

    Missing the `le` label is the trap: nothing errors, and the histogram
    ends up with a sum in one unit and boundaries in another, so every
    `histogram_quantile` over it is wrong by the scale factor.

    A summary is the mirror image: its quantile samples carry an observed
    *value*, so those are scaled, while `le`-style boundaries do not exist.
    """
    if scale == 1.0:
        return value, labels

    if sample_name.endswith("_count") or sample_name.endswith("_created"):
        return value, labels

    if sample_name.endswith("_sum"):
        return value * scale, labels

    if sample_name.endswith("_bucket"):
        upper = labels.get("le")
        if upper is None or upper in ("+Inf", "Inf"):
            return value, labels
        try:
            scaled_labels = dict(labels)
            scaled_labels["le"] = repr(float(upper) * scale)
            return value, scaled_labels
        except (TypeError, ValueError):
            return value, labels

    # A summary's quantile sample, or a plain counter/gauge: the value is an
    # observation in the engine's unit.
    return value * scale, labels


def get_unified_metric_family_config(
    config: dict, unified_metric_family_name: str
) -> dict:
    return config.get("gpustack_metrics", {}).get(unified_metric_family_name, {})
