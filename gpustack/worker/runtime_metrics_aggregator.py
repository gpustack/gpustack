from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Iterator, Tuple
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
    runtime_metric_labels,
)
from gpustack.utils.model_instance_workers import (
    ModelInstanceMetricsSource,
    get_model_instance_workloads,
    model_instance_metrics_sources,
)
from gpustack.worker.runtime_metrics_client import (
    Config as RunTimeMetricsClientConfig,
)
from gpustack.worker.runtime_metrics_client import Client as RuntimeMetricsClient
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
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

METRICS_CONFIG_FETCH_TIMEOUT_SECONDS = 2

# unified registry
unified_registry = CollectorRegistry()

# raw metrics registry
raw_registry = CollectorRegistry()


@dataclass(frozen=True)
class _RuntimeMetricsEndpoint:
    instance: ModelInstance
    source: ModelInstanceMetricsSource
    host: str
    worker_name: str
    worker_identity: tuple = ()

    def same_owner(self, other) -> bool:
        return (
            self.source == other.source
            and self.host == other.host
            and self.worker_identity == other.worker_identity
        )


# Ten collection intervals (Worker schedules aggregation every three seconds).
_RUNTIME_METRICS_CACHE_TTL_SECONDS = 30


class _RuntimeMetricsSnapshot(Mapping):
    """A published batch whose samples stop being readable after its TTL."""

    def __init__(self, metrics: dict, collected_at: float):
        self._metrics = metrics
        self._expires_at = collected_at + _RUNTIME_METRICS_CACHE_TTL_SECONDS

    def __getitem__(self, key):
        if monotonic() >= self._expires_at:
            raise KeyError(key)
        return self._metrics[key]

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def __len__(self):
        return len(self._metrics) if monotonic() < self._expires_at else 0

    def items(self):
        for key, metric in self._metrics.items():
            if monotonic() >= self._expires_at:
                return
            yield key, metric

    def values(self):
        for _, metric in self.items():
            yield metric


def runtime_metrics_values(cache: Optional[dict], kind: str) -> Iterator:
    """Read a batch through its freshness boundary at each exported family."""
    metrics = cache.get(kind) if cache is not None else None
    if metrics is not None:
        yield from metrics.values()


class RuntimeMetricsAggregator:
    def __init__(
        self,
        cache: dict = None,
        worker_id_getter=Callable[[], int],
        clientset: ClientSet = None,
    ):
        self._cache = cache
        self._metrics_client_config = RunTimeMetricsClientConfig(
            timeout=2, max_retries=0, insecure_tls=True
        )
        self._metrics_client = RuntimeMetricsClient(self._metrics_client_config)
        self._worker_id_getter = worker_id_getter
        self._clientset = clientset

        self._runtime_version_cache = TTLCache(maxsize=1024, ttl=300)

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

        # Version probes are per instance, never per sample or logical rank.
        runtime_versions = self._fetch_source_versions(
            endpoint_to_instance, endpoint_metrics
        )

        # A request can finish after a task was replaced, stopped, or reassigned.
        # Re-read the entire scoped set once, rather than issuing a GET per sample.
        _, current_sources, _ = self._find_active_model_endpoints(
            worker_id, metrics_config
        )

        # 4. Unified and raw aggregation
        unified_metrics = {}
        raw_metrics = {}
        for ep in endpoints:
            previous = endpoint_to_instance[ep]
            current = current_sources.get(ep)
            if current is None or not previous.same_owner(current):
                continue
            metrics = endpoint_metrics.get(ep)
            try:
                mi = current.instance
                m = instance_id_to_model.get(mi.id)
                runtime = get_backend(m)
                runtime_version = runtime_versions.get(mi.id)
                base_labels = self._build_base_labels(mi, m, runtime, current)
                self._add_scrape_status(
                    unified_metrics, raw_metrics, base_labels, metrics
                )
                if not metrics:
                    continue
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

        collected_at = monotonic()
        self._cache["unified"] = _RuntimeMetricsSnapshot(unified_metrics, collected_at)
        self._cache["raw"] = _RuntimeMetricsSnapshot(raw_metrics, collected_at)
        logger.trace(f"trace_id: {trace_id}, completed fetching runtime metrics.")

    def _fetch_source_versions(self, endpoints: dict, endpoint_metrics: dict) -> dict:
        versions, pending = {}, {}
        for endpoint, context in endpoints.items():
            mi, source = context.instance, context.source
            if (
                source.kind == "sglang_smg"
                or mi.id in versions
                or not endpoint_metrics.get(endpoint)
            ):
                continue
            if mi.api_detected_backend_version is not None:
                versions[mi.id] = mi.api_detected_backend_version
                continue
            key = (mi.id, source, context.host, context.worker_identity, mi.backend)
            if key in self._runtime_version_cache:
                versions[mi.id] = self._runtime_version_cache[key]
            else:
                versions[mi.id] = mi.backend_version
                api = f"{context.host}:{source.api_port}"
                pending.setdefault(mi.backend, {})[api] = (mi, key)
        # Probe each runtime concurrently within one shared batch deadline.
        probes = {api: runtime for runtime, items in pending.items() for api in items}
        detected = self._metrics_client.fetch_runtime_versions(probes) if probes else {}
        for runtime, items in pending.items():
            for api, (mi, key) in items.items():
                detected_version = detected.get(api)
                value = detected_version or mi.backend_version
                if detected_version:
                    self._runtime_version_cache[key] = detected_version
                versions[mi.id] = value
                if detected_version and get_model_instance_workloads(mi) is None:
                    self._update_model_instance(
                        mi.id,
                        expected_instance=mi,
                        api_detected_backend_version=detected_version,
                    )
        return versions

    def _add_scrape_status(self, unified_metrics, raw_metrics, labels, metrics):
        name = "gpustack:runtime_metrics_scrape_status"
        status = getattr(metrics, "status", None) or (
            "error" if metrics is None else "success" if metrics else "empty"
        )
        if status not in ("success", "empty", "error", "parse_error"):
            status = "error"
        if name not in raw_metrics:
            raw_metrics[name] = GaugeMetricFamily(name, "Runtime source scrape outcome")
            unified_metrics[name] = raw_metrics[name]
        raw_metrics[name].add_sample(name, {**labels, "status": status}, 1)

    def fetch_and_update_api_backend_version(
        self,
        model_instance: ModelInstance,
        endpoint: str,
        api_endpoint: Optional[str] = None,
    ) -> Optional[str]:
        if model_instance.api_detected_backend_version is not None:
            return model_instance.api_detected_backend_version

        api_endpoint = api_endpoint or (
            f"{model_instance.worker_ip}:{_api_port(model_instance)}"
        )
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
            fence = (
                {"expected_instance": model_instance}
                if getattr(self, "_clientset", None) is not None
                else {}
            )
            self._update_model_instance(
                model_instance.id, api_detected_backend_version=version, **fence
            )
            return version

        return model_instance.backend_version

    def _find_active_model_endpoints(
        self, worker_id: int, metrics_config: dict
    ) -> tuple[set, dict[str, _RuntimeMetricsEndpoint], dict[int, Model]]:
        """
        Get real exposition sources owned by this Worker's current identity.
        """
        model_instances, models = self._list_worker_models(worker_id)
        if not model_instances or not models:
            return set(), {}, {}

        model_id_to_model = {m.id: m for m in models.items}
        endpoints = set()
        endpoint_to_instance = {}
        instance_id_to_model = {}
        conflicting = set()
        for mi in model_instances.items:
            model = model_id_to_model.get(mi.model_id)
            if mi.cluster_id != self._worker.cluster_id or self._should_skip_endpoint(
                model=model,
                model_instance=mi,
                metrics_config=metrics_config,
            ):
                logger.trace(f"Skipping model instance {mi.id} in metrics aggregation.")
                continue

            for source in model_instance_metrics_sources(mi):
                if source.worker_id != worker_id:
                    continue
                endpoint = f"{self._worker.ip}:{source.metrics_port}"
                if endpoint in conflicting:
                    continue
                context = _RuntimeMetricsEndpoint(
                    mi,
                    source,
                    self._worker.ip,
                    self._worker.name,
                    (
                        self._worker.cluster_id,
                        getattr(self._worker, "system_principal_id", None),
                    ),
                )
                previous = endpoint_to_instance.get(endpoint)
                if previous is not None and not previous.same_owner(context):
                    # A socket cannot carry two authoritative instance identities.
                    conflicting.add(endpoint)
                    endpoint_to_instance.pop(endpoint)
                    endpoints.discard(endpoint)
                    self._warn_once(
                        ("source_conflict", endpoint),
                        "Skipping runtime metrics socket with conflicting ownership",
                    )
                    continue
                endpoints.add(endpoint)
                endpoint_to_instance[endpoint] = context
                instance_id_to_model[mi.id] = model

        return endpoints, endpoint_to_instance, instance_id_to_model

    def _list_worker_models(self, worker_id: int):
        """
        Read the full scoped set; worker_id only filters primary ownership.
        """
        self._worker = self._clientset.workers.get(id=worker_id, use_cache=False)
        if (
            self._worker.id != worker_id
            or self._worker.cluster_id is None
            or getattr(self._worker, "me", None) is not True
        ):
            raise ValueError("Runtime metrics require the current cluster-bound Worker")
        model_instances = self._clientset.model_instances.list(
            params={"cluster_id": self._worker.cluster_id},
            use_cache=True,
        )
        models = self._clientset.models.list(use_cache=True)
        return model_instances, models

    def _update_model_instance(self, id: int, expected_instance=None, **kwargs):
        try:
            mi_public = self._clientset.model_instances.get(id=id, use_cache=False)
            if (
                expected_instance is None
                or get_model_instance_workloads(mi_public) is not None
                or model_instance_metrics_sources(mi_public)
                != model_instance_metrics_sources(expected_instance)
            ):
                return

            mi = ModelInstanceUpdate(**mi_public.model_dump())
            for key, value in kwargs.items():
                setattr(mi, key, value)

            self._clientset.model_instances.update(id=id, model_update=mi)
        except Exception as e:
            logger.error(f"Failed to update model instance {id}: {e}")

    def _build_base_labels(self, mi, m, runtime, context=None):
        """
        Build base labels for each metric.
        """
        labels = {
            "worker_id": str(context.source.worker_id if context else mi.worker_id),
            "worker_name": context.worker_name if context else mi.worker_name or "",
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
        if context is not None:
            source = context.source
            labels.update(
                restart_count=str(source.restart_count),
                source_kind=source.kind,
                source_name=source.name,
                workload_name=source.workload_name,
            )
            if source.dp_rank is not None:
                labels["endpoint_dp_rank"] = str(source.dp_rank)
                labels["gpu_indexes"] = ",".join(str(i) for i in source.gpu_indexes)
            if source.attempt is not None:
                labels["component_attempt"] = str(source.attempt)
        return labels

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
        mapping_domain = (
            "SGLangSMG" if base_labels.get("source_kind") == "sglang_smg" else runtime
        )
        resolved = get_unified_metric_family_name(
            metrics_config, source_family_name, mapping_domain, runtime_version
        )
        unified_name, unified_scale = resolved or (None, 1.0)
        cfg = get_unified_metric_family_config(metrics_config, unified_name)
        keeps_sample_names = family.type in ("histogram", "summary", "info")
        value_names = {source_family_name, f"{source_family_name}_total"}
        for sample in family.samples:
            # OpenMetrics includes creation timestamps inside counter families.
            if not keeps_sample_names and sample.name not in value_names:
                continue
            platform_labels, can_unify = self._sample_scope(
                sample, base_labels, runtime
            )
            if platform_labels is None:
                continue
            labels = runtime_metric_labels(sample.labels, platform_labels)
            if source_family_name not in raw_metrics:
                raw_metrics[source_family_name] = create_prom_metric_family(
                    name=source_family_name,
                    type=family.type,
                    description=family.documentation,
                )
            raw_name = (
                f"{raw_metrics[source_family_name].name}_total"
                if family.type == "counter"
                else sample.name
            )
            raw_metrics[source_family_name].add_sample(
                name=raw_name,
                labels=labels,
                value=sample.value,
                timestamp=sample.timestamp,
            )
            if not (can_unify and unified_name and cfg):
                continue
            if unified_name not in unified_metrics:
                unified_metrics[unified_name] = create_prom_metric_family(
                    name=unified_name,
                    type=cfg.get("type"),
                    description=cfg.get("description"),
                )
            value, scaled_labels = scale_sample(
                sample.name,
                source_family_name,
                family.type,
                labels,
                sample.value,
                unified_scale,
            )
            unified_family = unified_metrics[unified_name]
            unified_sample_name = unified_family.name
            if keeps_sample_names:
                unified_sample_name = raw_name.replace(
                    source_family_name, unified_family.name, 1
                )
            elif unified_family.type == "counter":
                unified_sample_name += "_total"
            unified_family.add_sample(
                name=unified_sample_name,
                labels=scaled_labels,
                value=value,
                timestamp=sample.timestamp,
            )

    def _sample_scope(self, sample, base_labels, runtime):
        labels = dict(base_labels)
        kind = labels.get("source_kind")
        if kind == "sglang_smg":
            labels["exposition_scope"] = "router"
            return labels, True
        if kind == "sglang_engine":
            labels["exposition_scope"] = "shared_engine"
            return labels, True
        if kind != "vllm_engine":
            return labels, True
        # The supported Python shape uses a native global DP engine index.
        # Missing/unknown shapes (including unverified frontends) stay raw;
        # another engine's samples can never inherit this endpoint's rank.
        engine = sample.labels.get("engine")
        known_engine = (
            runtime == BackendEnum.VLLM
            and isinstance(engine, str)
            and engine.isdecimal()
        )
        if known_engine and int(engine) != int(labels["endpoint_dp_rank"]):
            return None, False
        owns_engine = known_engine
        labels["exposition_scope"] = "engine" if owns_engine else "unknown"
        if not owns_engine:
            labels.pop("endpoint_dp_rank", None)
            labels.pop("gpu_indexes", None)
        return labels, owns_engine

    def _should_skip_endpoint(
        self, model: Model, model_instance: ModelInstance, metrics_config: dict
    ) -> bool:
        # model and model instance must be valid
        if not model:
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
