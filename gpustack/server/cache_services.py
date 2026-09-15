import logging
from typing import Any, Dict, Optional, Tuple

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_providers import (
    CacheProvider,
    render_optional_template,
    resolved_field_values,
)
from gpustack.schemas.cache_services import (
    CacheConfigSnapshot,
    CacheService,
    CacheServiceEndpoint,
    CacheServiceInstance,
    CacheServiceStateEnum,
)
from gpustack.schemas.models import Model, get_backend
from gpustack.schemas.workers import Worker
from gpustack.utils.command import flatten_to_argv
from gpustack.utils.version import version_in_range
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    render_injection,
)

logger = logging.getLogger(__name__)


async def _resolve_managed_endpoint(
    session: AsyncSession,
    service: CacheService,
    provider: CacheProvider,
    worker: Optional[Worker],
) -> Tuple[Optional[CacheServiceEndpoint], Optional[str]]:
    """
    Pick the cache service instance a model instance should attach to and
    return its endpoint (instance worker IP + instance port). The endpoint
    params carry the placement as the neutral "locality" fact
    ("node_local" | "remote"); provider declarations map it to their own
    connector vocabulary via injection.locality_params.

    Providers declaring attach_locality node_local attach node-local
    only: measured remote transfer (the engine-driven copy path) is
    slower than running without the cache at all, and falling back would
    also funnel every uncovered engine onto a single instance — so an
    engine on a worker without a RUNNING cache instance starts degraded
    instead. Cluster-attachable providers serve any worker from any
    instance. Returns (None, reason) when no instance is usable.
    """
    instances = await CacheServiceInstance.all_by_fields(
        session, {"cache_service_id": service.id}
    )
    # Engines attach to one declared component's address (the
    # master, not its stores); single-component providers attach to
    # their sole ("") component.
    attach_component = provider.attach_component()
    running = sorted(
        (
            instance
            for instance in instances
            if (instance.component or "") == attach_component
            and instance.state == CacheServiceStateEnum.RUNNING
            and instance.port
        ),
        key=lambda instance: instance.id,
    )
    if not running:
        return None, (
            "Cache service has no running instance; "
            "instance starts without shared KV cache"
        )

    target = None
    if worker is not None and worker.id is not None:
        target = next(
            (instance for instance in running if instance.worker_id == worker.id),
            None,
        )
    node_local = target is not None
    if target is None:
        if provider.attach_locality == "node_local":
            if worker is None:
                return None, (
                    "Cache endpoint resolves with the instance's worker "
                    "at scheduling (node-local attach); instance starts "
                    "without shared KV cache"
                )
            worker_name = getattr(worker, "name", None) or f"id={worker.id}"
            return None, (
                f"No running cache instance on worker {worker_name}; "
                "instance starts without shared KV cache"
            )
        target = running[0]

    service_worker = await Worker.one_by_id(session, target.worker_id)
    if (
        service_worker is None
        or service_worker.deleted_at is not None
        or not service_worker.ip
    ):
        return None, (
            "Cache service worker is unavailable; "
            "instance starts without shared KV cache"
        )
    return (
        CacheServiceEndpoint(
            host=service_worker.ip,
            port=target.port,
            params={"locality": "node_local" if node_local else "remote"},
        ),
        None,
    )


def _declared_attach_address(
    provider: Optional[CacheProvider], resolved_fields: Dict[str, Any]
) -> Optional[str]:
    """The address engines should attach to when the attach component
    declares an indirection instead of answering as one instance — an HA
    master pool is reached through its coordination backend. None means
    the resolved instance address stands."""
    if provider is None or not provider.components:
        return None
    spec = provider.get_component(provider.attach_component())
    if spec is None:
        return None
    return render_optional_template(spec.address_template, resolved_fields)


async def resolve_instance_cache_config(
    session: AsyncSession,
    model: Model,
    worker: Optional[Worker] = None,
    spans_workers: bool = False,
) -> Optional[CacheConfigSnapshot]:
    """
    Resolve the shared-cache connection snapshot for an instance of the
    given model. Returns None when the model does not use a shared cache.
    A snapshot with injected=False means the instance starts degraded
    (without the shared cache); the reason field explains why.

    ``worker`` is the instance's assigned worker. A node-local provider
    resolves against it, so calls made before scheduling yield an explicit
    pending snapshot for such services.

    ``spans_workers`` marks an instance actually placed across several
    workers (subordinate workers assigned at scheduling). The
    distributed_inference_across_workers model flag is only a
    permission — most single-node placements carry it — so the
    node-local incompatibility is decided here, where the real
    placement is known, not at model validation.
    """
    ext = model.extended_kv_cache
    if not ext or not ext.is_shared():
        return None

    if not ext.cache_service_id:
        return CacheConfigSnapshot(
            cache_service_id=ext.cache_service_id or 0,
            injected=False,
            reason="No cache service specified; instance starts without shared KV cache",
        )

    service = await CacheService.one_by_id(session, ext.cache_service_id)
    if service is None or service.deleted_at is not None:
        return CacheConfigSnapshot(
            cache_service_id=ext.cache_service_id,
            injected=False,
            reason="Cache service not found; instance starts without shared KV cache",
        )

    # extended_kv_cache.chunk_size belongs to the in-process mode (the two
    # modes are mutually exclusive) and must not leak into shared mode:
    # the cache server renders the service value into its own command, so
    # honoring a deployment-side value here would let the engine chunk
    # differently from the server it attaches to.
    chunk_size = ((service.config.fields if service.config else None) or {}).get(
        "chunk_size"
    )
    snapshot_base = dict(
        cache_service_id=service.id,
        cache_service_name=service.name,
        provider_name=service.provider_name,
        provider_version=service.provider_version,
        chunk_size=chunk_size,
    )

    provider = get_cache_provider(service.provider_name)
    if provider is None:
        return CacheConfigSnapshot(
            **snapshot_base,
            injected=False,
            reason=(
                f"Unknown cache provider '{service.provider_name}'; "
                "instance starts without shared KV cache"
            ),
        )

    if spans_workers and provider.attach_locality == "node_local":
        # Declared attach contract, the same predicate
        # _resolve_managed_endpoint keys its no-remote-fallback rule on:
        # node_local connectors (MP-style, CUDA IPC) have no cross-host
        # path, so every subordinate worker of a spanning instance would
        # face a remote server. Cluster-attachable providers (e.g.
        # a distributed pool) serve spanning instances by
        # design and pass through — with a known calibration caveat: the
        # snapshot renders once with the main worker, so subordinate
        # workers see its local_hostname.
        return CacheConfigSnapshot(
            **snapshot_base,
            injected=False,
            reason=(
                f"Cache provider '{provider.name}' attaches node-locally; "
                "this instance spans multiple workers and starts without "
                "the shared KV cache"
            ),
        )

    # Resolution is driven by the service's instances rather than the
    # service-level state: the aggregate may lag behind the instances, and
    # a partially-running per-node service can still serve engines from
    # its RUNNING instances.
    endpoint, reason = await _resolve_managed_endpoint(
        session, service, provider, worker
    )
    if endpoint is None:
        return CacheConfigSnapshot(
            **snapshot_base,
            injected=False,
            reason=reason,
        )
    snapshot_endpoint = endpoint

    backend = get_backend(model)
    resolved_fields = resolved_field_values(
        provider.fields if provider else [],
        (service.config.fields if service.config else None) or {},
    )
    attach_address = _declared_attach_address(provider, resolved_fields)
    render_params: Dict[str, Any] = {
        "host": endpoint.host,
        "port": endpoint.port,
        "chunk_size": chunk_size,
        # Identifies the service to whatever the engine shares with its
        # cache servers — a coordination keyspace, for one.
        "service_id": service.id,
        # The consuming instance's own worker IP: a connector (e.g.
        # a store connector's) uses it as the client identity / RDMA peer address,
        # which defaults to localhost and would be wrong across nodes.
        "local_hostname": worker.ip if worker and worker.ip else None,
        # Convenience alias for connectors that take a single
        # host:port service address. A component declaring an
        # address_template answers with that instead while its
        # placeholders resolve — an HA master pool is reached through the
        # coordination backend, not through one elected instance.
        "master_server_address": (
            attach_address
            or (
                f"{endpoint.host}:{endpoint.port}"
                if endpoint.host and endpoint.port
                else endpoint.url
            )
        ),
    }
    # External-mode connection fields feed additional placeholders declared
    # by the provider; they never override the well-known keys above.
    # Declared field defaults backstop unset fields inside render_injection.
    for key, value in (endpoint.params or {}).items():
        render_params.setdefault(key, value)
    # Managed-mode field values feed the same namespace, so injection
    # templates may reference provider-declared fields too — resolved
    # through their visibility gates (a hidden field must not leak its
    # embedded-mode default into a standalone-store config).
    for key, value in resolved_fields.items():
        render_params.setdefault(key, value)
    # The instance worker's accelerator framework selects a
    # framework-scoped integration entry when the provider declares
    # one (e.g. a cann-specific vLLM contract); pre-scheduling calls
    # (worker None) fall back to the generic entry.
    worker_status = getattr(worker, "status", None)
    framework = next(
        (
            device.type
            for device in (getattr(worker_status, "gpu_devices", None) or [])
            if device.type
        ),
        None,
    )
    # A user parameter carrying the integration's connector slot takes
    # the slot over (the engine accepts one value and user args win) —
    # an intentional escape hatch, but it must never be silent: the
    # instance is reported as running without the platform-injected
    # cache, and none of the injection applies (the user owns the whole
    # connector wiring).
    integration = provider.integration_for(backend, framework)
    # Version floor for existing/unvalidated models: an engine below the
    # integration's declared range would crash on injected args it does
    # not know (e.g. --shutdown-timeout), so degrade instead of
    # injecting. Unknown or unparseable versions fail open.
    if (
        integration is not None
        and integration.versions
        and model.backend_version
        and version_in_range(model.backend_version, integration.versions) is False
    ):
        return CacheConfigSnapshot(
            **snapshot_base,
            endpoint=snapshot_endpoint,
            injected=False,
            reason=(
                f"Backend version {model.backend_version} is outside the "
                f"cache provider's supported '{backend}' range "
                f"({integration.versions}); "
                "instance starts without shared KV cache"
            ),
        )
    slot = integration.injection.kv_transfer_config if integration else None
    # backend_parameters is semantically a concatenated argv (an element
    # may be one token, a "--key value" pair, or a whole pasted command
    # line) — flatten exactly like the worker does before matching, or
    # the pasted forms slip through and take the slot over silently.
    user_argv = flatten_to_argv(model.backend_parameters or [])
    if slot and any(
        token == slot.flag or token.startswith(f"{slot.flag}=") for token in user_argv
    ):
        return CacheConfigSnapshot(
            **snapshot_base,
            endpoint=snapshot_endpoint,
            injected=False,
            reason=(
                f"User parameter {slot.flag} takes over the KV connector; "
                "instance starts without the platform-injected shared KV cache"
            ),
        )
    rendered = render_injection(provider, backend, render_params, framework)
    if rendered is None:
        on_framework = f" on {framework} workers" if framework else ""
        return CacheConfigSnapshot(
            **snapshot_base,
            endpoint=snapshot_endpoint,
            injected=False,
            reason=(
                f"Cache provider '{provider.name}' is not compatible with "
                f"backend '{backend}'{on_framework}; "
                "instance starts without shared KV cache"
            ),
        )

    env, args, files = rendered
    return CacheConfigSnapshot(
        **snapshot_base,
        endpoint=snapshot_endpoint,
        env=env,
        args=args,
        files=files,
        injected=True,
    )


async def resolve_instance_cache_config_safe(
    session: AsyncSession,
    model: Model,
    worker: Optional[Worker] = None,
    spans_workers: bool = False,
) -> Optional[CacheConfigSnapshot]:
    """
    resolve_instance_cache_config that degrades instead of raising: an
    unexpected resolution error yields an injected=False snapshot so
    callers on critical paths (e.g. the scheduler) keep going.
    """
    try:
        return await resolve_instance_cache_config(
            session, model, worker=worker, spans_workers=spans_workers
        )
    except Exception as e:
        logger.error(
            f"Failed to resolve shared cache config for model {model.name}: {e}"
        )
        ext = model.extended_kv_cache
        if not ext or not ext.is_shared():
            return None
        return CacheConfigSnapshot(
            cache_service_id=ext.cache_service_id or 0,
            injected=False,
            reason=(
                f"Failed to resolve cache config: {e}; "
                "instance starts without shared KV cache"
            ),
        )


METRICS_FETCH_TIMEOUT_SECONDS = 5

METRICS_RETENTION_SECONDS = 24 * 3600
"""Samples older than this are deleted on each collector pass."""
