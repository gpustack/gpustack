import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.cache_services import (
    CacheConfigSnapshot,
    CacheService,
    CacheServiceEndpoint,
    CacheServiceInstance,
    CacheServiceModeEnum,
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
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

_DEFAULT_SCHEME_PORTS = {
    "http": 80,
    "https": 443,
}


def _endpoint_host_port(
    endpoint: CacheServiceEndpoint,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract a connectable (host, port) pair from an endpoint, falling back
    to parsing the URL form when explicit host/port are absent.
    """
    host, port = endpoint.host, endpoint.port
    if host and port:
        return host, port

    if endpoint.url:
        parsed = urlparse(endpoint.url)
        url_host = host or parsed.hostname
        url_port = port or parsed.port
        if url_port is None and parsed.scheme:
            url_port = _DEFAULT_SCHEME_PORTS.get(parsed.scheme.lower())
        return url_host, url_port

    return host, port


async def probe_cache_service(
    provider: CacheProvider,
    endpoint: CacheServiceEndpoint,
    timeout: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    """
    Probe a cache service endpoint using the provider's declared health
    check scheme. Returns (reachable, error_message).
    """
    host, port = _endpoint_host_port(endpoint)
    if provider.health_check.target == "metrics":
        # metrics_url takes precedence over host+metrics_port, matching
        # the endpoint schema contract the exporter follows.
        if endpoint.metrics_url:
            parsed = urlparse(endpoint.metrics_url)
            host = parsed.hostname or host
            port = parsed.port or _DEFAULT_SCHEME_PORTS.get(
                (parsed.scheme or "http").lower()
            )
        else:
            port = endpoint.metrics_port
    if not host or not port:
        return False, "Endpoint host and port are not resolvable"

    scheme = (provider.health_check.scheme or "tcp").lower()
    if scheme == "http":
        return await _probe_http(host, port, provider.health_check.path, timeout)
    return await _probe_tcp(host, port, timeout)


async def _probe_tcp(
    host: str, port: int, timeout: float
) -> Tuple[bool, Optional[str]]:
    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        return True, None
    except Exception as e:
        return False, str(e) or e.__class__.__name__
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


async def _probe_http(
    host: str, port: int, path: Optional[str], timeout: float
) -> Tuple[bool, Optional[str]]:
    path = path or "/"
    if not path.startswith("/"):
        path = "/" + path
    url = f"http://{host}:{port}{path}"
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as client:
            async with client.get(url) as response:
                if 200 <= response.status < 400:
                    return True, None
                return False, f"HTTP status {response.status} from {url}"
    except Exception as e:
        return False, str(e) or e.__class__.__name__


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
    running = sorted(
        (
            instance
            for instance in instances
            if instance.state == CacheServiceStateEnum.RUNNING and instance.port
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

    ``worker`` is the instance's assigned worker. Node-local external
    endpoints resolve to it, so calls made before scheduling yield an
    explicit pending snapshot for such services.

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
    chunk_size = service.config.chunk_size if service.config else None
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
        # Mooncake's distributed pool) serve spanning instances by
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

    if service.mode == CacheServiceModeEnum.MANAGED:
        # Managed resolution is driven by the service's instances rather
        # than the service-level state: the aggregate may lag behind the
        # instances, and a partially-running per-node service can still
        # serve engines from its RUNNING instances.
        endpoint, reason = await _resolve_managed_endpoint(
            session, service, provider, worker
        )
        if endpoint is None:
            return CacheConfigSnapshot(
                **snapshot_base,
                injected=False,
                reason=reason,
            )
    elif service.state != CacheServiceStateEnum.RUNNING:
        return CacheConfigSnapshot(
            **snapshot_base,
            injected=False,
            reason=(
                f"Cache service is in state {service.state}; "
                "instance starts without shared KV cache"
            ),
        )
    else:
        endpoint = service.resolved_endpoint()

    # Declared password-typed values must not ride into the snapshot: it
    # lands on the model instance row, where the cache-service redaction
    # does not reach. The connector rendering below still sees the full
    # params — rendered env/args/files are what the worker actually needs.
    snapshot_endpoint = endpoint
    secret_names = {
        field.name for field in provider.external_fields if field.type == "password"
    }
    if endpoint and endpoint.params and secret_names:
        snapshot_endpoint = endpoint.model_copy(
            update={
                "params": {
                    key: value
                    for key, value in endpoint.params.items()
                    if key not in secret_names
                }
            }
        )

    backend = get_backend(model)
    render_params: Dict[str, Any] = {
        "host": endpoint.host,
        "port": endpoint.port,
        "chunk_size": chunk_size,
        "ram_size": service.config.ram_size if service.config else None,
        # The consuming instance's own worker IP: external connectors (e.g.
        # Mooncake) use it as the client identity / RDMA peer address, which
        # defaults to localhost and would be wrong across nodes.
        "local_hostname": worker.ip if worker and worker.ip else None,
        # Convenience alias for external connectors that take a single
        # host:port service address.
        "master_server_address": (
            f"{endpoint.host}:{endpoint.port}"
            if endpoint.host and endpoint.port
            else endpoint.url
        ),
    }
    # External-mode connection fields feed additional placeholders declared
    # by the provider; they never override the well-known keys above.
    # Declared field defaults backstop unset fields inside render_injection.
    for key, value in (endpoint.params or {}).items():
        render_params.setdefault(key, value)
    # Managed-mode field values feed the same namespace, so injection
    # templates may reference provider-declared fields too.
    for key, value in (
        (service.config.fields if service.config else None) or {}
    ).items():
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


class CacheServiceHealthChecker:
    """
    Periodically probes external cache services and flips their state
    between RUNNING and UNREACHABLE. Managed services are health-checked
    by the worker that runs them; only external services are probed here.
    """

    def __init__(self, interval: int = 30):
        self._interval = interval
        # DB rows are rewritten on state change; unchanged results are
        # persisted at most this often so last_check_at stays fresh without
        # flooding streaming watchers with no-op UPDATE events.
        self._unchanged_write_interval = timedelta(minutes=5)

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._check_external_services()
            except Exception as e:
                logger.error(f"Failed to check cache services: {e}")

    async def _check_external_services(self):
        async with async_session() as session:
            services = await CacheService.all_by_fields(
                session,
                fields={"mode": CacheServiceModeEnum.EXTERNAL},
                extra_conditions=[CacheService.deleted_at.is_(None)],
            )

        for service in services:
            try:
                await self._check_service(service)
            except Exception as e:
                logger.error(f"Failed to check cache service {service.name}: {e}")

    async def _check_service(self, service: CacheService):
        provider = get_cache_provider(service.provider_name)
        if provider is None:
            logger.warning(
                f"Skipping health check for cache service {service.name}: "
                f"unknown provider '{service.provider_name}'"
            )
            return

        endpoint = service.resolved_endpoint()
        reachable, message = await probe_cache_service(provider, endpoint)
        healthy = reachable
        new_state = (
            CacheServiceStateEnum.RUNNING
            if reachable
            else CacheServiceStateEnum.UNREACHABLE
        )

        now = datetime.now(timezone.utc)
        changed = service.state != new_state or service.healthy != healthy
        stale = (
            service.last_check_at is None
            or now - service.last_check_at >= self._unchanged_write_interval
        )
        if not changed and not stale:
            return

        async with async_session() as session:
            # Reload from DB and update health fields only, so concurrent
            # edits to other fields are not clobbered.
            to_update = await CacheService.one_by_id(session, service.id)
            if to_update is None or to_update.deleted_at is not None:
                return
            to_update.state = new_state
            to_update.state_message = message
            to_update.healthy = healthy
            to_update.last_check_at = now
            await to_update.update(session)

        if changed:
            logger.info(f"Marked external cache service {service.name} as {new_state}")


METRICS_FETCH_TIMEOUT_SECONDS = 5

METRICS_RETENTION_SECONDS = 24 * 3600
"""Samples older than this are deleted on each collector pass."""
