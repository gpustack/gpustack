"""HTTP client for the ``gpustack-operator`` worker-gateway.

The gpustack-operator's worker gateway serves the ``/apis/*`` routes documented in
``github.com/gpustack-operator/pkg/workergateway/service``.

It binds to a TLS-on-unix socket whose path is generated per server startup
(see ``gpustack.server.server.Server._enqueue_operator_process``) and handed to
this module via :func:`set_unix_path`.

The certificate is self-signed by the gpustack-operator at startup,
so the client connects with hostname/cert verification disabled — the unix-socket boundary is the trust boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import aiohttp
from aiohttp.connector import UnixClientConnectorError
from aiohttp.helpers import ceil_timeout

logger = logging.getLogger(__name__)

# The HTTP host portion of the URL is irrelevant when the connection is
# routed through a unix-socket connector, but aiohttp still requires a
# syntactically valid URL.
_BASE_URL = "https://gpustack-operator/apis"

# aiohttp's default session timeout is ``ClientTimeout(total=300)`` — a 5-minute
# overall deadline that starts at connect and fires regardless of whether data
# is flowing. That is fine for the short request/response helpers, but a watch
# stream is meant to stay open indefinitely, so the total deadline tears a
# healthy stream down every 5 minutes (surfacing as a ``TimeoutError`` on
# ``readline``). Drop the overall deadline for streaming requests; keep the
# bounded connect timeout so a broken socket still fails fast and reconnects.
_WATCH_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30)

_unix_path: Optional[str] = None


def set_gateway_unix_path(path: str) -> None:
    """Record the unix-socket path the operator's worker gateway is bound to.

    Called once at server startup, before the operator subprocess is spawned.
    """
    global _unix_path
    _unix_path = path


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _UnixTLSConnector(aiohttp.UnixConnector):
    """UnixConnector that wraps the unix-socket transport in TLS.

    aiohttp's stock UnixConnector ignores ``ssl=`` on the request and calls
    ``loop.create_unix_connection`` without an SSL context, so requests against
    an ``https://`` URL end up as plain HTTP on the wire. The operator's worker
    gateway terminates TLS on the unix socket, so we need to hand the SSL
    context to ``create_unix_connection`` directly.
    """

    def __init__(self, path: str, ssl_context: ssl.SSLContext) -> None:
        super().__init__(path=path)
        self._ssl_context = ssl_context

    async def _create_connection(self, req, traces, timeout):  # type: ignore[override]
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                _, proto = await self._loop.create_unix_connection(
                    self._factory,
                    self._path,
                    ssl=self._ssl_context,
                    server_hostname="localhost",
                )
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc
        return proto


@asynccontextmanager
async def _session() -> AsyncIterator[aiohttp.ClientSession]:
    if not _unix_path:
        raise RuntimeError(
            "operator worker gateway unix path is not configured; "
            "is gpustack-operator running?"
        )
    connector = _UnixTLSConnector(path=_unix_path, ssl_context=_ssl_context())
    async with aiohttp.ClientSession(connector=connector) as session:
        yield session


def _query(
    clusters: Optional[list[str]] = None,
    token: Optional[str] = None,
    namespace: Optional[str] = None,
    watch: bool = False,
    aggregated: bool = False,
    force: bool = False,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for c in clusters or []:
        params.append(("cluster", c))
    if token:
        params.append(("token", token))
    if namespace:
        params.append(("namespace", namespace))
    if watch:
        params.append(("watch", "true"))
    if aggregated:
        params.append(("aggregated", "true"))
    if force:
        params.append(("force", "true"))
    return params


async def subscribe_worker(
    cluster: str,
    token: str,
    force: bool = False,
    gvk: Optional[list[tuple[str, str, str]]] = None,
) -> None:
    """POST /apis/workers — subscribe a worker cluster to the gateway."""
    async with _session() as session:
        params = _query(clusters=[cluster], token=token, force=force)
        body = {"gvks": [f"{g}/{v}/{k}" if g else f"{v}/{k}" for g, v, k in gvk or []]}
        async with session.post(
            f"{_BASE_URL}/workers", params=params, json=body
        ) as resp:
            resp.raise_for_status()


async def unsubscribe_worker(
    cluster: str,
) -> None:
    """DELETE /apis/workers — unsubscribe a worker cluster from the gateway."""
    async with _session() as session:
        params = _query(clusters=[cluster])
        async with session.delete(f"{_BASE_URL}/workers", params=params) as resp:
            resp.raise_for_status()


async def list_instance_type_flavors(
    clusters: Optional[list[str]] = None,
    aggregated: bool = False,
) -> dict[str, Any]:
    """GET /apis/instancetypeflavors — list InstanceTypeFlavors across one or more clusters."""
    return await _get_json(
        "/instancetypeflavors", _query(clusters=clusters, aggregated=aggregated)
    )


async def watch_instance_type_flavors(
    clusters: Optional[list[str]] = None,
    aggregated: bool = False,
) -> AsyncIterator[str]:
    """GET /apis/instancetypeflavors?watch=true — stream InstanceTypeFlavor change events."""
    params = _query(clusters=clusters, aggregated=aggregated, watch=True)
    async for evt in _stream("/instancetypeflavors", params):
        yield evt


async def list_instance_types(
    clusters: Optional[list[str]] = None,
    aggregated: bool = False,
) -> dict[str, Any]:
    """GET /apis/instancetypes — list InstanceTypes across one or more clusters."""
    return await _get_json(
        "/instancetypes", _query(clusters=clusters, aggregated=aggregated)
    )


async def watch_instance_types(
    clusters: Optional[list[str]] = None,
    aggregated: bool = False,
) -> AsyncIterator[str]:
    """GET /apis/instancetypes?watch=true — stream InstanceType change events."""
    params = _query(clusters=clusters, aggregated=aggregated, watch=True)
    async for evt in _stream("/instancetypes", params):
        yield evt


async def list_instances(
    clusters: Optional[list[str]] = None,
    namespace: Optional[str] = None,
) -> dict[str, Any]:
    """GET /apis/instances — list Instances across one or more clusters."""
    return await _get_json("/instances", _query(clusters=clusters, namespace=namespace))


async def watch_instances(
    clusters: Optional[list[str]] = None,
    namespace: Optional[str] = None,
) -> AsyncIterator[str]:
    """GET /apis/instances?watch=true — stream Instance change events."""
    params = _query(clusters=clusters, namespace=namespace, watch=True)
    async for evt in _stream("/instances", params):
        yield evt


async def _get_json(path: str, params: list[tuple[str, str]]) -> dict[str, Any]:
    async with _session() as session:
        async with session.get(f"{_BASE_URL}{path}", params=params) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _stream(path: str, params: list[tuple[str, str]]) -> AsyncIterator[str]:
    """Consume a newline-delimited JSON stream from the worker gateway and
    re-emit each event using the project's ``<json>\\n\\n`` framing.

    The gateway encodes each ``manager.WorkerEvent`` with ``json.NewEncoder``
    which writes a single JSON object followed by ``\\n`` per event. We
    validate each line is well-formed JSON, then forward it with the same
    framing produced by :meth:`gpustack.mixins.active_record.streaming` so
    downstream clients can consume both kinds of streams uniformly.
    """
    async with _session() as session:
        async with session.get(
            f"{_BASE_URL}{path}", params=params, timeout=_WATCH_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            while True:
                line = await resp.content.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("discarding malformed event from %s: %r", path, line)
                    continue
                yield line.decode("utf-8") + "\n\n"
