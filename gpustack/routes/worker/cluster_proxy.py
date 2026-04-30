import asyncio
import functools
import logging
import os
import ssl
from pathlib import Path
from typing import Optional, Tuple

import aiohttp
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import (
    InternalServerErrorException,
)


router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)


SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
TOKEN_PATH = SERVICE_ACCOUNT_DIR / "token"
CA_PATH = SERVICE_ACCOUNT_DIR / "ca.crt"

# Hop-by-hop and connection-control headers we never forward in either direction.
_REQUEST_HEADER_SKIP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    # Drop the inbound auth — the API server is reached with the SA token below.
    "authorization",
    "cookie",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
}
_RESPONSE_HEADER_SKIP = {
    "transfer-encoding",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
}


_session_lock = asyncio.Lock()


def _read_token() -> str:
    return TOKEN_PATH.read_text().strip()


@functools.lru_cache(maxsize=1)
def _resolve_kube_target() -> Tuple[str, ssl.SSLContext]:
    # Cached for the lifetime of the worker process: KUBERNETES_SERVICE_HOST/PORT
    # and the cluster CA certificate are static for a pod, and building an
    # SSLContext (parses CA, sets up trust store) is non-trivial. The
    # ServiceAccount token is *not* cached here — kubelet rotates the projected
    # token file in place, so it is re-read per request via _read_token().
    # lru_cache does not memoize raised exceptions, so transient setup errors
    # (e.g. token file briefly missing during pod start) will be retried.
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS") or os.environ.get(
        "KUBERNETES_SERVICE_PORT"
    )
    if not host or not port:
        raise InternalServerErrorException(
            message=(
                "Worker is not running inside a Kubernetes pod: "
                "KUBERNETES_SERVICE_HOST/PORT environment variables are not set."
            )
        )
    if not TOKEN_PATH.is_file() or not CA_PATH.is_file():
        raise InternalServerErrorException(
            message=(
                "Worker pod ServiceAccount credentials not found at "
                f"{SERVICE_ACCOUNT_DIR}; ensure automountServiceAccountToken is enabled."
            )
        )
    # IPv6 host literal needs brackets in URL.
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    base_url = f"https://{host}:{port}"
    ssl_ctx = ssl.create_default_context(cafile=str(CA_PATH))
    return base_url, ssl_ctx


async def _get_kube_session(request: Request) -> aiohttp.ClientSession:
    session: Optional[aiohttp.ClientSession] = getattr(
        request.app.state, "kube_api_session", None
    )
    if session is not None and not session.closed:
        return session

    async with _session_lock:
        session = getattr(request.app.state, "kube_api_session", None)
        if session is not None and not session.closed:
            return session

        _, ssl_ctx = _resolve_kube_target()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=64)
        # No explicit total timeout — keep watch / log streams open until the
        # client disconnects. Connect timeout still applies.
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
        )
        request.app.state.kube_api_session = session
        return session


@router.api_route(
    "/cluster-proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def cluster_proxy(path: str, request: Request):
    """
    Forward an HTTP request to the in-cluster Kubernetes API server using the
    pod's ServiceAccount credentials.

    Designed to be invoked by the GPUStack server through the standard
    server→worker request channel. The server is responsible for
    authenticating the original caller; this endpoint trusts the worker
    bearer token (see worker_auth dependency).
    """
    base_url, _ = _resolve_kube_target()
    target_url = f"{base_url}/{path}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _REQUEST_HEADER_SKIP
    }
    headers["Authorization"] = f"Bearer {_read_token()}"

    # Stream the request body through to avoid buffering large payloads
    # (e.g. apply of big manifests) in worker memory.
    body = (
        request.stream() if request.method not in ("GET", "HEAD", "OPTIONS") else None
    )

    params = list(request.query_params.multi_items()) or None

    session = await _get_kube_session(request)
    resp = await session.request(
        method=request.method,
        url=target_url,
        headers=headers,
        data=body,
        params=params,
        allow_redirects=False,
    )

    async def streamer():
        try:
            async for chunk in resp.content.iter_any():
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "cluster-proxy stream interrupted for %s %s: %s",
                request.method,
                target_url,
                e,
            )
        finally:
            resp.release()

    response_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _RESPONSE_HEADER_SKIP
    }

    return StreamingResponse(
        streamer(),
        status_code=resp.status,
        headers=response_headers,
        media_type=resp.headers.get("Content-Type"),
    )
