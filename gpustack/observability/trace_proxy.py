"""Loopback OTLP/gRPC forwarder that proxies Worker/engine traces over Server HTTP."""

from concurrent import futures
import logging
from typing import Mapping, Optional

import grpc
import httpx

from gpustack.ssl_context import make_ssl_context
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)

_OTLP_EXPORT_METHOD = "/opentelemetry.proto.collector.trace.v1.TraceService/Export"
_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_FORWARD_TIMEOUT_SECONDS = 5.0
_HTTP_TO_GRPC_STATUS = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.PERMISSION_DENIED,
    403: grpc.StatusCode.PERMISSION_DENIED,
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    415: grpc.StatusCode.INVALID_ARGUMENT,
}


class _RawOTLPTraceHandler(grpc.GenericRpcHandler):
    """Route OTLP TraceService/Export without deserializing protobuf spans."""

    def __init__(self, export_fn):
        self._handler = grpc.unary_unary_rpc_method_handler(
            export_fn,
            request_deserializer=lambda data: data,
            response_serializer=lambda data: data,
        )

    def service(self, handler_call_details):
        if handler_call_details.method == _OTLP_EXPORT_METHOD:
            return self._handler
        return None


class LocalOTLPTraceProxy:
    """Receive local OTLP/gRPC batches on loopback and POST them to the Server."""

    def __init__(
        self,
        server_url: str,
        headers: Mapping[str, str],
        *,
        port: int = 0,
        client: Optional[httpx.Client] = None,
    ):
        self._ingest_url = f"{server_url.rstrip('/')}/v2/traces/otlp"
        self._headers = {**headers, "Content-Type": "application/x-protobuf"}
        self._requested_port = port
        self._owns_client = client is None
        self._client = client
        self._server: Optional[grpc.Server] = None
        self._endpoint: Optional[str] = None

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint

    def start(self) -> str:
        if self._server is not None and self._endpoint is not None:
            return self._endpoint
        server = grpc.server(
            futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="otlp-trace-proxy"
            ),
            maximum_concurrent_rpcs=64,
            options=[("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES)],
        )
        server.add_generic_rpc_handlers((_RawOTLPTraceHandler(self._export),))
        bound_port = server.add_insecure_port(f"127.0.0.1:{self._requested_port}")
        if not bound_port:
            raise RuntimeError("Failed to bind local OTLP trace proxy on loopback")
        if self._client is None:
            self._client = httpx.Client(
                headers=self._headers,
                verify=make_ssl_context(),
                trust_env=use_proxy_env_for_url(self._ingest_url),
                timeout=_FORWARD_TIMEOUT_SECONDS,
            )
        server.start()
        self._server = server
        self._endpoint = f"127.0.0.1:{bound_port}"
        logger.info("Started local OTLP trace proxy on %s.", self._endpoint)
        return self._endpoint

    def stop(self) -> None:
        server, self._server = self._server, None
        self._endpoint = None
        if server is not None:
            server.stop(grace=1).wait()
        client = self._client
        if self._owns_client and client is not None:
            self._client = None
            client.close()

    def _export(self, request_bytes: bytes, context: grpc.ServicerContext) -> bytes:
        if len(request_bytes) > _MAX_MESSAGE_BYTES:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "Trace batch exceeds size limit"
            )
        client = self._client
        if client is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Trace proxy is not running")
        try:
            response = client.post(
                self._ingest_url,
                content=request_bytes,
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.debug("Failed to forward OTLP trace batch to Server: %s", exc)
            context.abort(grpc.StatusCode.UNAVAILABLE, "Server trace ingest failed")
        if 200 <= response.status_code < 300:
            return response.content
        logger.debug(
            "Server trace ingest rejected batch with status %s.", response.status_code
        )
        status_code = _HTTP_TO_GRPC_STATUS.get(
            response.status_code, grpc.StatusCode.UNAVAILABLE
        )
        context.abort(
            status_code,
            f"Server trace ingest returned {response.status_code}",
        )
        return b""
