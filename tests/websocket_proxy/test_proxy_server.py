"""
Tests for proxy_server module (HTTPSProxyServer).
"""

import asyncio
from contextlib import asynccontextmanager
import pytest
import uuid
import uvicorn
from fastapi import FastAPI
from gpustack.websocket_proxy.proxy_server import HTTPSProxyServer
from gpustack.websocket_proxy.message_server import MessageServerHandler, router


def get_free_port(host: str) -> int:
    """Get a free port on the host"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def start_websocket_server(
    message_handler: MessageServerHandler, host: str, port: int
):
    """Start WebSocket server using uvicorn"""
    app = FastAPI()
    app.state.message_server_handler = message_handler
    app.include_router(router)

    actual_port = get_free_port(host) if port == 0 else port

    config = uvicorn.Config(app, host=host, port=actual_port, log_level="error")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.5)

    return server, server_task, actual_port


class TestProxyAuthenticator:
    """Test HTTPSProxyServer authenticator functionality."""

    @pytest.mark.asyncio
    async def test_no_authenticator_passes(self):
        """Test that requests pass when no authenticator is configured."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        _, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Proxy without authenticator
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
            authenticator=None,  # No authenticator
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test server
        async def handle_request(reader, writer):
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect client
        from gpustack.websocket_proxy.message_client import MessageClient

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        try:
            # Connect through proxy using CONNECT tunnel
            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            connect_request = f"CONNECT 127.0.0.1:{test_port} HTTP/1.1\r\nHost: 127.0.0.1:{test_port}\r\n\r\n"
            writer.write(connect_request.encode())
            await writer.drain()

            connect_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=5.0
            )
            assert connect_response.startswith(b"HTTP/1.1 200")

            # Make HTTP request through tunnel
            request = f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{test_port}\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()

            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200 OK" in response

            writer.close()
            await writer.wait_closed()
        finally:
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_authenticator_allows_valid_request(self):
        """Test that authenticator returning True allows the request."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        _, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Authenticator that allows requests with valid auth header
        async def auth_check(headers: dict) -> bool:
            return headers.get("authorization") == "Bearer valid_token"

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
            authenticator=auth_check,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test server
        async def handle_request(reader, writer):
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect client
        from gpustack.websocket_proxy.message_client import MessageClient

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        try:
            # Connect through proxy using CONNECT tunnel with auth
            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            connect_request = (
                f"CONNECT 127.0.0.1:{test_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{test_port}\r\n"
                f"Authorization: Bearer valid_token\r\n\r\n"
            )
            writer.write(connect_request.encode())
            await writer.drain()

            connect_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=5.0
            )
            assert connect_response.startswith(b"HTTP/1.1 200")

            # Make HTTP request through tunnel (auth already validated by CONNECT)
            request = f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{test_port}\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()

            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200 OK" in response, f"Expected 200 OK, got: {response}"

            writer.close()
            await writer.wait_closed()
        finally:
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_authenticator_rejects_invalid_request(self):
        """Test that authenticator returning False returns 401 Unauthorized."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Authenticator that rejects requests
        async def auth_check(headers: dict) -> bool:
            return headers.get("authorization") == "Bearer valid_token"

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
            authenticator=auth_check,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Connect client
        from gpustack.websocket_proxy.message_client import MessageClient

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        try:
            # Make request WITHOUT valid auth
            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            request = "GET / HTTP/1.1\r\nHost: 127.0.0.1:9999\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()

            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"401" in response, f"Expected 401 Unauthorized, got: {response}"

            writer.close()
            await writer.wait_closed()
        finally:
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_authenticator_with_basic_auth(self):
        """Test authenticator with Basic authentication scheme."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        _, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        import base64

        async def auth_check(headers: dict) -> bool:
            auth = headers.get("authorization", "")
            if auth.lower().startswith("basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode("utf-8")
                    return decoded == "admin:secret"
                except Exception:
                    return False
            return False

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
            authenticator=auth_check,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test server
        async def handle_request(reader, writer):
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect client
        from gpustack.websocket_proxy.message_client import MessageClient

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        try:
            # Connect through proxy using CONNECT tunnel with Basic auth
            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            credentials = base64.b64encode(b"admin:secret").decode("utf-8")
            connect_request = (
                f"CONNECT 127.0.0.1:{test_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{test_port}\r\n"
                f"Authorization: Basic {credentials}\r\n\r\n"
            )
            writer.write(connect_request.encode())
            await writer.drain()

            connect_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=5.0
            )
            assert connect_response.startswith(b"HTTP/1.1 200")

            # Make HTTP request through tunnel (auth already validated by CONNECT)
            request = f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{test_port}\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()

            response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            assert b"200 OK" in response, f"Expected 200 OK, got: {response}"

            writer.close()
            await writer.wait_closed()
        finally:
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass

            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass


class TestHTTPProxy:
    """Test HTTP proxy direct forwarding (not CONNECT tunnel)."""

    @pytest.mark.asyncio
    async def test_http_proxy_with_content_length(self):
        """Test HTTP proxy with Content-Length request body (direct TCP forwarding)."""

        # Direct connection manager - creates direct TCP connections without WebSocket tunnel
        def direct_connection_manager_getter(_target_ip: str):
            from gpustack.websocket_proxy.connection_manager import ConnectionManager

            return ConnectionManager(websocket=None)

        # HTTP proxy with direct TCP connection
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=direct_connection_manager_getter,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Track received body
        received_body = None

        # Start test HTTP server
        async def handle_request(reader, writer):
            nonlocal received_body
            try:
                # Read all request data
                request_data = b""
                while True:
                    chunk = await reader.read(8192)
                    if not chunk:
                        break
                    request_data += chunk
                    # Check if we have complete request (headers + body)
                    if b"\r\n\r\n" in request_data:
                        # Parse Content-Length from headers
                        header_end = request_data.find(b"\r\n\r\n")
                        headers = request_data[:header_end].decode(
                            "utf-8", errors="ignore"
                        )
                        content_length = 0
                        for line in headers.split("\r\n"):
                            if line.lower().startswith("content-length:"):
                                content_length = int(line.split(":")[1].strip())
                                break
                        # Check if we have complete body
                        body_start = header_end + 4
                        body_len = len(request_data) - body_start
                        if body_len >= content_length:
                            received_body = request_data[
                                body_start : body_start + content_length
                            ]
                            break
                    if len(request_data) > 65536:
                        break

                # Send response
                response_body = b"OK"
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Length: {len(response_body)}\r\n"
                    f"\r\n"
                ).encode() + response_body
                writer.write(response)
                await writer.drain()
            except Exception as e:
                print(f"[Test Server] Error: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        try:
            # Send HTTP request directly through proxy (not CONNECT tunnel)
            post_body = b"This is the POST body via HTTP proxy"
            request = (
                f"POST http://127.0.0.1:{test_port}/ HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{test_port}\r\n"
                f"Content-Length: {len(post_body)}\r\n"
                f"\r\n"
            ).encode() + post_body

            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            writer.write(request)
            await writer.drain()

            # Read response
            response = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            assert b"200 OK" in response, f"Expected 200 OK, got: {response}"
            assert (
                received_body == post_body
            ), f"Body mismatch: {received_body!r} != {post_body!r}"

            writer.close()
            await writer.wait_closed()
        finally:
            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()

    @pytest.mark.asyncio
    async def test_http_proxy_with_chunked_body(self):  # noqa C901
        """Test HTTP proxy with chunked transfer encoding request body (direct TCP forwarding)."""

        # Direct connection manager - creates direct TCP connections without WebSocket tunnel
        def direct_connection_manager_getter(_target_ip: str):
            from gpustack.websocket_proxy.connection_manager import ConnectionManager

            return ConnectionManager(websocket=None)

        # HTTP proxy with direct TCP connection
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=direct_connection_manager_getter,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Track received body
        received_body = None

        # Start test HTTP server that handles chunked body
        async def handle_chunked_request(reader, writer):
            nonlocal received_body
            import logging

            logger = logging.getLogger("test_server")
            try:
                logger.debug("[Test Server] Starting to read headers")
                # Read headers
                header_data = b""
                while b"\r\n\r\n" not in header_data:
                    chunk = await reader.read(8192)
                    if not chunk:
                        logger.debug(
                            "[Test Server] Connection closed while reading headers"
                        )
                        break
                    header_data += chunk

                logger.debug(f"[Test Server] Received headers: {header_data[:200]!r}")
                header_end = header_data.find(b"\r\n\r\n")
                if header_end < 0:
                    logger.debug("[Test Server] Headers incomplete, returning")
                    return

                headers = header_data[:header_end].decode("utf-8", errors="ignore")
                body_data = header_data[header_end + 4 :]
                logger.debug(
                    f"[Test Server] Headers complete, body_data: {body_data!r}"
                )

                # Check for Transfer-Encoding: chunked
                is_chunked = any(
                    "transfer-encoding" in line.lower() and "chunked" in line.lower()
                    for line in headers.split("\r\n")
                )
                logger.debug(f"[Test Server] is_chunked: {is_chunked}")

                if is_chunked:
                    # Decode chunked body
                    body = b""
                    buffer = body_data  # Use buffer to track unprocessed data

                    async def read_until_crlf():
                        """Read from buffer or socket until we have a complete line ending with CRLF."""
                        nonlocal buffer
                        while b"\r\n" not in buffer:
                            c = await reader.read(1)
                            if not c:
                                return False  # EOF
                            buffer += c
                        return True

                    while True:
                        logger.debug(f"[Test Server] Buffer: {buffer!r}")
                        # Read chunk size line
                        logger.debug("[Test Server] Reading chunk size line")
                        if not await read_until_crlf():
                            logger.debug("[Test Server] EOF while reading chunk size")
                            break

                        line_end = buffer.find(b"\r\n")
                        line = buffer[:line_end]
                        buffer = buffer[line_end + 2 :]
                        logger.debug(f"[Test Server] Chunk size line: {line!r}")
                        chunk_size = int(line.strip(), 16)
                        if chunk_size == 0:
                            logger.debug(
                                "[Test Server] Got chunk size 0, chunked body complete"
                            )
                            break

                        # Read chunk data
                        logger.debug(
                            f"[Test Server] Need {chunk_size} bytes of chunk data, buffer has {len(buffer)}"
                        )
                        while len(buffer) < chunk_size:
                            needed = chunk_size - len(buffer)
                            c = await reader.read(needed)
                            if not c:
                                logger.debug(
                                    "[Test Server] EOF while reading chunk data"
                                )
                                break
                            buffer += c

                        chunk = buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        body += chunk
                        logger.debug(f"[Test Server] Read chunk: {chunk!r}")

                        # Read trailing \r\n after chunk
                        if not await read_until_crlf():
                            logger.debug(
                                "[Test Server] EOF while reading chunk terminator"
                            )
                            break

                        buffer = buffer[2:]  # Skip the \r\n

                    received_body = body
                    logger.debug(
                        f"[Test Server] Chunked body complete: {received_body!r}"
                    )
                else:
                    received_body = body_data
                    logger.debug(f"[Test Server] Non-chunked body: {received_body!r}")

                print(f"[Test Server] Received body: {received_body!r}")

                # Send response
                response_body = b"OK"
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Length: {len(response_body)}\r\n"
                    f"\r\n"
                ).encode() + response_body
                logger.debug("[Test Server] Sending response")
                writer.write(response)
                await writer.drain()
                logger.debug("[Test Server] Response sent, draining")
            except Exception as e:
                logger.exception(f"[Test Server] Error: {e}")
            finally:
                logger.debug("[Test Server] Closing connection")
                writer.close()
                await writer.wait_closed()
                logger.debug("[Test Server] Connection closed")

        test_server = await asyncio.start_server(handle_chunked_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        try:
            # Send HTTP request with chunked body
            post_body = b"Hello, chunked world!"
            # Chunked body format: <hex size>\r\n<data>\r\n...0\r\n\r\n
            chunked_body = (
                f"{len(post_body):x}\r\n".encode() + post_body + b"\r\n0\r\n\r\n"
            )

            request = (
                f"POST http://127.0.0.1:{test_port}/ HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{test_port}\r\n"
                f"Transfer-Encoding: chunked\r\n"
                f"\r\n"
            ).encode() + chunked_body

            reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
            writer.write(request)
            await writer.drain()

            # Read response
            response = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            assert b"200 OK" in response, f"Expected 200 OK, got: {response}"
            assert (
                received_body == post_body
            ), f"Body mismatch: {received_body!r} != {post_body!r}"

            writer.close()
            await writer.wait_closed()
        finally:
            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass

            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()


class TestMetricsAuthentication:
    """Metrics scrapes through the forward proxy must authenticate.

    The proxy reaches hosts the caller cannot (tunnel workers), so a
    request that names its own target — any path, ``/metrics`` included —
    must carry a credential before the proxy connects anywhere.
    """

    def _direct_connection_manager_getter(self, _target_ip: str):
        from gpustack.websocket_proxy.connection_manager import ConnectionManager

        return ConnectionManager(websocket=None)

    async def _start_proxy(self, authenticator):
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=self._direct_connection_manager_getter,
            authenticator=authenticator,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)
        proxy_addr = proxy_server.server.sockets[0].getsockname()
        return proxy_server, proxy_task, proxy_addr[0], proxy_addr[1]

    @staticmethod
    async def _get(
        proxy_port: int, target_port: int, path: str, headers: dict
    ) -> bytes:
        """Send an absolute-URI GET to the proxy for ``target_port``."""
        request = f"GET http://127.0.0.1:{target_port}{path} HTTP/1.1\r\n"
        request += f"Host: 127.0.0.1:{target_port}\r\n"
        for key, value in headers.items():
            request += f"{key}: {value}\r\n"
        request += "\r\n"
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(request.encode())
        await writer.drain()
        response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        for line in response.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                response += await asyncio.wait_for(
                    reader.readexactly(int(line.split(b":", 1)[1])), timeout=5.0
                )
                break
        writer.close()
        await writer.wait_closed()
        return response

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/metrics",
            "/metrics?debug=1",
        ],
    )
    async def test_unauthenticated_metrics_scrape_is_rejected(self, path):
        """A GET whose URI path is /metrics still requires credentials."""

        async def deny_all(headers: dict) -> bool:
            return False

        proxy_server, proxy_task, _, proxy_port = await self._start_proxy(deny_all)

        try:
            # A live target proves the rejection happens before any dial.
            test_server = await asyncio.start_server(
                self._serve_metrics, "127.0.0.1", 0
            )
            test_port = test_server.sockets[0].getsockname()[1]

            response = await self._get(proxy_port, test_port, path, {})
            assert response.startswith(
                b"HTTP/1.1 401"
            ), f"Expected 401 for {path}, got: {response!r}"

            test_server.close()
            await test_server.wait_closed()
        finally:
            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass
            await proxy_server.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "auth_header",
        [
            {"Proxy-Authorization": "Bearer scrape-token"},
            {"Authorization": "Bearer scrape-token"},
        ],
    )
    async def test_authenticated_metrics_scrape_is_relayed(self, auth_header):
        """A credential the authenticator accepts still reaches the target."""

        async def accept_configured_token(headers: dict) -> bool:
            return (
                headers.get("proxy-authorization") or headers.get("authorization")
            ) == "Bearer scrape-token"

        proxy_server, proxy_task, _, proxy_port = await self._start_proxy(
            accept_configured_token
        )

        test_server = await asyncio.start_server(self._serve_metrics, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        try:
            response = await self._get(proxy_port, test_port, "/metrics", auth_header)
            assert response.startswith(
                b"HTTP/1.1 200"
            ), f"Expected relayed 200, got: {response!r}"
            assert b"gpustack_worker_up 1" in response
        finally:
            proxy_task.cancel()
            try:
                await proxy_task
            except asyncio.CancelledError:
                pass
            await proxy_server.stop()
            test_server.close()
            await test_server.wait_closed()

    @staticmethod
    async def _serve_metrics(reader, writer):
        # Drain the request head before responding: closing a socket with
        # unread data sends a TCP reset, which discards the response the
        # relay has not read back yet.
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break

        body = b"gpustack_worker_up 1\n"
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: text/plain; version=0.0.4\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()


class TestHeaderRedaction:
    """The trace-level header log must not record credentials."""

    @pytest.mark.parametrize(
        "header",
        [
            "authorization",
            "proxy-authorization",
            "Authorization",
            "Proxy-Authorization",
        ],
    )
    def test_credential_headers_are_masked(self, header):
        from gpustack.websocket_proxy.proxy_server import _redact_headers

        headers = {header: "Bearer gpustack_secret_value"}
        redacted = _redact_headers(headers)

        assert redacted[header] == "<redacted>"
        # The caller's dict is left untouched: the real headers still
        # travel with the request.
        assert headers[header] == "Bearer gpustack_secret_value"

    def test_other_headers_pass_through(self):
        from gpustack.websocket_proxy.proxy_server import _redact_headers

        headers = {"host": "127.0.0.1:8080", "accept": "text/plain"}
        assert _redact_headers(headers) == headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['rank', 'invalid', 'connect'])
async def test_model_target_is_resolved_once_without_uri_fallback(monkeypatch, case):
    from unittest.mock import AsyncMock, MagicMock
    from fastapi import HTTPException
    from gpustack.websocket_proxy import proxy_server as module

    reader = asyncio.StreamReader()
    method = 'CONNECT' if case == 'connect' else 'GET'
    reader.feed_data(
        f'{method} http://other-worker:49999/v1/models HTTP/1.1\r\nX-GPUStack-Model-Instance: model-1-7-dp1.static\r\n\r\n'.encode()
    )
    reader.feed_eof()
    connection = MagicMock(send_error=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(module, 'AsyncIOConnection', lambda **kwargs: connection)
    route = AsyncMock(return_value=('10.0.0.2', 41001))
    if case == 'invalid':
        route.side_effect = HTTPException(status_code=404, detail='Unavailable rank')
    getter = MagicMock(return_value=object())
    forward = AsyncMock()
    monkeypatch.setattr(module, '_handle_http', forward)
    proxy = HTTPSProxyServer('127.0.0.1', 0, getter, header_router=route)
    await proxy._handle_client(reader, MagicMock())
    route.assert_awaited_once()
    if case == 'rank':
        getter.assert_called_once_with('10.0.0.2')
        assert forward.call_args.args[-1] == ('10.0.0.2', 41001)
    else:
        getter.assert_not_called()
        forward.assert_not_awaited()
        assert connection.send_error.call_args.args[0] == (
            404 if case == 'invalid' else 400
        )
    connection.close.assert_awaited_once()


@asynccontextmanager
async def _component_tunnel_client(port):
    import httpx
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gpustack.websocket_proxy.connection import AsyncIOConnection

    class ChunkConnection(AsyncIOConnection):
        async def read(self, n=-1, timeout=None):
            return await super().read(65536 if n < 0 else n, timeout)

    async def connect(target):
        assert target == f'tcp://127.0.0.1:{port}'
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        return ChunkConnection(reader, writer)

    target = ('127.0.0.1', port)
    route = AsyncMock(return_value=target)
    proxy = HTTPSProxyServer(
        '127.0.0.1', 0, lambda ip: SimpleNamespace(connect=connect), header_router=route
    )
    async with await asyncio.start_server(
        proxy._handle_client, '127.0.0.1', 0
    ) as server:
        proxy_port = server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient(
            proxy=f'http://127.0.0.1:{proxy_port}', timeout=5
        ) as client:
            yield client, route, proxy_port


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['v1/chat/completions', 'generate'])
@pytest.mark.parametrize('chunked', [False, True])
async def test_tunnel_preserves_native_rank_payload(path, chunked):
    import json
    from aiohttp import web
    from tests.routes.test_worker_proxy import _fake_inference_server

    received = []

    async def handler(request):
        raw = await request.read()
        received.append(
            (
                json.loads(raw),
                {key.lower(): value for key, value in request.headers.items()},
                raw,
            )
        )
        return web.json_response({'ok': True})

    async with _fake_inference_server(handler) as port:
        async with _component_tunnel_client(port) as (client, route, _):
            raw = json.dumps(
                {'prompt': '你好', 'routed_dp_rank': 9, 'data_parallel_rank': 8},
                ensure_ascii=False,
            ).encode()

            async def chunks():
                yield raw[:3]
                yield raw[3:]

            response = await client.post(
                f'http://unselected-worker:49999/{path}',
                content=chunks() if chunked else raw,
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'X-Data-Parallel-Rank': '9',
                    'X-SMG-Routing-Key': 'conversation-123',
                },
            )
    assert response.status_code == 200
    data, headers, sent = received[0]
    assert headers['x-smg-routing-key'] == 'conversation-123'
    assert data == {'prompt': '你好', 'routed_dp_rank': 9, 'data_parallel_rank': 8}
    assert sent == raw
    assert headers['x-data-parallel-rank'] == '9'
    if chunked:
        assert headers['transfer-encoding'] == 'chunked'
    else:
        assert int(headers['content-length']) == len(raw)
    route.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('chunked', [False, True])
@pytest.mark.parametrize('body', [b'not-json', b'x' * 33])
async def test_tunnel_preserves_body_and_backend_error(chunked, body):
    from aiohttp import web
    from tests.routes.test_worker_proxy import _fake_inference_server

    received = []
    error = {'error': {'message': 'native backend validation error'}}

    async def handler(request):
        received.append(await request.read())
        return web.json_response(error, status=422)

    async def chunks():
        yield body[:3]
        yield body[3:]

    async with _fake_inference_server(handler) as port:
        async with _component_tunnel_client(port) as (client, route, _):
            response = await client.post(
                'http://unselected/v1/chat/completions',
                content=chunks() if chunked else body,
                headers={'Content-Type': 'application/octet-stream'},
            )
    assert received == [body]
    assert response.status_code == 422
    assert response.json() == error
    route.assert_awaited_once()


@pytest.mark.asyncio
async def test_component_tunnel_disconnect_releases_upstream_stream():
    from aiohttp import web
    from tests.routes.test_worker_proxy import _fake_inference_server

    closed = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'data: first\n\n')
        while request.transport is not None and not request.transport.is_closing():
            await asyncio.sleep(0.01)
        closed.set()
        return response

    async with _fake_inference_server(handler) as port:
        async with _component_tunnel_client(port) as (client, route, _):
            async with client.stream(
                'POST',
                'http://unselected/v1/chat/completions',
                json={'stream': True, 'routed_dp_rank': 9},
            ) as response:
                assert response.status_code == 200
                async for chunk in response.aiter_bytes():
                    assert b'data: first' in chunk
                    break
            await asyncio.wait_for(closed.wait(), 2)
    route.assert_awaited_once()
