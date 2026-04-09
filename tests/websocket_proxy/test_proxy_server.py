"""
Tests for proxy_server module (HTTPSProxyServer).
"""

import asyncio
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
