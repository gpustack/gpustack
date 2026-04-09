"""
Test suite for WebSocket proxy benchmark functionality.

Tests WebSocket proxy over HTTP tunnel with various scenarios.
"""

import asyncio
import pytest
import time
import uuid
import uvicorn
from fastapi import FastAPI

try:
    import aiohttp
except ImportError:
    pytest.fail("aiohttp package not installed")

from gpustack.websocket_proxy.proxy_server import HTTPSProxyServer
from gpustack.websocket_proxy.message_server import MessageServerHandler, router
from gpustack.websocket_proxy.message_client import MessageClient


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

    # Get a free port if port is 0
    actual_port = get_free_port(host) if port == 0 else port

    config = uvicorn.Config(app, host=host, port=actual_port, log_level="error")
    server = uvicorn.Server(config)

    # Start server in background
    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(0.5)

    return server, server_task, actual_port


class TestProxyWebSocketTunnel:
    """Test proxy functionality over WebSocket tunnel."""

    @pytest.mark.asyncio
    async def test_response_data_integrity(self):
        """Test that the proxy returns exact response data from the server."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Test server returns specific response body
        expected_body = b"Hello, WebSocket Proxy!"
        expected_ctype = "text/plain"

        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {expected_ctype}\r\n"
                    f"Content-Length: {len(expected_body)}\r\n"
                    "\r\n"
                ).encode() + expected_body
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                body = await resp.read()
                assert resp.status == 200
                assert (
                    body == expected_body
                ), f"Body mismatch: {body!r} != {expected_body!r}"
                assert resp.headers.get("Content-Type") == expected_ctype

        # Cleanup
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
    async def test_chunked_transfer_encoding(self):
        """Test that chunked transfer encoding is correctly forwarded through the proxy."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Test server returns chunked response
        # Chunked response: each chunk is "size\r\ndata\r\n", final chunk is "0\r\n\r\n"
        chunks = [b"Hello", b" World", b"!"]
        expected_body = b"".join(chunks)

        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                # Build chunked response
                response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                for chunk in chunks:
                    response += f"{len(chunk):x}\r\n".encode()
                    response += chunk + b"\r\n"
                response += b"0\r\n\r\n"
                writer.write(response)
                await writer.drain()
            except Exception as e:
                print(f"[Test Server] Error: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                body = await resp.read()
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                assert (
                    body == expected_body
                ), f"Body mismatch: {body!r} != {expected_body!r}"
                # aiohttp should automatically decode chunked transfer
                print(f"[Test] Chunked response received: {body!r}")

        # Cleanup
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
    async def test_response_json_data(self):
        """Test that JSON response is correctly forwarded through the proxy."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        import json

        expected_json = {"status": "ok", "data": [1, 2, 3], "message": "proxy works"}
        expected_body = json.dumps(expected_json).encode()

        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(expected_body)}\r\n"
                    "\r\n"
                ).encode() + expected_body
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        url = f"http://127.0.0.1:{test_port}/api/data"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                assert (
                    data == expected_json
                ), f"JSON mismatch: {data} != {expected_json}"

        # Cleanup
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
    async def test_large_response_data(self):
        """Test that large response body is fully returned through the proxy."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # 64KB response body
        response_size = 64 * 1024
        expected_body = b"X" * response_size

        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                response = (
                    "HTTP/1.1 200 OK\r\n" f"Content-Length: {response_size}\r\n" "\r\n"
                ).encode() + expected_body
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                body = await resp.read()
                assert (
                    len(body) == response_size
                ), f"Size mismatch: {len(body)} != {response_size}"
                assert body == expected_body, "Body content mismatch"

        # Cleanup
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
    async def test_response_headers_forwarded(self):
        """Test that response headers are correctly forwarded through the proxy."""
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        expected_body = b"headers-test"
        custom_header_value = "custom-value-123"

        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/custom\r\n"
                    f"X-Custom-Header: {custom_header_value}\r\n"
                    f"Content-Length: {len(expected_body)}\r\n"
                    "\r\n"
                ).encode() + expected_body
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())
        await asyncio.sleep(0.5)

        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                assert resp.status == 200
                assert resp.headers.get("Content-Type") == "text/custom"
                assert resp.headers.get("X-Custom-Header") == custom_header_value
                body = await resp.read()
                assert body == expected_body

        # Cleanup
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
    async def test_single_get_request(self):
        """Test a single GET request through WebSocket proxy tunnel."""
        # Setup WebSocket server
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )
        print(f"[Test] WebSocket server started on port {ws_port}")

        # Setup HTTP proxy server
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test HTTP server
        async def handle_request(reader, writer):
            try:
                await reader.read(8192)
                response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect MessageClient to register the test server IP
        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id="test-client",
            cidrs=["127.0.0.1/32"],  # Register test server IP
        )
        client_task = asyncio.create_task(client.run())

        await asyncio.sleep(0.5)

        # Make request through proxy
        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            start = time.time()
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                body = await resp.text()
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                assert body == "OK", f"Expected 'OK', got {body!r}"
            elapsed = time.time() - start

        # Cleanup
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

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Request took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_single_post_request(self):
        """Test a single POST request through WebSocket proxy tunnel."""
        # Setup WebSocket server
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        _, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Setup HTTP proxy server
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test HTTP server
        request_size = 1024  # 1KB
        response_size = 2048  # 2KB

        async def handle_request(reader, writer):
            try:
                # Read all headers
                content_length = 0
                while True:
                    line = await reader.readline()
                    if not line:
                        return
                    if line == b'\r\n':
                        break
                    if line.lower().startswith(b'content-length:'):
                        content_length = int(line.split(b':')[1].strip())

                # Read body if present
                if content_length > 0:
                    await reader.readexactly(content_length)

                # Send response
                response_body = b'X' * response_size
                response = (
                    "HTTP/1.1 200 OK\r\n" f"Content-Length: {response_size}\r\n" "\r\n"
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

        # Connect MessageClient
        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())

        await asyncio.sleep(0.5)

        # Make POST request through proxy
        url = f"http://127.0.0.1:{test_port}/"
        data = b'A' * request_size
        connector = aiohttp.TCPConnector(limit=1)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            start = time.time()
            async with session.post(
                url, data=data, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                received = await resp.read()
            elapsed = time.time() - start

        # Verify response content
        assert (
            len(received) == response_size
        ), f"Response size mismatch: {len(received)} != {response_size}"
        assert received == b'X' * response_size, "Response body content mismatch"

        # Cleanup
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

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Request took too long: {elapsed:.2f}s"


class TestProxyThroughput:
    """Test proxy throughput with different payload sizes."""

    @pytest.mark.asyncio
    async def test_small_payload_throughput(self):
        """Test throughput with 512B payload."""
        await self._test_throughput(
            request_size=512, response_size=512, num_requests=50, concurrency=5
        )

    @pytest.mark.asyncio
    async def test_medium_payload_throughput(self):
        """Test throughput with 4KB payload."""
        await self._test_throughput(
            request_size=4096, response_size=4096, num_requests=50, concurrency=5
        )

    async def _test_throughput(
        self, request_size, response_size, num_requests, concurrency
    ):
        """Helper method to test throughput."""
        # Setup WebSocket server
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Setup HTTP proxy server
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test HTTP server
        async def handle_request(reader, writer):
            try:
                await reader.read(8192)

                response_body = b'X' * response_size
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Length: {response_size}\r\n"
                    f"\r\n"
                ).encode() + response_body

                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect MessageClient
        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())

        await asyncio.sleep(0.5)

        # Make requests through proxy
        url = f"http://127.0.0.1:{test_port}/"
        data = b'A' * request_size
        connector = aiohttp.TCPConnector(limit=concurrency)

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:
            start = time.time()

            async def make_request():
                req_start = time.time()
                async with session.post(
                    url, data=data, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    received = await resp.read()
                    assert (
                        received == b'X' * response_size
                    ), "Response body content mismatch"
                return time.time() - req_start

            # Run requests in batches
            for batch_start in range(0, num_requests, concurrency):
                batch_size = min(concurrency, num_requests - batch_start)
                tasks = [make_request() for _ in range(batch_size)]
                await asyncio.gather(*tasks)

                done = batch_start + batch_size
                if done % 25 == 0:
                    print(f"[Test] Progress: {done}/{num_requests}")

        elapsed = time.time() - start

        # Cleanup
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

        # Calculate throughput
        total_bytes = num_requests * (request_size + response_size)
        throughput_bps = total_bytes / elapsed
        throughput_mbps = throughput_bps / (1024 * 1024)

        print(
            f"\n[Test] Throughput: {throughput_mbps:.2f} MB/s ({throughput_bps / 1024:.0f} KB/s)"
        )

        # Throughput targets
        assert throughput_mbps >= 0.1, f"Throughput too low: {throughput_mbps:.2f} MB/s"


class TestProxyLatency:
    """Test proxy latency characteristics."""

    @pytest.mark.asyncio
    async def test_latency_distribution(self):
        """Test distribution of request latencies."""
        # Setup WebSocket server
        message_handler = MessageServerHandler(
            listen_address="127.0.0.1",
            listen_port=0,
            proxy_port=0,
        )
        ws_server, ws_task, ws_port = await start_websocket_server(
            message_handler, "127.0.0.1", 0
        )

        # Setup HTTP proxy server
        proxy_server = HTTPSProxyServer(
            host="127.0.0.1",
            port=0,
            connection_manager_getter=message_handler.get_connection_manager_by_ip_in_cidr,
        )
        proxy_task = asyncio.create_task(proxy_server.start())
        await asyncio.sleep(0.5)

        proxy_addr = proxy_server.server.sockets[0].getsockname()
        proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

        # Start test HTTP server
        async def handle_request(reader, writer):
            try:
                await reader.read(8192)

                response = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
                writer.write(response)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        test_server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        test_port = test_server.sockets[0].getsockname()[1]

        # Connect MessageClient
        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())

        await asyncio.sleep(0.5)

        # Measure latencies
        url = f"http://127.0.0.1:{test_port}/"
        connector = aiohttp.TCPConnector(limit=10)
        num_requests = 50

        async with aiohttp.ClientSession(
            proxy=f"http://{proxy_host}:{proxy_port}", connector=connector
        ) as session:

            async def make_request():
                req_start = time.time()
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    await resp.text()
                return time.time() - req_start

            tasks = [make_request() for _ in range(num_requests)]
            latencies = await asyncio.gather(*tasks)

        # Cleanup
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

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies) * 1000
        p50_latency = sorted(latencies)[int(len(latencies) * 0.5)] * 1000
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] * 1000
        max_latency = max(latencies) * 1000

        print(f"\n[Test] Latency Statistics ({num_requests} requests):")
        print(f"[Test] Average: {avg_latency:.2f}ms")
        print(f"[Test] P50: {p50_latency:.2f}ms")
        print(f"[Test] P95: {p95_latency:.2f}ms")
        print(f"[Test] Max: {max_latency:.2f}ms")

        # Performance targets
        assert avg_latency < 500, f"Average latency too high: {avg_latency:.2f}ms"
        assert p95_latency < 1000, f"P95 latency too high: {p95_latency:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
