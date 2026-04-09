"""
Tests for connection module (TunnelConnection, AsyncIOConnection).
"""

import asyncio
import pytest
import uuid
import uvicorn
from fastapi import FastAPI
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

    actual_port = get_free_port(host) if port == 0 else port

    config = uvicorn.Config(app, host=host, port=actual_port, log_level="error")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.5)

    return server, server_task, actual_port


class TestConnectTunnel:
    """Test CONNECT tunnel functionality.

    These tests verify that POST request bodies can be sent through a CONNECT tunnel.
    In a proper implementation, the proxy should establish a direct TCP connection
    to the target and relay data bidirectionally.
    """

    @pytest.mark.asyncio
    async def test_connect_tunnel_with_post_body(self):  # noqa C901
        """Test POST request body through CONNECT tunnel.

        This test sends a raw TCP connection through the proxy using CONNECT,
        then sends a POST request through that tunnel. The target server should
        receive the POST body and respond.

        Note: This test may fail with the current implementation because
        WebSocket tunnel mode is designed for HTTP proxy, not raw TCP tunneling.
        """
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

        # Track what body we received at the target server
        received_body = None

        # Start test HTTP server
        async def handle_request(reader, writer):
            nonlocal received_body
            try:
                # Read headers first
                header_data = b""
                while b"\r\n\r\n" not in header_data:
                    chunk = await reader.read(8192)
                    if not chunk:
                        break
                    header_data += chunk

                header_end = header_data.find(b"\r\n\r\n")
                if header_end < 0:
                    return

                headers = header_data[:header_end].decode("utf-8", errors="ignore")
                body_received = header_data[header_end + 4 :]

                # Parse Content-Length
                content_length = 0
                for line in headers.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())
                        break

                # Continue reading body if needed
                while len(body_received) < content_length:
                    chunk = await reader.read(8192)
                    if not chunk:
                        break
                    body_received += chunk

                received_body = body_received
                print(f"[Test Server] Received body: {received_body!r}")

                # Send response
                response_body = b'POST_RECEIVED'
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

        # Connect MessageClient
        client = MessageClient(
            server_endpoint=f"ws://127.0.0.1:{ws_port}",
            client_id=uuid.uuid4(),
            cidrs=["127.0.0.1/32"],
        )
        client_task = asyncio.create_task(client.run())

        await asyncio.sleep(0.5)

        # Make a raw TCP connection through the proxy using CONNECT
        reader, writer = await asyncio.open_connection(proxy_host, proxy_port)

        try:
            # Send CONNECT request
            connect_request = f"CONNECT 127.0.0.1:{test_port} HTTP/1.1\r\nHost: 127.0.0.1:{test_port}\r\n\r\n"
            writer.write(connect_request.encode())
            await writer.drain()

            # Read CONNECT response with timeout
            connect_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=5.0
            )
            assert connect_response.startswith(
                b"HTTP/1.1 200"
            ), f"CONNECT failed: {connect_response}"

            # Now send a POST request through the tunnel
            post_body = b"This is the POST body sent through CONNECT tunnel"
            post_request = (
                f"POST / HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{test_port}\r\n"
                f"Content-Length: {len(post_body)}\r\n"
                f"Connection: keep-alive\r\n"
                f"\r\n"
            ).encode() + post_body

            writer.write(post_request)
            await writer.drain()

            # Read response with timeout
            response = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            assert b"200 OK" in response, f"Expected 200 OK in response: {response}"
            assert (
                received_body == post_body
            ), f"Body mismatch: {received_body!r} != {post_body!r}"

        finally:
            writer.close()
            await writer.wait_closed()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
