#!/usr/bin/env python3
"""
Message Protocol Example - Connect-based Proxy Demo

This example demonstrates the connect-based proxy flow:
1. Client connects to Server and registers
2. Server sends CONNECT_REQUEST to Client (request to connect to target)
3. Client connects to target, sends CONNECT_RESPONSE (connect succeed)
4. Data flows through the tunnel via persistent socket connections

Server Federation:
- Multiple servers can form a federation to share client registrations
- Each server maintains a registry of clients from all peers
- When a client connects/disconnects, all peers are notified

Run:
    - Start server: python src/main.py server
    - Start client: python src/main.py client

Client API:
    - GET /clients - List all connected clients
      Returns: {"clients": [...], "total": N}
    - GET /clients/{client_id} - Get details for a specific client
      Returns: client info including CIDRs, unix sockets, active sessions
    - GET /clients/{client_id}/connections - Get active tunnel connections for a client
      Returns: {"connections": [...], "total": N}

Federation API:
    - POST /register-peer - Register a peer server
      Body: {"address": "host:port", "port": 8765, "proxy_port": 8000}
      Returns: {"status": "ok", "server_id": "uuid"}
    - DELETE /register-peer/{server_id} - Remove a peer server
      Returns: {"status": "ok"}
    - GET /peers - List all connected peers (outgoing and incoming)
      Returns: {"peers": [...]}

WebSocket Endpoints:
    - ws://host:port/connect - Client and peer server connections
"""

import asyncio
import argparse
import logging
import uuid
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .proxy_server import HTTPSProxyServer
from .message_server import MessageServerHandler, router
from .message_client import MessageClient

logger = logging.getLogger(__name__)


# Pydantic models for API requests
class RegisterPeerRequest(BaseModel):
    address: str
    port: int = 8765


class RegisterPeerResponse(BaseModel):
    status: str
    server_id: Optional[uuid.UUID] = None


class RemovePeerResponse(BaseModel):
    status: str


# Global message handler reference for API endpoints
_message_handler: Optional[MessageServerHandler] = None


def _create_server_app(message_handler: MessageServerHandler):  # noqa: C901
    """Create FastAPI app with all routes for the server."""
    app = FastAPI()
    app.state.message_server_handler = message_handler

    # WebSocket endpoint
    app.include_router(router)

    # Federation API endpoints
    @app.post("/register-peer", response_model=RegisterPeerResponse)
    async def register_peer(request: RegisterPeerRequest):
        """Register a peer server"""
        peer_id = await message_handler.add_peer(request.address, request.port)
        if peer_id:
            return RegisterPeerResponse(status="ok", server_id=peer_id)
        return RegisterPeerResponse(status="error", server_id=None)

    @app.delete("/register-peer/{peer_id}", response_model=RemovePeerResponse)
    async def remove_peer(peer_id: str):
        """Remove a peer server by UUID"""
        try:
            peer_uuid = uuid.UUID(peer_id)
            await message_handler.remove_peer(peer_uuid)
        except ValueError:
            # Try to remove by address instead
            await message_handler.remove_peer_by_address(peer_id)
        return RemovePeerResponse(status="ok")

    @app.get("/clients")
    async def list_clients():
        """List all connected clients."""
        clients = []
        for client_id, info in message_handler.client_registry.items():
            conn_mgr = message_handler.connection_managers.get(client_id)
            active_sessions = list(conn_mgr.connections.keys()) if conn_mgr else []
            clients.append(
                {
                    "client_id": str(client_id),
                    "cidrs": info.cidrs,
                    "unix_sockets": info.unix_sockets,
                    "active_sessions": [str(s) for s in active_sessions],
                    "session_count": len(active_sessions),
                }
            )
        return {"clients": clients, "total": len(clients)}

    @app.get("/clients/{client_id}")
    async def get_client(client_id: str):
        """Get details for a specific client."""
        try:
            client_uuid = uuid.UUID(client_id)
        except ValueError:
            raise HTTPException(404, detail=f"Invalid client ID format: {client_id}")
        info = message_handler.client_registry.get(client_uuid)
        if not info:
            raise HTTPException(404, detail=f"Client not found: {client_id}")
        conn_mgr = message_handler.connection_managers.get(client_uuid)
        active_sessions = list(conn_mgr.connections.keys()) if conn_mgr else []
        return {
            "client_id": client_id,
            "cidrs": info.cidrs,
            "unix_sockets": info.unix_sockets,
            "server_id": str(info.server_id) if info.server_id else None,
            "active_sessions": [str(s) for s in active_sessions],
            "session_count": len(active_sessions),
        }

    @app.get("/clients/{client_id}/connections")
    async def get_client_connections(client_id: str):
        """Get active tunnel connections for a specific client."""
        try:
            client_uuid = uuid.UUID(client_id)
        except ValueError:
            raise HTTPException(404, detail=f"Invalid client ID format: {client_id}")
        info = message_handler.client_registry.get(client_uuid)
        if not info:
            raise HTTPException(404, detail=f"Client not found: {client_id}")
        conn_mgr = message_handler.connection_managers.get(client_uuid)
        if not conn_mgr:
            return {"connections": [], "total": 0}
        connections = []
        for session_id, conn in conn_mgr.connections().items():
            connections.append(
                {
                    "session_id": str(session_id),
                    "is_pending": conn.is_pending,
                    "is_connected": conn.is_connected,
                }
            )
        return {"connections": connections, "total": len(connections)}

    @app.get("/peers")
    async def list_peers():
        """List all connected peers (outgoing and incoming)"""
        all_peers = []
        # Outgoing peers (we connected to)
        for peer in message_handler.peers.values():
            all_peers.append(
                {
                    "server_id": str(peer.server_id),
                    "address": (
                        f"{peer.listen_address}:{peer.listen_port}"
                        if peer.listen_address
                        else ""
                    ),
                    "proxy_port": peer.proxy_port,
                    "connected": peer.connected,
                    "type": "outgoing",
                }
            )
        # Incoming serving_peers (connected to us)
        for peer in message_handler.serving_peers.values():
            all_peers.append(
                {
                    "server_id": str(peer.server_id),
                    "address": (
                        f"{peer.listen_address}:{peer.listen_port}"
                        if peer.listen_address
                        else ""
                    ),
                    "proxy_port": peer.proxy_port,
                    "connected": peer.connected,
                    "type": "incoming",
                }
            )
        return {"peers": all_peers}

    return app


async def _run_server(args):
    """Run the server role."""
    global _message_handler

    server_id = uuid.UUID(args.server_id) if args.server_id else uuid.uuid4()

    message_handler = MessageServerHandler(
        server_id=server_id,
        listen_address=args.host,
        listen_port=args.port,
        proxy_port=args.proxy_port,
    )
    _message_handler = message_handler

    proxy = HTTPSProxyServer(
        host=args.host,
        port=args.proxy_port,
        connection_manager_getter=message_handler.get_connection_manager,
    )

    logger.debug(f"[Server] Starting WebSocket server on {args.host}:{args.port}")
    logger.debug(f"[Server] WebSocket endpoint: ws://{args.host}:{args.port}/connect")
    logger.debug(f"[Server] HTTP proxy endpoint: http://{args.host}:{args.proxy_port}/")
    logger.debug(
        f"[Server] Federation API: http://{args.host}:{args.port}/register-peer"
    )

    app = _create_server_app(message_handler)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    uvicorn_server = uvicorn.Server(config)

    # Start proxy server in background
    proxy_task = asyncio.create_task(proxy.start())

    # Run uvicorn (it handles signals internally)
    await uvicorn_server.serve()

    # Uvicorn exited, stop proxy
    proxy_task.cancel()
    try:
        await proxy_task
    except asyncio.CancelledError:
        pass
    await proxy.stop()


async def _run_client(args):
    """Run the client role."""
    client = MessageClient(
        server_endpoint=f"http://{args.host}:{args.port}",
        client_id=args.client_id,
        cidrs=args.cidr,
        unix_sockets=args.unix_sockets,
    )
    await client.run()


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WebSocket Message Protocol Example')
    parser.add_argument(
        'role', choices=['server', 'client'], help='Run as server or client'
    )
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    parser.add_argument(
        '--proxy-port', type=int, default=8000, help='HTTP proxy port (server only)'
    )

    # Server-specific options
    parser.add_argument(
        '--server-id', default=None, help='Server ID (auto-generated if not provided)'
    )

    # Client-specific options
    parser.add_argument(
        '--client-id', default=None, help='Client ID (auto-generated if not provided)'
    )
    parser.add_argument(
        '--cidr',
        action='append',
        default=[],
        help='CIDR to register (can be specified multiple times)',
    )
    parser.add_argument(
        '--unix-socket',
        action='append',
        default=[],
        dest='unix_sockets',
        help='Unix socket path to register (can be specified multiple times)',
    )
    return parser.parse_args()


async def main():
    args = _parse_args()

    if args.role == 'server':
        await _run_server(args)
    else:
        await _run_client(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
