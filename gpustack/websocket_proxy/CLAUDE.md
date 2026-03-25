# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a WebSocket-based HTTP/HTTPS forward proxy that supports both direct TCP connections and tunneling through WebSocket clients. The system uses a custom binary message protocol for communication between server and client. It also supports **server federation** — multiple servers share client registrations so a proxy request can be forwarded to a peer server whose client can reach the target.

## Common Commands

### Running the Application

```bash
# Start the server (WebSocket server on ws://host:8765/connect, HTTP proxy on host:8000)
python -m gpustack.websocket_proxy.main server --host localhost --port 8765 --proxy-port 8000

# Start a client (connects to server and registers CIDRs/unix sockets)
python -m gpustack.websocket_proxy.main client --host localhost --port 8765 --cidr "10.0.0.0/8" --unix-socket "/var/run/docker.sock"
```

### Running Tests

```bash
# Run all websocket_proxy tests
pytest tests/websocket_proxy/ -v

# Run specific test file
pytest tests/websocket_proxy/test_message.py -v
pytest tests/websocket_proxy/test_authenticator.py -v
pytest tests/websocket_proxy/test_connection.py -v
pytest tests/websocket_proxy/test_server_federation.py -v
pytest tests/websocket_proxy/test_proxy_server.py -v
pytest tests/websocket_proxy/test_websocket_bench.py -v

# Run specific test class
pytest tests/websocket_proxy/test_authenticator.py::TestAuthenticatorClass -v
pytest tests/websocket_proxy/test_server_federation.py::TestServerFederation -v
```

## Architecture

### Message Protocol ([message.py](gpustack/websocket_proxy/message.py))

The core is a binary message protocol using a registry pattern. Format: `[version=1 byte][type=1 byte][payload]`.

Message types:

- `ConnectRequestMessage` — Server asks client to connect to a target (TCP or Unix socket)
- `ConnectResponseMessage` — Client responds with success/failure + error
- `DataMessage` — Bidirectional data flow (supports GZIP compression)
- `DisconnectMessage` — Connection close notification
- `HeartbeatMessage` — Keep-alive ping
- `ListClientsMessage/ListClientsResponseMessage` — Client discovery
- `ClientUpdateMessage` — Server broadcasts client registrations to peers

Server federation uses **header-based registration** (not message-based). Server info is exchanged via WebSocket handshake headers (`x-server-id`). When connecting to a peer, `listen_address`, `listen_port`, and `proxy_port` are obtained from the peer's response headers.

### Connection Flow

1. **Client Registration**: Client connects to `/connect` WebSocket with headers `x-client-id`, `x-cidrs`, `x-unix-sockets`
2. **Proxy Request**: HTTP client hits the proxy server (port 8000) with full URI (HTTP) or `CONNECT host:port` (HTTPS)
3. **Routing**: Proxy server matches target IP against registered CIDRs via Patricia Trie — first checks local clients, then peers via `RemoteConnectionManager`
4. **Tunnel Setup**: Server sends `ConnectRequestMessage`; client opens connection to target and responds with `ConnectResponseMessage`
5. **Data Flow**: For WebSocket tunnels, `HTTPSProxyServer` uses `tunnel()` to relay data between client and server. Server's `TunnelConnection` queues incoming data via `handle_data()`; client reads from queue and forwards to target.

### Server Federation ([message_server.py](gpustack/websocket_proxy/message_server.py))

`MessageServerHandler` manages both client connections and server federation:

- `add_peer()` / `remove_peer()` — Manage outgoing peer connections via REST API
- `handle_server_federation()` — Accepts incoming peer connections and sends server info via response headers
- `broadcast_client_update()` — Sends `ClientUpdateMessage` to all peers when a client connects/disconnects
- `RemoteConnectionManager` — When no local client matches, proxies the HTTP request to a peer's proxy port instead of using WebSocket tunnel
- `authenticator` — Optional HMAC-SHA256 authentication for peer connections. When configured, outgoing connections inject `x-auth-signature` header; incoming connections verify the signature before accepting.
- `callback_on_connect` — Optional async callback invoked when a client or server federation connection is established. Signature: `Callable[[Optional[ServerInfo], Optional[RegisteredClientInfo]], Coroutine[Any, Any, None]]`. Called after `websocket.accept()`, before handling messages. Errors are caught and logged without affecting the connection.

### Key Classes

**[authenticator.py](gpustack/websocket_proxy/authenticator.py)**:

- `Authenticator` — Abstract base with `inject_headers()` (for outgoing connections) and `authenticate()` (for incoming connections)
- `HMACAuthenticator` — HMAC-SHA256 implementation using `x-server-id` and `x-auth-signature` headers
- `NoOpAuthenticator` — Accepts all connections without authentication
- `create_authenticator(key)` — Factory that returns `HMACAuthenticator` if key is provided, otherwise `NoOpAuthenticator`

**[connection_manager.py](gpustack/websocket_proxy/connection_manager.py)**:

- `ConnectionManager` — Server-side; holds WebSocket + dict of `TunnelConnection` by session_id. Sends `ConnectRequestMessage`, waits on `_pending_future`, then dispatches subsequent messages.
- `ClientConnectionManager` — Client-side; handles incoming `ConnectRequestMessage`, opens target connection, responds, then uses `tunnel()` to stream data between target and server.
- `RemoteConnectionManager` — Forwards HTTP requests directly to a peer's proxy port (used when target IP matches a peer's registered client). **TCP-only**: raises `ValueError` for Unix socket targets.
- `BaseConnectionManager` — Abstract base class (Protocol) for connection managers with `connect()` and `disconnect()` methods.

**[connection.py](gpustack/websocket_proxy/connection.py)**:

- `TunnelConnection` — Holds `session_id`, optional `websocket`. States: pending (`_pending_future` set) → connected (`set_connected()`). `handle_data()` queues incoming data into `_response_queue`. `read()` dequeues data (blocking). `write()` wraps `DataMessage` + `_send_to_websocket()`. Used by `ConnectionManager` on server side and by `ClientConnectionManager` on client side.

**[proxy_server.py](gpustack/websocket_proxy/proxy_server.py)**:

- `HTTPSProxyServer` — asyncio `StreamReader/Writer` based. Routes via `_get_target_ip()` + `connection_manager_getter`. Two data paths: direct TCP tunnel (`_tunnel()`) or WebSocket relay (`_relay_http_via_websocket()`). Filters hop-by-hop headers per RFC 7230. `_send_error()` sends proper HTTP error responses with reason phrase in status line and error details in body.
- `HeaderAuthenticator` — `Callable[[Dict[str, str]], Coroutine[Any, Any, bool]]`. Async callable invoked immediately after reading headers, before any routing. Return `True` to allow, `False` to reject with 401.
- `HeaderRouter` — `Callable[[Dict[str, str]], Coroutine[Any, Any, Tuple[Optional[str], int]]]`. Optional async callable for flexible routing. When URI has no hostname (e.g., `GET /`), called with headers to resolve `(host, port)`. Return `(None, 0)` to fall back to URI parsing.
- `IOConnection` — Abstract base class (`ABC`) with abstract methods `read()`, `write()`, `close()`. Implemented by `AsyncIOConnection` (asyncio StreamReader/Writer) and `TunnelConnection` (WebSocket tunnel).
- `AsyncIOConnection` — Wraps asyncio `StreamReader/Writer`. Sends HTTP error responses via `send_error()` and CONNECT established via `write_connect_established()`.
- `tunnel()` — Core tunneling function that bidirectionally relays data between `client_connection` and `remote_connection`. Optional `request` param sends initial data (used for HTTP requests). Optional `response_relay` param customizes how response is forwarded (defaults to `relay()` which streams raw bytes).

**[message_server.py](gpustack/websocket_proxy/message_server.py)**:

- `MessageServerHandler` — FastAPI WebSocket endpoint; dispatches to either `handle_client_connection` (has `x-cidrs` header) or `handle_server_federation` (has `x-server-id` header). Maintains `client_registry` (client_id → RegisteredClientInfo), `_cidr_registry` (Patricia Trie for CIDR lookups), `peers`/`serving_peers` (outgoing/incoming federation connections).
- `_safe_callback()` — Helper that wraps `callback_on_connect` with error handling. Exceptions are logged but never propagate.
- `callback_on_disconnect` — Optional async callback invoked when a client disconnects. Uses generation tracking to filter stale callbacks (client reconnects before disconnect is processed). Signature same as `callback_on_connect`.

**[patricia_trie.py](gpustack/websocket_proxy/patricia_trie.py)**:

- `CIDRRegistry` — Wraps py-radix for O(k) longest-prefix-match lookups (k = address bits). `insert(cidr, client_id)`, `remove_client(client_id)`, `find_best_match(ip)`.

### WebSocket Compatibility

The code works with both FastAPI/Starlette (`send_bytes()`) and the `websockets` library (`send()`). Detection: `if hasattr(ws, 'send_bytes'): ... else: ...`. All send/receive paths need this check.

### MessageClient ([message_client.py](gpustack/websocket_proxy/message_client.py))

Client that connects to server and handles `ConnectRequestMessage`:

- `MessageClient` — Uses `ClientConnectionManager` to handle incoming messages. Registers with server via headers (`x-client-id`, `x-cidrs`, `x-unix-sockets`). Implements automatic reconnection with exponential backoff (1s initial, 60s max) and jitter. `update_cidrs()` triggers reconnect to reregister with new CIDRs.
