#!/usr/bin/env python3
"""
Message Server - Handles WebSocket connections and message protocol
"""

import uuid
import asyncio
import contextlib
import ipaddress
import logging
import time
from typing import Optional, Dict, List, Tuple, Callable, Coroutine, Any, TypeAlias
from contextlib import asynccontextmanager

from .patricia_trie import CIDRRegistry
from fastapi import WebSocket, Depends, APIRouter
from websockets.asyncio.client import connect as ws_connect
from .connection_manager import ConnectionManager, RemoteConnectionManager
from .message import (
    SessionBaseMessage,
    HeartbeatMessage,
    ClientUpdateMessage,
    ClientUpdateInfo,
    RegisteredClientInfo,
    ServerInfo,
    ServerPeer,
    parse_message,
    pack_message,
)
from .authenticator import Authenticator, NoOpAuthenticator
from .constants import default_connect_path

logger = logging.getLogger(__name__)


VERSION = "0.1.0"


# ==================== Type Aliases ====================

ServerInfoGetter: TypeAlias = Callable[[WebSocket], Optional['ServerInfo']]
RegisteredClientInfoGetter: TypeAlias = Callable[
    [WebSocket], Optional['RegisteredClientInfo']
]
WSCallback: TypeAlias = Callable[
    [Optional['ServerInfo'], Optional['RegisteredClientInfo']],
    Coroutine[Any, Any, None],
]


# ==================== Helper Functions ====================


def default_server_info_getter(websocket: WebSocket) -> Optional['ServerInfo']:
    """Extract server info from headers if this is a server connection"""
    return ServerInfo.from_headers(websocket.headers)


def default_client_info_getter(
    websocket: WebSocket,
) -> Optional['RegisteredClientInfo']:
    """Extract client info from headers if this is a client connection"""
    return RegisteredClientInfo.from_headers(websocket.headers)


# ==================== Registration Claim Validation ====================

# A worker registers the CIDRs and unix-socket paths it serves so the proxy
# can route traffic to it. Neither claim may be trusted verbatim: a client
# claiming the whole address space (0.0.0.0/0, ::/0) or another worker's
# socket path would receive traffic meant for someone else. Workers
# register their own address (/32, /128) or at most the subnet they serve,
# so anything shorter than a /8 is not a legitimate claim.
MIN_CIDR_PREFIXLEN = 8

# Runtime paths a client may proxy unix sockets from. Kept to well-known
# locations so one client cannot claim another's socket by path.
UNIX_SOCKET_PREFIXES = ("/var/run/", "/run/", "/tmp/")

# A client that has sent at least one heartbeat is expected to keep the
# signal flowing; a websocket silent for this long is a half-open tunnel and
# is reaped. Clients on older versions never heartbeat and are never reaped.
DEFAULT_CLIENT_READ_TIMEOUT = 45.0


def _valid_cidr_claim(cidr: str) -> bool:
    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return False
    return network.prefixlen >= MIN_CIDR_PREFIXLEN


def _valid_socket_claim(path: str) -> bool:
    return path.startswith(UNIX_SOCKET_PREFIXES)


def _authorized_host_route(cidr: str, authorized_addresses: List[str]) -> bool:
    """Whether ``cidr`` is the host route of one of the client's authorized addresses.

    Workers only ever register their own address as a /32 (or /128).
    Restricting authorized claims to exactly that keeps one worker from
    registering a subnet that overlaps another worker's routes — under
    longest-prefix match the more specific claim would then receive the
    overlapped traffic. Subnet claims stay legal in the generic protocol
    module (no authorized addresses); only the production wiring passes
    them in.
    """
    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return False
    if network.prefixlen != network.max_prefixlen:
        return False
    return str(network.network_address) in authorized_addresses


def _invalid_client_claims(client_info: 'RegisteredClientInfo') -> List[str]:
    """Return the claims in ``client_info`` a client is not allowed to make."""
    invalid = [c for c in client_info.cidrs if not _valid_cidr_claim(c)]
    invalid += [s for s in client_info.unix_sockets if not _valid_socket_claim(s)]
    return invalid


# ==================== Dataclasses ====================


class MessageServerHandler:
    """Simple server that handles message protocol and connect-based proxy"""

    _server_info: ServerInfo

    def __init__(
        self,
        listen_address: str,
        listen_port: int,
        proxy_port: int,
        server_id: Optional[uuid.UUID] = None,
        client_info_getter: RegisteredClientInfoGetter = default_client_info_getter,
        server_info_getter: ServerInfoGetter = default_server_info_getter,
        authenticator: Authenticator = None,
        callback_on_connect: Optional[WSCallback] = None,
        callback_on_disconnect: Optional[WSCallback] = None,
        client_read_timeout: float = DEFAULT_CLIENT_READ_TIMEOUT,
    ):
        self._server_info = ServerInfo(
            server_id=server_id or uuid.uuid4(),
            listen_address=listen_address,
            listen_port=listen_port,
            proxy_port=proxy_port,
        )
        self._callback_on_connect = callback_on_connect
        self._callback_on_disconnect = callback_on_disconnect

        # Info extractors - can be overridden for customization
        self._client_info_getter = client_info_getter
        self._server_info_getter = server_info_getter
        self._authenticator = authenticator or NoOpAuthenticator()

        # Client management
        self.client_registry: Dict[uuid.UUID, RegisteredClientInfo] = (
            {}
        )  # client_id -> client_info
        self.connection_managers: Dict[uuid.UUID, ConnectionManager] = {}
        self._cidr_registry = CIDRRegistry()

        # Generation tracking for disconnect callbacks
        self._client_generations: Dict[uuid.UUID, int] = {}  # client_id -> generation
        self._generation_lock = asyncio.Lock()
        self._client_read_timeout = client_read_timeout

        # Serializes a client's registration/release together with the peer
        # broadcast that announces it, so a reconnecting websocket cannot
        # interleave its "add" with the previous websocket's "remove" in
        # flight and leave peers with a stale (missing) route. Entries are
        # reference-counted and dropped once the last user leaves, so worker
        # churn does not accumulate locks forever.
        self._client_update_locks: Dict[uuid.UUID, list] = {}
        self._update_locks_guard = asyncio.Lock()

        # Serializes CIDR ownership checks against claim writes across ALL
        # clients. The per-client sections cannot do this: two different
        # clients claiming the same CIDR race through their own locks, and
        # the later write would silently steal the other's route.
        self._cidr_claims_lock = asyncio.Lock()

        # Per-client chain of queued peer-update sends. Sends never run under
        # the per-client section lock (a wedged peer's full write buffer
        # would then stall registration/release); chaining keeps each
        # client's updates ordered as produced.
        self._client_broadcast_chains: Dict[uuid.UUID, asyncio.Task] = {}

        # Server federation
        self.peers: Dict[uuid.UUID, ServerPeer] = {}  # Outgoing: servers I connected to
        self.serving_peers: Dict[uuid.UUID, ServerPeer] = (
            {}
        )  # Incoming: servers that connected to me
        self.peer_tasks: Dict[uuid.UUID, asyncio.Task] = {}  # server_id -> task

    @asynccontextmanager
    async def _client_update_section(self, client_id: uuid.UUID):
        """Run a critical section serialized against every other
        registration/release/peer-update for this client.

        The keyed lock is dropped once the last user (including waiters)
        leaves, so a long-lived server does not accumulate one lock per
        client id it has ever served.
        """
        async with self._update_locks_guard:
            entry = self._client_update_locks.get(client_id)
            if entry is None:
                entry = [asyncio.Lock(), 0]
                self._client_update_locks[client_id] = entry
            entry[1] += 1
            lock = entry[0]
        try:
            async with lock:
                yield
        finally:
            async with self._update_locks_guard:
                entry[1] -= 1
                if entry[1] == 0 and self._client_update_locks.get(client_id) is entry:
                    del self._client_update_locks[client_id]

    async def _get_next_generation(self, client_id: uuid.UUID) -> int:
        """Get the next generation number for a client (thread-safe)."""
        async with self._generation_lock:
            gen = self._client_generations.get(client_id, 0) + 1
            self._client_generations[client_id] = gen
            return gen

    async def _safe_callback(
        self,
        callback: WSCallback,
        server_info: Optional[ServerInfo],
        client_info: Optional[RegisteredClientInfo],
    ) -> None:
        """Execute callback with error handling, does not raise exceptions."""
        try:
            await callback(server_info, client_info)
        except Exception as e:
            logger.error(f"[Server] callback_on_connect error: {e}", exc_info=True)

    async def _safe_disconnect_callback(
        self,
        callback: WSCallback,
        client_info: Optional[RegisteredClientInfo],
        generation: int,
    ) -> None:
        """Execute disconnect callback with error handling and stale callback filtering."""
        if client_info is None:
            return
        async with self._generation_lock:
            current_gen = self._client_generations.get(client_info.client_id, 0)
            if generation < current_gen:
                logger.debug(
                    f"[Server] Stale disconnect callback ignored: "
                    f"client={client_info.client_id}, callback_gen={generation}, "
                    f"current_gen={current_gen}"
                )
                return
        try:
            await callback(None, client_info)
        except Exception as e:
            logger.error(f"[Server] callback_on_disconnect error: {e}", exc_info=True)

    def _find_peer_by_server_id(self, server_id: uuid.UUID) -> Optional[ServerPeer]:
        """Find a peer by server_id, checking both outgoing and incoming peers"""
        peer = self.peers.get(server_id)
        if peer:
            return peer
        return self.serving_peers.get(server_id)

    def get_connection_manager_by_ip_in_cidr(
        self, target_ip: str
    ) -> Optional[ConnectionManager]:
        """Find a ConnectionManager by matching IP against registered CIDRs using Patricia Trie.

        Returns local ConnectionManager for local clients, or RemoteConnectionManager
        for peer clients.
        """
        client_id = self._cidr_registry.find_best_match(target_ip)
        if not client_id:
            return None

        client_info = self.client_registry.get(client_id)
        if not client_info:
            return None

        # Check if this is a local client (registered on this server)
        if client_info.server_id == self._server_info.server_id:
            return self.connection_managers.get(client_id)

        # Peer client - find the peer and return RemoteConnectionManager
        peer = self._find_peer_by_server_id(client_info.server_id)
        if peer and peer.listen_address and peer.proxy_port:
            return RemoteConnectionManager(peer.listen_address, peer.proxy_port)

        return None

    def get_connection_manager(self, target_ip: str) -> Optional[ConnectionManager]:
        """Get connection manager for target IP (local or remote)."""
        return self.get_connection_manager_by_ip_in_cidr(target_ip)

    async def add_peer(self, address: str, port: int) -> Optional[uuid.UUID]:
        """Add a peer server and connect to it. Returns the peer_id when connected."""
        # Create a future that will be resolved when the peer connects
        future = asyncio.Future()

        asyncio.create_task(self.connect_to_peer(address, port, future))

        # Wait for the peer to connect (with timeout)
        try:
            peer_id = await asyncio.wait_for(future, timeout=10.0)
            return peer_id
        except asyncio.TimeoutError:
            logger.debug(
                f"[Server] Timeout waiting for peer to connect: {address}:{port}"
            )
            return None

    async def remove_peer(self, peer_id: uuid.UUID):
        """Remove a peer server by UUID"""
        logger.debug(f"[Server] Attempting to remove peer by UUID: {peer_id}")
        return await self._remove_peer_impl(peer_id)

    async def remove_peer_by_address(self, address: str):
        """Remove a peer server by address (host:port)"""
        logger.debug(f"[Server] Attempting to remove peer by address: {address}")
        target_peer_id = self._get_peer_id_by_address(address)
        if not target_peer_id:
            return False
        return await self._remove_peer_impl(target_peer_id)

    def _get_peer_id_by_address(self, address: str) -> Optional[uuid.UUID]:
        """Helper to find peer_id by address (host:port)"""
        for peer_id, peer in self.peers.items():
            peer_addr = f"{peer.listen_address}:{peer.listen_port}"
            if peer_addr == address:
                return peer_id
        for peer_id, peer in self.serving_peers.items():
            peer_addr = f"{peer.listen_address}:{peer.listen_port}"
            if peer_addr == address:
                return peer_id
        logger.debug(f"[Server] No peer found with address: {address}")
        return None

    async def _remove_peer_impl(self, peer_id: uuid.UUID) -> bool:
        """Internal implementation for removing a peer (checks both peers and serving_peers)"""
        # Check outgoing peers first
        peer = self.peers.pop(peer_id, None)
        if not peer:
            # Check incoming serving_peers
            peer = self.serving_peers.pop(peer_id, None)

        if peer:
            logger.debug(f"[Server] Found peer to remove: {peer.server_id}")
            if peer.websocket:
                await peer.websocket.close()
                logger.debug(f"[Server] Closed websocket for peer: {peer_id}")
        else:
            logger.debug(f"[Server] Peer not found: {peer_id}")
            return False

        task = self.peer_tasks.pop(peer_id, None)
        if task:
            task.cancel()
            logger.debug(f"[Server] Cancelled task for peer: {peer_id}")

        logger.debug(f"[Server] Removed peer: {peer_id}")
        return True

    async def connect_to_peer(self, host: str, port: int, future: asyncio.Future):
        """Connect to a peer server"""
        peer_key = f"{host}:{port}"
        try:
            ws_uri = f"ws://{host}:{port}{default_connect_path}"
            logger.debug(f"[Server] Connecting to peer: {ws_uri}")

            # Connect with server info in headers (header-based registration)
            headers = self._server_info.to_headers()
            self._authenticator.inject_headers(headers)
            websocket = await ws_connect(ws_uri, additional_headers=headers)

            # websockets handshake responses can carry duplicate `Date` headers;
            # `dict(Headers)` uses the mapping protocol and raises
            # MultipleValuesError on any repeated key. ServerInfo.from_headers
            # accepts any object with a `.get(key)` method, so pass the raw
            # Headers through.
            peer_info = ServerInfo.from_headers(websocket.response.headers)
            if not peer_info:
                logger.debug("[Server] Peer did not provide valid registration headers")
                await websocket.close()
                if not future.done():
                    future.set_result(None)
                return

            peer_server_id = peer_info.server_id
            peer = ServerPeer(
                server_id=peer_info.server_id,
                listen_address=peer_info.listen_address,
                listen_port=peer_info.listen_port,
                proxy_port=peer_info.proxy_port,
                websocket=websocket,
                connected=True,
            )
            self.peers[peer_server_id] = peer
            logger.debug(f"[Server] Registered with peer: {peer_server_id}")

            # Resolve the future to notify add_peer that connection is complete
            if not future.done():
                future.set_result(peer_server_id)

            # Start handling messages from peer
            task = asyncio.create_task(self.handle_peer(websocket, peer_server_id))
            self.peer_tasks[peer_server_id] = task

        except Exception as e:
            # Connection failed - peer may not be running or rejected connection
            logger.debug(f"[Server] Failed to connect to peer {peer_key}: {e}")
            if not future.done():
                future.set_exception(e)

    async def handle_peer(self, websocket, peer_server_id: uuid.UUID):
        """Handle messages from a peer server"""
        try:
            # Check if this is a Starlette WebSocket or websockets WebSocket
            if hasattr(websocket, 'receive'):
                # Starlette WebSocket
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    raw_data = message.get("bytes") or message.get("text", "").encode()
                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received from peer: {msg.get_type()}")

                    if isinstance(msg, ClientUpdateMessage):
                        await self.handle_peer_client_update(msg)
            else:
                # websockets library WebSocket
                while True:
                    raw_data = await websocket.recv()
                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received from peer: {msg.get_type()}")

                    if isinstance(msg, ClientUpdateMessage):
                        await self.handle_peer_client_update(msg)

        except Exception as e:
            logger.debug(f"[Server] Peer connection error: {e}")
        finally:
            # Clean up peer on disconnect (check both peers and serving_peers)
            if peer_server_id in self.peers:
                del self.peers[peer_server_id]
            if peer_server_id in self.serving_peers:
                del self.serving_peers[peer_server_id]
            if peer_server_id in self.peer_tasks:
                del self.peer_tasks[peer_server_id]

            # Clean up clients registered through this peer
            await self._remove_clients_from_peer(peer_server_id)

            logger.debug(f"[Server] Peer disconnected: {peer_server_id}")

    def _cidr_claim_owners(
        self, client_id: uuid.UUID, cidrs: List[str]
    ) -> List[Tuple[str, uuid.UUID]]:
        """Return ``(cidr, owner)`` pairs for claims owned by another client.

        ``CIDRRegistry.insert`` overwrites an existing mapping, so accepting
        such a claim would silently redirect the owning client's tunneled
        traffic to the caller.
        """
        owned = []
        for cidr in cidrs:
            owner = self._cidr_registry.find_exact(cidr)
            if owner is not None and owner != client_id:
                owned.append((cidr, owner))
        return owned

    async def release_client(
        self, client_id: uuid.UUID, only_if_websocket: Optional[WebSocket] = None
    ) -> bool:
        """Drop a client's registration and route claims without waiting for its socket.

        ``handle_client`` releases these in its ``finally`` block, which only
        runs once the websocket reports a disconnect. A tunnel that is
        half-open — NAT keepalives never reach the server, or the socket was
        never detected as closed — keeps its claims reserved indefinitely, and
        the claim-owner check then rejects the next worker that claims the
        same CIDR. Callers use this when the client is known to be gone for
        another reason (its worker row was deleted) so the claims are freed
        immediately instead of after an unbounded wait.

        When ``only_if_websocket`` is given, the release is skipped (under the
        same lock) unless the registration still belongs to that websocket, so
        a caller's earlier ownership check cannot be invalidated by a
        reconnect registering in between.

        Returns whether anything was registered for ``client_id``.
        """
        async with self._client_update_section(client_id):
            connection_manager = self.connection_managers.get(client_id)
            if only_if_websocket is not None and (
                connection_manager is None
                or connection_manager.websocket is not only_if_websocket
            ):
                return False
            client_info = self.client_registry.pop(client_id, None)
            connection_manager = self.connection_managers.pop(client_id, None)
            if client_info is None and connection_manager is None:
                return False

            self._cidr_registry.remove_client(client_id)
            cidr_list = client_info.cidrs if client_info else []
            socket_list = client_info.unix_sockets if client_info else []
            # Enqueue before the next await: a task cancelled at the
            # generation bump must not leave peers holding a stale mirrored
            # entry for an already-released client.
            self._enqueue_client_update("remove", client_id, cidr_list, socket_list)
            logger.debug(f"[Server] Released client: {client_id}")

            # Advance the generation so the socket's own disconnect path treats
            # its callback as stale: the client is gone on purpose, and writing
            # its state back to the database would resurrect a deleted worker.
            await self._get_next_generation(client_id)

            websocket_to_close = (
                connection_manager.websocket if connection_manager is not None else None
            )

        # Close outside the section: a hung close handshake on a half-open
        # socket must not stall the registrations, releases, and peer updates
        # serialized on this client.
        if websocket_to_close is not None:
            try:
                await websocket_to_close.close()
            except Exception as e:
                logger.debug(f"[Server] Error closing released client {client_id}: {e}")
        return True

    async def force_release(self, client_id: uuid.UUID) -> bool:
        """Release a client's tunnel on whatever server replica holds it.

        ``release_client`` only frees state this process knows about. A
        delete request may land on a replica that is not the tunnel's owner,
        in which case the local release (at most a mirrored registry entry)
        must be followed by a ``force_remove`` update so the owning server
        releases and closes the socket; an ordinary ``remove`` would leave
        the owner's live websocket and generation untouched.

        ``force_remove`` requires every server replica to run this version:
        an older peer decodes it as ``remove`` and only drops its mirror,
        leaving the owning replica's live tunnel untouched.
        """
        owns_tunnel = client_id in self.connection_managers
        released = await self.release_client(client_id)
        if not owns_tunnel:
            self._enqueue_client_update("force_remove", client_id, [], [])
        return released

    async def handle_peer_client_update(self, msg: ClientUpdateMessage):
        """Handle client update from peer"""
        for update in msg.updates:
            if update.action == "add":
                client_info = RegisteredClientInfo(
                    client_id=update.client_id,
                    cidrs=update.cidrs,
                    unix_sockets=update.unix_sockets,
                    server_id=msg.server_id,
                )
                invalid = _invalid_client_claims(client_info)
                if invalid:
                    logger.warning(
                        f"[Server] Dropping client update from peer {msg.server_id}"
                        f" for client {update.client_id} with invalid claims: {invalid}"
                    )
                    continue
                foreign: List[Tuple[str, uuid.UUID]] = []
                async with self._client_update_section(update.client_id):
                    # Check and write in one await-free, registry-wide
                    # critical section, so a concurrent registration for a
                    # different client cannot slip a claim in between.
                    async with self._cidr_claims_lock:
                        foreign = self._cidr_claim_owners(
                            update.client_id, update.cidrs
                        )
                        if not foreign:
                            self.client_registry[update.client_id] = client_info
                            # The update's claim set is authoritative: a
                            # re-registered client whose address changed must
                            # not keep its old CIDRs.
                            self._cidr_registry.update_client(
                                update.client_id, update.cidrs
                            )
                if foreign:
                    logger.warning(
                        f"[Server] Dropping client update from peer {msg.server_id}"
                        f" for client {update.client_id} claiming CIDRs owned by"
                        " another client: "
                        + ", ".join(
                            f"{cidr} (owner {owner})" for cidr, owner in foreign
                        )
                    )
                    continue
                logger.debug(f"[Server] Added client from peer: {update.client_id}")
            elif update.action == "force_remove":
                # One path covers both cases, entirely under the per-client
                # lock: on the owning replica this closes the socket, drops
                # the route, and advances the generation; on a mirror-only
                # replica it drops the mirrored registry entry.
                await self.release_client(update.client_id)
            elif update.action == "remove":
                # Remove client from local registry
                if update.client_id in self.client_registry:
                    self._cidr_registry.remove_client(update.client_id)
                    del self.client_registry[update.client_id]
                    logger.debug(
                        f"[Server] Removed client from peer: {update.client_id}"
                    )

    async def _remove_clients_from_peer(self, peer_server_id: uuid.UUID):
        """Remove all clients that were registered through a peer server."""
        clients_to_remove = [
            client_id
            for client_id, client_info in self.client_registry.items()
            if client_info.server_id == peer_server_id
        ]

        for client_id in clients_to_remove:
            self._cidr_registry.remove_client(client_id)
            del self.client_registry[client_id]
            logger.debug(
                f"[Server] Removed client {client_id} from disconnected peer {peer_server_id}"
            )

    def send_client_update_to_peer(self, websocket, action: str) -> None:
        """Announce this server's clients to peers (used on peer connect).

        Each client's update is enqueued on its per-client chain instead of
        being packed into one snapshot send: a release racing the
        announcement could otherwise deliver a stale snapshot ``add`` after
        its queued ``remove`` and resurrect an already-released route on the
        new peer. The chains broadcast to every connected peer, which
        includes the one this announcement is for.
        """
        for client_id, client_info in list(self.client_registry.items()):
            # Only send clients owned by this server
            if client_info.server_id == self._server_info.server_id:
                self._enqueue_client_update(
                    action,
                    client_id,
                    client_info.cidrs,
                    client_info.unix_sockets,
                )

    def _enqueue_client_update(
        self,
        action: str,
        client_id: uuid.UUID,
        cidrs: List[str],
        unix_sockets: List[str],
    ) -> None:
        """Queue a client update to all peers without awaiting the sends.

        Registration and release paths call this while holding the client's
        per-client section. Awaiting the sends there would let one wedged
        peer websocket (a full write buffer) stall every other operation
        serialized on that client. The sends run outside the section,
        chained per client so peers observe a client's updates in the order
        the server produced them.
        """
        update = ClientUpdateInfo(
            client_id=client_id,
            action=action,
            cidrs=cidrs,
            unix_sockets=unix_sockets,
        )
        msg_data = pack_message(
            ClientUpdateMessage(server_id=self._server_info.server_id, updates=[update])
        )
        prior = self._client_broadcast_chains.get(client_id)

        async def _send_in_order() -> None:
            if prior is not None and not prior.done():
                with contextlib.suppress(Exception):
                    await prior
            await self._send_client_update_to_peers(msg_data)

        task = asyncio.create_task(_send_in_order())
        self._client_broadcast_chains[client_id] = task

        def _forget(done: asyncio.Task) -> None:
            if self._client_broadcast_chains.get(client_id) is done:
                del self._client_broadcast_chains[client_id]

        task.add_done_callback(_forget)

    async def _send_client_update_to_peers(self, msg_data: bytes) -> None:
        """Send a packed client update to every peer (outgoing and incoming)."""
        # Broadcast to outgoing peers
        for peer_id, peer in self.peers.items():
            if peer.connected and peer.websocket:
                try:
                    if hasattr(peer.websocket, 'send_bytes'):
                        await peer.websocket.send_bytes(msg_data)
                    else:
                        await peer.websocket.send(msg_data)
                except Exception as e:
                    logger.debug(
                        f"[Server] Error sending update to peer {peer_id}: {e}"
                    )

        # Broadcast to incoming serving_peers
        for peer_id, peer in self.serving_peers.items():
            if peer.connected and peer.websocket:
                try:
                    if hasattr(peer.websocket, 'send_bytes'):
                        await peer.websocket.send_bytes(msg_data)
                    else:
                        await peer.websocket.send(msg_data)
                except Exception as e:
                    logger.debug(
                        f"[Server] Error sending update to serving_peer {peer_id}: {e}"
                    )

    async def handle_client_connection(
        self,
        websocket: WebSocket,
        client_info: RegisteredClientInfo,
        authorized_addresses: Optional[List[str]] = None,
    ):
        """Handle a client WebSocket connection.

        ``authorized_addresses`` are the addresses the authenticated client is
        allowed to register routes for (a worker's own ``ip`` /
        ``advertise_address``). When provided, every claimed CIDR must be the
        host route of one of them. It is ``None`` for unauthenticated
        deployments, where only the generic claim policy (which also permits
        subnet claims) applies.
        """
        # Accept first — if the handshake fails nothing needs cleanup.
        await websocket.accept()

        client_id = client_info.client_id
        invalid = _invalid_client_claims(client_info)
        if invalid:
            logger.warning(
                f"[Server] Rejecting client {client_id} with invalid claims: {invalid}"
            )
            await websocket.close(
                code=1008, reason="invalid client registration claims"
            )
            return
        if authorized_addresses is not None:
            unauthorized = [
                cidr
                for cidr in client_info.cidrs
                if not _authorized_host_route(cidr, authorized_addresses)
            ]
            if unauthorized:
                logger.warning(
                    f"[Server] Rejecting client {client_id} claiming CIDRs outside "
                    f"its addresses: {unauthorized}"
                )
                await websocket.close(
                    code=1008, reason="invalid client registration claims"
                )
                return
        cidr_list = client_info.cidrs
        socket_list = client_info.unix_sockets

        # Set server_id so send_client_update_to_peer can filter correctly
        client_info.server_id = self._server_info.server_id

        # Registration and its "add" broadcast are serialized against the
        # release path (and its "remove" broadcast) by the per-client lock, so
        # a reconnecting websocket cannot interleave the two on peers. The
        # generation is taken under the same lock so generations follow
        # registration order.
        registered = False
        rejected = False
        generation: Optional[int] = None
        foreign: List[Tuple[str, uuid.UUID]] = []
        try:
            async with self._client_update_section(client_id):
                # The ownership check and the claim write run in one
                # await-free critical section guarded registry-wide, so two
                # different clients claiming the same CIDR cannot both pass
                # the check and have the later write steal the other's route.
                async with self._cidr_claims_lock:
                    foreign = self._cidr_claim_owners(client_id, cidr_list)
                    if not foreign:
                        self.connection_managers[client_id] = ConnectionManager(
                            websocket
                        )
                        self.client_registry[client_id] = client_info

                        # The registration's claim set is authoritative: a
                        # re-registered client whose address changed must not
                        # keep its old CIDRs.
                        self._cidr_registry.update_client(client_id, cidr_list)
                        registered = True

                if not registered:
                    logger.warning(
                        f"[Server] Rejecting client {client_id} claiming CIDRs "
                        "owned by another client: "
                        + ", ".join(
                            f"{cidr} (owner {owner})" for cidr, owner in foreign
                        )
                    )
                    rejected = True
                else:
                    # Get generation for this connection (used to filter
                    # stale disconnect callbacks).
                    generation = await self._get_next_generation(client_id)

                    logger.debug(
                        f"[Server] Client registered via WS: {client_id}, "
                        f"CIDRs: {cidr_list}"
                    )

                    # Broadcast new client to peers
                    self._enqueue_client_update(
                        "add", client_id, cidr_list, socket_list
                    )

            if rejected:
                # Close outside the per-client section, like release_client:
                # a hung close handshake must not stall the operations
                # serialized on this client.
                await websocket.close(
                    code=1008, reason="invalid client registration claims"
                )
                return

            if self._callback_on_connect:
                await self._safe_callback(self._callback_on_connect, None, client_info)
            await self.handle_client(websocket, client_id, generation)
        finally:
            # Safety net for anything between registration and handle_client
            # that raises or is cancelled. On the normal disconnect path this
            # runs AFTER handle_client's own finally (which releases the
            # registration and fires the disconnect callback), so the
            # ownership check below simply fails and the generation is left
            # untouched. Deregistration only happens when this websocket still
            # owns the registration, checked under the release lock so a
            # reconnect registering in between cannot be released by mistake.
            # Shielded so a cancellation of this task cannot abort the cleanup
            # mid-way.
            await asyncio.shield(
                self.release_client(client_id, only_if_websocket=websocket)
            )

    async def handle_server_federation(
        self,
        websocket: WebSocket,
        server_info: ServerInfo,
    ):
        """Handle incoming server-to-server federation connection"""
        our_server_id = self._server_info.server_id
        logger.debug(
            f"[Server] handle_server_federation: incoming={server_info.server_id}, self={our_server_id}"
        )
        # Prevent adding self as peer
        if server_info.server_id == our_server_id:
            logger.debug("[Server] Ignoring self-connection attempt")
            await websocket.close()
            return

        # Accept with our server info in response headers
        await websocket.accept(headers=self._server_info.to_bytes_headers())

        # Add to serving_peers (incoming connections)
        peer = ServerPeer(
            server_id=server_info.server_id,
            listen_address=server_info.listen_address,
            listen_port=server_info.listen_port,
            proxy_port=server_info.proxy_port,
            websocket=websocket,
            connected=True,
        )
        self.serving_peers[server_info.server_id] = peer

        logger.debug(f"[Server] Serving peer connected: {server_info.server_id}")

        if self._callback_on_connect:
            await self._safe_callback(self._callback_on_connect, server_info, None)

        # Send existing clients to new peer
        self.send_client_update_to_peer(websocket, "add")

        await self.handle_peer(websocket, server_info.server_id)

    async def _reap_when_stale(
        self,
        websocket: WebSocket,
        client_id: uuid.UUID,
        liveness: dict,
    ) -> None:
        """Close the websocket when a heartbeating client has gone silent.

        Only clients that have proven liveness (sent at least one heartbeat)
        are subject to the idle timeout; clients on older versions never
        heartbeat and cannot be told from a live silent connection. Reaping
        closes the socket instead of cancelling ``receive()`` with a timeout —
        a cancelled receive can discard a frame already being delivered.
        """
        while True:
            await asyncio.sleep(self._client_read_timeout)
            idle = time.monotonic() - liveness["last_activity"]
            if liveness["heartbeat_seen"] and idle >= self._client_read_timeout:
                logger.info(
                    f"[Server] Reaping stale tunnel for client {client_id}: "
                    f"no traffic for {idle:.0f}s"
                )
                with contextlib.suppress(Exception):
                    await websocket.close()
                return

    async def handle_client(
        self, websocket: WebSocket, client_id: uuid.UUID, generation: int
    ):
        """Handle a client connection"""
        liveness = {"heartbeat_seen": False, "last_activity": time.monotonic()}
        watchdog = asyncio.create_task(
            self._reap_when_stale(websocket, client_id, liveness)
        )
        try:
            while True:
                try:
                    message = await websocket.receive()
                    liveness["last_activity"] = time.monotonic()
                    if message.get("type") == "websocket.disconnect":
                        break

                    if "text" in message:
                        raw_data = (
                            message["text"].encode()
                            if isinstance(message["text"], str)
                            else message["text"]
                        )
                    elif "bytes" in message:
                        raw_data = message["bytes"]
                    else:
                        continue

                    msg = parse_message(raw_data)
                    logger.trace(f"[Server] Received: {msg.get_type()}")

                    if isinstance(msg, SessionBaseMessage):
                        if client_id in self.connection_managers:
                            await self.connection_managers[client_id].dispatch(msg)
                    elif isinstance(msg, HeartbeatMessage):
                        liveness["heartbeat_seen"] = True
                        response = HeartbeatMessage(timestamp=msg.timestamp)
                        await websocket.send_bytes(pack_message(response))

                except asyncio.CancelledError:
                    # Client is being closed, exit gracefully
                    break
                except Exception as e:
                    logger.debug(f"[Server] Error processing message: {e}")

        except Exception as e:
            logger.debug(f"[Server] Client error: {e}")
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog

            # Get client info before removing
            client_info = self.client_registry.get(client_id)
            cidr_list = client_info.cidrs if client_info else []
            socket_list = client_info.unix_sockets if client_info else []

            # The release and its "remove" broadcast are serialized against
            # registrations (and their "add" broadcast) by the per-client
            # section, and the sends themselves are chained per client, so
            # peers never observe a reconnect's "add" before an in-flight
            # "remove" for the same client.
            async with self._client_update_section(client_id):
                # A reconnecting client can register a new websocket under the
                # same client_id before this one's close is noticed. Release
                # the registration only while it still belongs to this
                # websocket.
                current = self.connection_managers.get(client_id)
                if current is not None and current.websocket is websocket:
                    self._cidr_registry.remove_client(client_id)
                    self.client_registry.pop(client_id, None)
                    del self.connection_managers[client_id]

                    # Broadcast client disconnection to peers
                    self._enqueue_client_update(
                        "remove", client_id, cidr_list, socket_list
                    )

            # Call disconnect callback (filtered by generation to avoid stale callbacks)
            if self._callback_on_disconnect:
                await self._safe_disconnect_callback(
                    self._callback_on_disconnect, client_info, generation
                )

            logger.debug(f"[Server] Client disconnected: {client_id}")


def handler_getter(websocket: WebSocket) -> MessageServerHandler:
    """FastAPI dependency: return the ``MessageServerHandler`` from ``app.state``."""
    return getattr(websocket.app.state, "message_server_handler", None)


def authenticator_getter(websocket: WebSocket) -> Authenticator:
    """FastAPI dependency: return the ``Authenticator`` from ``app.state``.

    Falls back to ``NoOpAuthenticator`` (accepts all connections) when no
    authenticator has been configured on the application.
    """
    authenticator: Authenticator = getattr(
        websocket.app.state, "websocket_authenticator", NoOpAuthenticator()
    )
    return authenticator


def _authenticated_worker_addresses(websocket: WebSocket) -> Optional[List[str]]:
    """Addresses the authenticated worker may register routes for.

    ``BearerTokenAuthenticator`` stores the resolved user on the websocket
    scope; the worker on it is the identity the bearer token was checked
    against, so its own addresses are the only routes it may claim.

    Returns ``None`` only when the connection carries no authenticated
    worker (standalone deployment). An authenticated worker with no
    reported addresses gets an empty list instead: falling back to
    ``None`` would disable the address check entirely and re-open the
    generic protocol policy (subnet claims) to it.
    """
    user = websocket.scope.get("user")
    worker = getattr(user, "worker", None)
    if worker is None:
        return None
    return [
        address
        for address in (
            getattr(worker, "ip", None),
            getattr(worker, "advertise_address", None),
        )
        if address
    ]


router = APIRouter()


@router.websocket(default_connect_path)
async def websocket_endpoint(
    websocket: WebSocket,
    handler: MessageServerHandler = Depends(handler_getter),
    authenticator: Authenticator = Depends(authenticator_getter),
):
    """WebSocket endpoint - handles both client and server connections"""
    try:
        if not await authenticator.authenticate(websocket):
            logger.debug("[Server] Authentication failed for connection")
            await websocket.close(code=4001, reason="Authentication failed")
            return
    except Exception as e:
        logger.debug(f"Server failed with: {e}")
        await websocket.close(code=1008, reason="Server Error")
        return

    # Check if this is a server federation connection
    server_info = handler._server_info_getter(websocket)
    if server_info:
        logger.debug(
            f"[Server] Detected server federation connection: {server_info.server_id}"
        )
        await handler.handle_server_federation(websocket, server_info)
        return

    client_info = handler._client_info_getter(websocket)
    if client_info:
        logger.debug(f"[Server] Detected client connection: {client_info.client_id}")
        await handler.handle_client_connection(
            websocket,
            client_info,
            authorized_addresses=_authenticated_worker_addresses(websocket),
        )
        return

    logger.debug(
        "[Server] No valid server or client info found in headers, rejecting connection"
    )
    await websocket.close(code=1008, reason="not a valid client or server connection")
