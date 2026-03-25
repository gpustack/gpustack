import uuid
import gzip
import io
import json
import struct
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Type, Callable, TypeVar, Dict, Literal
from enum import IntEnum

# Protocol version
PROTOCOL_VERSION = 0x01

# TypeVar for message classes
M = TypeVar('M', bound='BaseMessage')


# ==================== Info Dataclasses ====================


@dataclass
class BaseClientInfo:
    """Base class for client information"""

    client_id: uuid.UUID
    cidrs: List[str] = field(default_factory=list)
    unix_sockets: List[str] = field(default_factory=list)

    def to_headers(self) -> Dict[str, str]:
        """Convert to headers dict for websockets client library."""
        return {
            'x-client-id': str(self.client_id),
            'x-cidrs': ','.join(self.cidrs),
            'x-unix-sockets': ','.join(self.unix_sockets),
        }


@dataclass
class RegisteredClientInfo(BaseClientInfo):
    """Information about a registered client"""

    server_id: Optional[uuid.UUID] = None  # Which server owns this client

    @classmethod
    def from_headers(cls, headers) -> Optional['RegisteredClientInfo']:
        """Create RegisteredClientInfo from headers dict or Starlette Headers object."""
        cidrs_str = (
            headers.get('x-cidrs', '')
            if hasattr(headers, 'get')
            else headers.get('x-cidrs', '')
        )
        unix_sockets_str = (
            headers.get('x-unix-sockets', '')
            if hasattr(headers, 'get')
            else headers.get('x-unix-sockets', '')
        )

        if not cidrs_str and not unix_sockets_str:
            return None

        client_id_str = headers.get('x-client-id', '')
        try:
            client_id = uuid.UUID(client_id_str) if client_id_str else uuid.uuid4()
        except ValueError:
            client_id = uuid.uuid4()
        cidr_list = [cidr.strip() for cidr in cidrs_str.split(',') if cidr.strip()]
        socket_list = [s.strip() for s in unix_sockets_str.split(',') if s.strip()]

        return cls(
            client_id=client_id,
            cidrs=cidr_list,
            unix_sockets=socket_list,
        )


@dataclass
class ServerInfo:
    """Information about a peer server"""

    server_id: uuid.UUID
    listen_address: Optional[str] = None
    listen_port: Optional[int] = None
    proxy_port: Optional[int] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to headers dict for websockets client library."""
        headers = {'x-server-id': str(self.server_id)}
        if self.listen_address is not None:
            headers['x-server-listen-address'] = self.listen_address
        if self.listen_port is not None:
            headers['x-server-listen-port'] = str(self.listen_port)
        if self.proxy_port is not None:
            headers['x-server-proxy-port'] = str(self.proxy_port)
        return headers

    def to_bytes_headers(self) -> List[Tuple[bytes, bytes]]:
        """Convert to headers list of tuples for Starlette WebSocket accept."""
        headers = [(b'x-server-id', str(self.server_id).encode())]
        if self.listen_address is not None:
            headers.append((b'x-server-listen-address', self.listen_address.encode()))
        if self.listen_port is not None:
            headers.append((b'x-server-listen-port', str(self.listen_port).encode()))
        if self.proxy_port is not None:
            headers.append((b'x-server-proxy-port', str(self.proxy_port).encode()))
        return headers

    @classmethod
    def from_headers(cls, headers) -> Optional['ServerInfo']:
        """Create ServerInfo from headers dict or Starlette Headers object."""
        try:

            def get_header(key: str) -> Optional[str]:
                # Starlette Headers object
                if hasattr(headers, 'get'):
                    try:
                        val = headers.get(key)
                        if val is not None:
                            return val
                    except (TypeError, KeyError):
                        pass
                # Dict with bytes or str keys
                if isinstance(headers, dict):
                    val = (
                        headers.get(key)
                        or headers.get(key.encode())
                        or headers.get(key.lower())
                    )
                    if val is not None:
                        return val.decode() if isinstance(val, bytes) else val
                return None

            server_id_str = get_header('x-server-id')
            if not server_id_str:
                return None
            return cls(
                server_id=uuid.UUID(server_id_str),
                listen_address=get_header('x-server-listen-address'),
                listen_port=(
                    int(v)
                    if (v := get_header('x-server-listen-port')) is not None
                    else None
                ),
                proxy_port=(
                    int(v)
                    if (v := get_header('x-server-proxy-port')) is not None
                    else None
                ),
            )
        except (ValueError, TypeError):
            return None


@dataclass
class ServerPeer(ServerInfo):
    """Information about a connected peer server"""

    websocket: Optional[object] = None
    connected: bool = False


# Binary message types
class BinaryType(IntEnum):
    # Client <-> Server messages
    CONNECT_REQUEST = 0x01
    CONNECT_RESPONSE = 0x02
    DATA = 0x03
    DISCONNECT = 0x04
    HEARTBEAT = 0x05
    LIST_CLIENTS = 0x06
    LIST_CLIENTS_RESPONSE = 0x07
    # Server <-> Server messages
    CLIENT_UPDATE = 0x08


# Protocol types
class BinaryProtocol(IntEnum):
    TCP = 0x01
    UDP = 0x02
    UNIX = 0x03


# Type strings
TYPE_CONNECT_REQUEST = "connect_request"
TYPE_CONNECT_RESPONSE = "connect_response"
TYPE_DATA = "data"
TYPE_DISCONNECT = "disconnect"
TYPE_HEARTBEAT = "heartbeat"
TYPE_LIST_CLIENTS = "list_clients"
TYPE_LIST_CLIENTS_RESPONSE = "list_clients_response"
TYPE_CLIENT_UPDATE = "client_update"

# Compression flags
DATA_COMPRESSION_NONE = 0x00
DATA_COMPRESSION_GZIP = 0x01


# ==================== Protocol Helpers ====================


def protocol_to_bytes(protocol: str) -> int:
    if protocol == "tcp":
        return BinaryProtocol.TCP
    elif protocol == "udp":
        return BinaryProtocol.UDP
    elif protocol == "unix":
        return BinaryProtocol.UNIX
    else:
        return 0


def bytes_to_protocol(b: int) -> str:
    if b == BinaryProtocol.TCP:
        return "tcp"
    elif b == BinaryProtocol.UDP:
        return "udp"
    elif b == BinaryProtocol.UNIX:
        return "unix"
    else:
        return ""


# ==================== Message Registry ====================


class MessageRegistry:
    """Registry for message types and their serialization"""

    _registry: dict[BinaryType, Type[M]] = {}
    _type_to_binary: dict[str, BinaryType] = {}

    @classmethod
    def register(
        cls, binary_type: BinaryType, msg_type: str
    ) -> Callable[[Type[M]], Type[M]]:
        def decorator(message_cls: Type[M]) -> Type[M]:
            cls._registry[binary_type] = message_cls
            cls._type_to_binary[msg_type] = binary_type
            return message_cls

        return decorator

    @classmethod
    def get_message_class(cls, binary_type: BinaryType) -> Optional[Type[M]]:
        return cls._registry.get(binary_type)

    @classmethod
    def get_binary_type(cls, msg_type: str) -> Optional[BinaryType]:
        return cls._type_to_binary.get(msg_type)


# ==================== Base Message Class ====================


class BaseMessage:
    """Base class for all messages with built-in serialization"""

    def get_type(self) -> str:
        raise NotImplementedError

    def pack(self) -> bytes:
        """Serialize message to binary format"""
        result = bytearray([PROTOCOL_VERSION])
        result.append(MessageRegistry.get_binary_type(self.get_type()))
        result.extend(self._pack_payload())
        return bytes(result)

    def _pack_payload(self) -> bytes:
        """Subclasses implement this to pack their payload"""
        raise NotImplementedError

    @classmethod
    def parse(cls, data: bytes) -> M:
        """Parse binary data into a message"""
        if len(data) < 2:
            raise ValueError("Message too short")

        version = data[0]
        if version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol version: {version}")

        msg_type = data[1]
        payload = data[2:]

        # Check if msg_type is a valid BinaryType value
        if msg_type not in BinaryType._value2member_map_:
            raise ValueError(f"Unknown binary message type: {msg_type}")

        message_cls = MessageRegistry.get_message_class(BinaryType(msg_type))
        if not message_cls:
            raise ValueError(f"Unknown binary message type: {msg_type}")

        return message_cls._parse_payload(payload)

    @classmethod
    def _parse_payload(cls, payload: bytes) -> M:
        """Subclasses implement this to parse their payload"""
        raise NotImplementedError


@dataclass
class SessionBaseMessage(BaseMessage):
    """Base class for messages that have a session_id (used for dispatching to ConnectionManager)"""

    session_id: uuid.UUID


# ==================== Message Definitions ====================

DataCompressor = Callable[[bytes], bytes]


def compress_gzip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(data)
    return buf.getvalue()


def decompress_gzip(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
        return gz.read()


_DATA_COMPRESSORS: dict[int, DataCompressor] = {
    DATA_COMPRESSION_NONE: lambda x: x,
    DATA_COMPRESSION_GZIP: compress_gzip,
}

_DATA_DECOMPRESSORS: dict[int, DataCompressor] = {
    DATA_COMPRESSION_NONE: lambda x: x,
    DATA_COMPRESSION_GZIP: decompress_gzip,
}


def _pack_string_list(strings: List[str]) -> bytes:
    """Pack a list of strings"""
    result = struct.pack(">H", len(strings))
    for s in strings:
        result += bytes([len(s)])
        result += s.encode()
    return result


def _unpack_string_list(payload: bytes, pos: int) -> tuple[List[str], int]:
    """Unpack a list of strings, returns (list, new_pos)"""
    count = struct.unpack(">H", payload[pos : pos + 2])[0]
    pos += 2
    strings = []
    for _ in range(count):
        if pos >= len(payload):
            raise ValueError("Invalid message")
        slen = payload[pos]
        pos += 1
        if pos + slen > len(payload):
            raise ValueError("Invalid message")
        strings.append(payload[pos : pos + slen].decode())
        pos += slen
    return strings, pos


def _pack_error(error: Optional[str]) -> bytes:
    if error:
        return bytes([len(error)]) + error.encode()
    return bytes([0])


def _unpack_error(payload: bytes, pos: int) -> tuple[Optional[str], int]:
    if pos >= len(payload):
        return None, pos
    err_len = payload[pos]
    pos += 1
    if err_len == 0:
        return None, pos
    if pos + err_len > len(payload):
        raise ValueError("Invalid message")
    return payload[pos : pos + err_len].decode(), pos + err_len


@MessageRegistry.register(BinaryType.CONNECT_REQUEST, TYPE_CONNECT_REQUEST)
@dataclass
class ConnectRequestMessage(SessionBaseMessage):
    """Server tells client to connect to target

    URL format: tcp://host:port or unix:///path/to/socket
    """

    target_url: str

    def get_type(self) -> str:
        return TYPE_CONNECT_REQUEST

    def _pack_payload(self) -> bytes:
        url_bytes = self.target_url.encode()
        return self.session_id.bytes + struct.pack(">H", len(url_bytes)) + url_bytes

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'ConnectRequestMessage':
        if len(payload) < 18:
            raise ValueError("Invalid connect request message")
        session_id = uuid.UUID(bytes=payload[:16])
        url_len = struct.unpack(">H", payload[16:18])[0]
        if len(payload) < 18 + url_len:
            raise ValueError("Invalid connect request message")
        target_url = payload[18 : 18 + url_len].decode()
        return cls(session_id=session_id, target_url=target_url)


@MessageRegistry.register(BinaryType.CONNECT_RESPONSE, TYPE_CONNECT_RESPONSE)
@dataclass
class ConnectResponseMessage(SessionBaseMessage):
    """Client response to connect request"""

    success: bool
    error: Optional[str] = None

    def get_type(self) -> str:
        return TYPE_CONNECT_RESPONSE

    def _pack_payload(self) -> bytes:
        result = self.session_id.bytes
        result += bytes([1 if self.success else 0])
        result += _pack_error(self.error)
        return result

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'ConnectResponseMessage':
        if len(payload) < 17:
            raise ValueError("Invalid connect response message")
        session_id = uuid.UUID(bytes=payload[:16])
        success = bool(payload[16])
        error, _ = _unpack_error(payload, 17)
        return cls(session_id=session_id, success=success, error=error)


@MessageRegistry.register(BinaryType.DATA, TYPE_DATA)
@dataclass
class DataMessage(SessionBaseMessage):
    """Data transmission message"""

    data: bytes
    compression: int = DATA_COMPRESSION_NONE

    def get_type(self) -> str:
        return TYPE_DATA

    def _pack_payload(self) -> bytes:
        compressor = _DATA_COMPRESSORS.get(self.compression, lambda x: x)
        compressed = compressor(self.data)
        result = self.session_id.bytes
        result += bytes([self.compression])
        result += struct.pack(">I", len(compressed))
        result += compressed
        return result

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'DataMessage':
        if len(payload) < 21:
            raise ValueError("Invalid data message")
        session_id = uuid.UUID(bytes=payload[:16])
        compression = payload[16]
        data_len = struct.unpack(">I", payload[17:21])[0]
        if len(payload) < 21 + data_len:
            raise ValueError("Invalid data message")

        decompressor = _DATA_DECOMPRESSORS.get(compression, lambda x: x)
        data = decompressor(payload[21 : 21 + data_len])

        return cls(session_id=session_id, data=data, compression=compression)


@MessageRegistry.register(BinaryType.DISCONNECT, TYPE_DISCONNECT)
@dataclass
class DisconnectMessage(SessionBaseMessage):
    """Connection close message"""

    error: Optional[str] = None

    def get_type(self) -> str:
        return TYPE_DISCONNECT

    def _pack_payload(self) -> bytes:
        result = self.session_id.bytes
        result += _pack_error(self.error)
        return result

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'DisconnectMessage':
        if len(payload) < 16:
            raise ValueError("Invalid disconnect message")
        session_id = uuid.UUID(bytes=payload[:16])
        error, _ = _unpack_error(payload, 16)
        return cls(session_id=session_id, error=error)


@MessageRegistry.register(BinaryType.HEARTBEAT, TYPE_HEARTBEAT)
@dataclass
class HeartbeatMessage(BaseMessage):
    """Keep-alive heartbeat"""

    timestamp: int = 0

    def get_type(self) -> str:
        return TYPE_HEARTBEAT

    def _pack_payload(self) -> bytes:
        return struct.pack(">Q", self.timestamp)

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'HeartbeatMessage':
        if len(payload) < 8:
            raise ValueError("Invalid heartbeat message")
        timestamp = struct.unpack(">Q", payload[:8])[0]
        return cls(timestamp=timestamp)


@MessageRegistry.register(BinaryType.LIST_CLIENTS, TYPE_LIST_CLIENTS)
@dataclass
class ListClientsMessage(BaseMessage):
    """Request list of connected clients"""

    def get_type(self) -> str:
        return TYPE_LIST_CLIENTS

    def _pack_payload(self) -> bytes:
        return b""

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'ListClientsMessage':
        return cls()


@dataclass
class ClientInfo(BaseClientInfo):
    """Information about a connected client (for LIST_CLIENTS_RESPONSE)"""

    pass


@MessageRegistry.register(BinaryType.LIST_CLIENTS_RESPONSE, TYPE_LIST_CLIENTS_RESPONSE)
@dataclass
class ListClientsResponseMessage(BaseMessage):
    """Response with list of clients"""

    clients: List[ClientInfo] = field(default_factory=list)

    def get_type(self) -> str:
        return TYPE_LIST_CLIENTS_RESPONSE

    def _pack_payload(self) -> bytes:
        result = struct.pack(">H", len(self.clients))
        for client in self.clients:
            result += client.client_id.bytes
            result += _pack_string_list(client.cidrs)
            result += _pack_string_list(client.unix_sockets)
        return result

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'ListClientsResponseMessage':
        if len(payload) < 2:
            raise ValueError("Invalid list clients response message")
        count = struct.unpack(">H", payload[:2])[0]
        clients = []
        pos = 2

        for _ in range(count):
            if pos + 16 > len(payload):
                raise ValueError("Invalid list clients response message")
            client_id = uuid.UUID(bytes=payload[pos : pos + 16])
            pos += 16

            cidrs, pos = _unpack_string_list(payload, pos)
            unix_sockets, pos = _unpack_string_list(payload, pos)

            clients.append(
                ClientInfo(
                    client_id=client_id,
                    cidrs=cidrs,
                    unix_sockets=unix_sockets,
                )
            )

        return cls(clients=clients)


# ==================== Server <-> Server Messages ====================


@dataclass
class ClientUpdateInfo(BaseClientInfo):
    """Client information in an update message"""

    action: Literal["add", "remove"] = ""


@MessageRegistry.register(BinaryType.CLIENT_UPDATE, TYPE_CLIENT_UPDATE)
@dataclass
class ClientUpdateMessage(BaseMessage):
    """Server notifies peers about client changes"""

    server_id: uuid.UUID
    updates: List[ClientUpdateInfo] = field(default_factory=list)

    def get_type(self) -> str:
        return TYPE_CLIENT_UPDATE

    def _pack_payload(self) -> bytes:
        result = self.server_id.bytes
        result += struct.pack(">H", len(self.updates))
        for update in self.updates:
            result += update.client_id.bytes
            result += bytes([1 if update.action == "add" else 0])
            result += _pack_string_list(update.cidrs)
            result += _pack_string_list(update.unix_sockets)
        return result

    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'ClientUpdateMessage':
        if len(payload) < 18:
            raise ValueError("Invalid client update message")
        server_id = uuid.UUID(bytes=payload[:16])
        pos = 16

        count = struct.unpack(">H", payload[pos : pos + 2])[0]
        pos += 2

        updates = []
        for _ in range(count):
            if pos + 16 > len(payload):
                raise ValueError("Invalid client update message")
            client_id = uuid.UUID(bytes=payload[pos : pos + 16])
            pos += 16

            action = "add" if payload[pos] == 1 else "remove"
            pos += 1

            cidrs, pos = _unpack_string_list(payload, pos)
            unix_sockets, pos = _unpack_string_list(payload, pos)

            updates.append(
                ClientUpdateInfo(
                    client_id=client_id,
                    action=action,
                    cidrs=cidrs,
                    unix_sockets=unix_sockets,
                )
            )

        return cls(server_id=server_id, updates=updates)


# ==================== Convenience Functions ====================


def pack_message(msg: BaseMessage) -> bytes:
    """Pack a message into binary format (backward compatible)"""
    return msg.pack()


def parse_message(data: bytes) -> BaseMessage:
    """Parse binary data into a message (backward compatible)"""
    return BaseMessage.parse(data)


def message_to_json(msg: BaseMessage) -> str:
    """Convert message to JSON string (for debugging)"""

    def serialize_value(v):
        if isinstance(v, uuid.UUID):
            return str(v)
        if isinstance(v, bytes):
            return v.hex()
        if isinstance(v, list):
            return [serialize_value(x) for x in v]
        if isinstance(v, ClientInfo):
            return {
                "client_id": str(v.client_id),
                "cidrs": v.cidrs,
                "unix_sockets": v.unix_sockets,
            }
        return v

    result = {"type": msg.get_type()}
    for field_name in msg.__dataclass_fields__:
        value = getattr(msg, field_name)
        result[field_name] = serialize_value(value)

    return json.dumps(result)
