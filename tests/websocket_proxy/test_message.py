import uuid
import pytest
import json
from gpustack.websocket_proxy.message import (
    # Classes
    ConnectRequestMessage,
    ConnectResponseMessage,
    DataMessage,
    DisconnectMessage,
    HeartbeatMessage,
    ListClientsMessage,
    ListClientsResponseMessage,
    ClientInfo,
    # Server <-> Server messages
    ClientUpdateMessage,
    ClientUpdateInfo,
    # Functions
    parse_message,
    message_to_json,
    # Constants
    DATA_COMPRESSION_NONE,
    DATA_COMPRESSION_GZIP,
    PROTOCOL_VERSION,
)


class TestConnectRequestMessage:
    def test_tcp(self):
        session_id = uuid.uuid4()
        msg = ConnectRequestMessage(
            session_id=session_id,
            target_url="tcp://192.168.1.1:8080",
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, ConnectRequestMessage)
        assert parsed.session_id == session_id
        assert parsed.target_url == "tcp://192.168.1.1:8080"

    def test_unix(self):
        session_id = uuid.uuid4()
        msg = ConnectRequestMessage(
            session_id=session_id,
            target_url="unix:///var/run/app.sock",
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, ConnectRequestMessage)
        assert parsed.target_url == "unix:///var/run/app.sock"


class TestConnectResponseMessage:
    def test_success(self):
        session_id = uuid.uuid4()
        msg = ConnectResponseMessage(session_id=session_id, success=True)

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, ConnectResponseMessage)
        assert parsed.success is True
        assert parsed.error is None

    def test_failure(self):
        session_id = uuid.uuid4()
        msg = ConnectResponseMessage(
            session_id=session_id, success=False, error="Connection refused"
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.success is False
        assert parsed.error == "Connection refused"


class TestDataMessage:
    def test_no_compression(self):
        session_id = uuid.uuid4()
        test_data = b"Hello, World!"
        msg = DataMessage(
            session_id=session_id, data=test_data, compression=DATA_COMPRESSION_NONE
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, DataMessage)
        assert parsed.session_id == session_id
        assert parsed.data == test_data
        assert parsed.compression == DATA_COMPRESSION_NONE

    def test_with_compression(self):
        session_id = uuid.uuid4()
        # Use repeated data to benefit from compression
        test_data = b"A" * 1000
        msg = DataMessage(
            session_id=session_id, data=test_data, compression=DATA_COMPRESSION_GZIP
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.data == test_data
        assert parsed.compression == DATA_COMPRESSION_GZIP


class TestDisconnectMessage:
    def test_normal_disconnect(self):
        session_id = uuid.uuid4()
        msg = DisconnectMessage(session_id=session_id)

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.session_id == session_id
        assert parsed.error is None

    def test_with_error(self):
        session_id = uuid.uuid4()
        msg = DisconnectMessage(session_id=session_id, error="Server closed")

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.error == "Server closed"


class TestHeartbeatMessage:
    def test_pack_and_parse(self):
        import time

        msg = HeartbeatMessage(timestamp=int(time.time()))

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, HeartbeatMessage)
        assert parsed.timestamp == msg.timestamp


class TestListClientsMessage:
    def test_pack_and_parse(self):
        msg = ListClientsMessage()

        data = msg.pack()
        parsed = parse_message(data)

        assert isinstance(parsed, ListClientsMessage)


class TestListClientsResponseMessage:
    def test_empty_clients(self):
        msg = ListClientsResponseMessage(clients=[])

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.clients == []

    def test_multiple_clients(self):
        clients = [
            ClientInfo(
                client_id=uuid.uuid4(),
                cidrs=["192.168.1.100"],
                unix_sockets=["/var/run/a.sock"],
            ),
            ClientInfo(
                client_id=uuid.uuid4(),
                cidrs=["10.0.0.1", "10.0.0.2"],
                unix_sockets=[],
            ),
        ]
        msg = ListClientsResponseMessage(clients=clients)

        data = msg.pack()
        parsed = parse_message(data)

        assert len(parsed.clients) == 2
        assert parsed.clients[0].cidrs == ["192.168.1.100"]
        assert parsed.clients[1].cidrs == ["10.0.0.1", "10.0.0.2"]


class TestMessageToJson:
    def test_connect_request_message(self):
        msg = ConnectRequestMessage(
            session_id=uuid.uuid4(),
            target_url="tcp://192.168.1.1:8080",
        )

        json_str = message_to_json(msg)
        data = json.loads(json_str)

        assert data["type"] == "connect_request"
        assert "session_id" in data
        assert data["target_url"] == "tcp://192.168.1.1:8080"

    def test_data_message(self):
        msg = DataMessage(session_id=uuid.uuid4(), data=b"\x00\x01\x02\x03")

        json_str = message_to_json(msg)
        data = json.loads(json_str)

        assert data["type"] == "data"
        assert data["data"] == "00010203"


class TestProtocolVersion:
    def test_invalid_version(self):
        msg = ConnectRequestMessage(
            session_id=uuid.uuid4(), target_url="tcp://1.2.3.4:80"
        )
        data = msg.pack()

        # Corrupt version byte
        corrupted = bytes([0xFF]) + data[1:]

        with pytest.raises(ValueError, match="Unsupported protocol version"):
            parse_message(corrupted)


class TestInvalidMessages:
    def test_too_short(self):
        with pytest.raises(ValueError, match="Message too short"):
            parse_message(b"\x01")

    def test_unknown_type(self):
        # Create valid header with unknown type
        data = bytes([PROTOCOL_VERSION, 0xFF])

        with pytest.raises(ValueError, match="Unknown binary message type"):
            parse_message(data)


class TestClientUpdateMessage:
    def test_add_client(self):
        server_id = uuid.uuid4()
        client_id = uuid.uuid4()
        msg = ClientUpdateMessage(
            server_id=server_id,
            updates=[
                ClientUpdateInfo(
                    client_id=client_id,
                    action="add",
                    cidrs=["192.168.1.100"],
                    unix_sockets=["/var/run/app.sock"],
                )
            ],
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.server_id == server_id
        assert len(parsed.updates) == 1
        assert parsed.updates[0].action == "add"
        assert parsed.updates[0].cidrs == ["192.168.1.100"]

    def test_remove_client(self):
        server_id = uuid.uuid4()
        client_id = uuid.uuid4()
        msg = ClientUpdateMessage(
            server_id=server_id,
            updates=[ClientUpdateInfo(client_id=client_id, action="remove")],
        )

        data = msg.pack()
        parsed = parse_message(data)

        assert parsed.updates[0].action == "remove"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
