"""Socket-level regression test for the CLOSE-WAIT leak.

The leak cannot be observed through mocks: it is the combined behavior of a
real TCP connection receiving the peer's FIN and httpcore only evicting dead
pooled connections when the next request touches the pool. This test uses a
loopback server (same precedent as tests/utils/test_network_ports.py) and
asserts kernel socket state via psutil:

1. positive control: a FINed keep-alive connection in a dormant pool does
   sit in CLOSE-WAIT (the leak condition is really reproduced);
2. the same connection is released as soon as HTTPClient.close() runs —
   the invariant every short-lived ClientSet now relies on.
"""

import socket
import threading
import time

import psutil
import pytest

from gpustack.client import ClientSet

pytestmark = pytest.mark.skipif(
    not hasattr(psutil, "CONN_CLOSE_WAIT"), reason="psutil lacks CLOSE_WAIT status"
)


class _KeepAliveServer:
    """Minimal HTTP/1.1 keep-alive server so httpx pools the connection."""

    def __init__(self):
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.port = self._listener.getsockname()[1]
        self._conns = []
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while True:
            conn, _ = self._listener.accept()
            self._conns.append(conn)
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn):
        try:
            buf = b""
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                buf += data
                if b"\r\n\r\n" in buf:
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Length: 2\r\n"
                        b"Connection: keep-alive\r\n"
                        b"\r\nok"
                    )
                    buf = b""
        except OSError:
            pass

    def send_fin_to_all(self):
        """Simulate a server/gateway teardown: FIN on every connection."""
        for conn in self._conns:
            try:
                conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass

    def url(self):
        return f"http://127.0.0.1:{self.port}"


def _close_wait_count(port):
    connections = psutil.Process().net_connections(kind="tcp")
    return len(
        [
            c
            for c in connections
            if c.status == psutil.CONN_CLOSE_WAIT and c.raddr and c.raddr.port == port
        ]
    )


def _await_close_wait(port, expected, attempts=20):
    """CLOSE-WAIT is kernel state set when the FIN arrives; poll briefly so
    the test does not depend on scheduling."""
    for _ in range(attempts):
        if _close_wait_count(port) == expected:
            return True
        time.sleep(0.05)
    return _close_wait_count(port) == expected


def test_fin_ed_pooled_connection_is_released_by_close():
    server = _KeepAliveServer()
    clientset = ClientSet(base_url=server.url())
    clientset.http_client.get_httpx_client().get("/probe")

    server.send_fin_to_all()

    # Positive control: without a close, the FINed connection sits in
    # CLOSE-WAIT. If this stops reproducing, the test below proves nothing.
    assert _await_close_wait(server.port, 1), "leak condition not reproduced"

    clientset.http_client.close()

    assert _await_close_wait(
        server.port, 0
    ), "pooled connection still in CLOSE-WAIT after HTTPClient.close()"


def test_fin_ed_pooled_connection_lingers_without_close():
    """Documents the failure mode the fix removes: a pool that never issues
    another request keeps its half-closed socket in CLOSE-WAIT."""
    server = _KeepAliveServer()
    clientset = ClientSet(base_url=server.url())
    clientset.http_client.get_httpx_client().get("/probe")

    server.send_fin_to_all()
    assert _await_close_wait(server.port, 1), "leak condition not reproduced"

    # The pool is dormant: no further request ever touches it.
    time.sleep(0.3)
    assert _close_wait_count(server.port) == 1

    clientset.http_client.close()
    assert _await_close_wait(server.port, 0)
