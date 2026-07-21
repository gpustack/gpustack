import logging
from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


def _split_host_port(value: str):
    """Split a Host value into ``(host, port)``; port is ``None`` when absent.

    Handles ``example.com``, ``example.com:8080``, ``[::1]``, ``[::1]:8080``
    and bare IPv6 literals (``::1``).
    """
    value = value.strip()
    if value.startswith("["):  # bracketed IPv6, optional :port
        end = value.find("]")
        if end == -1:
            return value, None
        host = value[1:end]
        rest = value[end + 1 :]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except ValueError:
                return host, None
        return host, None
    if value.count(":") == 1:  # host:port (a bare IPv6 has 2+ colons)
        host, _, port = value.partition(":")
        try:
            return host, int(port)
        except ValueError:
            return host, None
    return value, None  # bare hostname or bare IPv6


def _bare_host(value: str) -> str:
    """Lowercased hostname with brackets and port stripped."""
    return _split_host_port(value)[0].lower()


class ForwardedHostPortMiddleware:
    """
    Middleware to support X-Forwarded-Host.
    It rewrites the 'server' and 'headers' in the ASGI scope accordingly.

    X-Forwarded-Host is only honored when the forwarded host is trusted:
    either ``trusted_hosts`` contains ``"*"`` (trust any) or the forwarded
    host matches an allowlisted entry. Otherwise the header is ignored and
    the real Host is preserved, preventing Host header injection.
    """

    def __init__(self, app: ASGIApp, trusted_hosts: Optional[list] = None):
        self.app = app
        trusted_hosts = trusted_hosts or []
        self.wildcard = "*" in trusted_hosts
        self.allowed_hosts = {
            _bare_host(host) for host in trusted_hosts if host and host != "*"
        }
        # Debounce warnings; capped so a hostile client rotating fake hosts
        # cannot grow this unbounded.
        self._warned_hosts = set()

    def _is_trusted(self, forwarded_host: bytes) -> bool:
        if self.wildcard:
            return True
        # HTTP header octets carry latin-1 semantics (RFC 7230 / ASGI); latin-1
        # decodes any byte without raising, so no guard is needed.
        return _bare_host(forwarded_host.decode("latin-1")) in self.allowed_hosts

    def _warn_untrusted(self, forwarded_host: bytes):
        if forwarded_host in self._warned_hosts:
            return
        if len(self._warned_hosts) < 1024:
            self._warned_hosts.add(forwarded_host)
        logger.warning(
            f"Ignoring untrusted X-Forwarded-Host {forwarded_host!r}; "
            "not in trusted_hosts / server_external_url."
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict((k.lower(), v) for k, v in scope.get("headers", []))
            # X-Forwarded-Host
            forwarded_host = headers.get(b"x-forwarded-host")
            if forwarded_host:
                # Only use the first value if multiple hosts are present
                host = forwarded_host.split(b",")[0].strip()
                if self._is_trusted(host):
                    # Update the host header
                    new_headers = [
                        (k, v) if k != b"host" else (b"host", host)
                        for k, v in scope["headers"]
                    ]
                    # If no host header, add it
                    if not any(k == b"host" for k, _ in new_headers):
                        new_headers.append((b"host", host))
                    scope["headers"] = new_headers
                    # Update scope["server"] with the forwarded host/port.
                    # latin-1 matches header transport semantics and never raises,
                    # even under wildcard trust where the host is not pre-decoded.
                    host_name, host_port = _split_host_port(host.decode("latin-1"))
                    server = list(scope.get("server", (None, None)))
                    server[0] = host_name
                    if host_port is not None:
                        server[1] = host_port
                    scope["server"] = tuple(server)
                else:
                    self._warn_untrusted(host)
        await self.app(scope, receive, send)
