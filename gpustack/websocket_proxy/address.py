"""IPv4/IPv6 host and authority helpers for websocket_proxy."""

from typing import Optional, Tuple, Union
from urllib.parse import urlsplit


def strip_brackets(host: str) -> str:
    """Strip surrounding RFC 3986 square brackets from an IPv6 literal."""
    if host and host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def format_host(host: str) -> str:
    """Wrap bare IPv6 literals in ``[...]`` while leaving IPv4/DNS/bracketed hosts unchanged."""
    if not host:
        return host
    bare = strip_brackets(host)
    if ":" in bare and "%" in bare:
        bare = bare.split("%", 1)[0]
    return f"[{bare}]" if ":" in bare else bare


def join_host_port(host: str, port: Union[int, str]) -> str:
    """Combine ``host`` and ``port`` into ``host:port`` or ``[ipv6]:port``."""
    return f"{format_host(host)}:{port}"


def split_authority(
    authority: str, default_port: Optional[int] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Parse ``host[:port]`` or ``[ipv6][:port]`` into ``(bare_host, port)``."""
    if not authority:
        return None, default_port
    candidate = authority
    if candidate.count(":") > 1 and not candidate.startswith("["):
        candidate = f"[{candidate}]"
    parsed = urlsplit(f"//{candidate}")
    return parsed.hostname, parsed.port if parsed.port is not None else default_port
