import asyncio
import contextlib
import errno
from functools import lru_cache
import os
import random
import socket
import time
from typing import Optional, Set, Tuple, List, Union
from urllib.parse import urlparse
import aiohttp
import psutil
from datetime import datetime, timezone
import ipaddress

import requests


def normalize_route_path(path: str) -> str:
    """
    Normalize the route path by adding / at the beginning if not present.
    """

    if not path.startswith("/"):
        path = "/" + path
    return path


def get_first_non_loopback_ip(expected_ifname: Optional[str] = None) -> str:
    """
    Get the first non-loopback IPv4 address of the machine.

    Returns:
        The IPv4 address as a string.
    """

    # Fallback to scanning all interfaces
    for name, addrs in psutil.net_if_addrs().items():
        if expected_ifname is not None and name != expected_ifname:
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                return addr.address
    if expected_ifname is not None:
        raise Exception(
            f"No non-loopback IPv4 address found on interface {expected_ifname}."
        )
    raise Exception("No non-loopback IPv4 address found.")


def is_ipaddress(ip_str: str) -> bool:
    """
    Check if the given string is a valid IP address.

    Returns:
        True if valid IP address, False otherwise.
    """
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def _get_ifname_by_local_ip(
    ip_address: str,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> Optional[str]:
    """
    Given an IP address, return the interface name if it exists and is not loopback/link-local.

    Returns:
        The interface name as a string, or None if not found.
    """

    try:
        ip = ipaddress.ip_address(ip_address)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_link_local:
        return None

    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == address_family and addr.address == ip_address:
                return ifname

    return None


def get_ifname_by_ip_hostname(
    ip_address_hostname: str,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> Optional[str]:
    """
    Get the interface name by IP address using psutil.

    Args:
        ip_address_hostname:
            The IP address or hostname to look for. If a hostname is provided, it will be resolved to an IP address.
        address_family:
            The address family (default is socket.AF_INET).

    Returns:
        The interface name associated with the given IP address or hostname.
    """

    local_ifname = _get_ifname_by_local_ip(
        ip_address_hostname, address_family=address_family
    )
    if local_ifname is not None:
        return local_ifname

    cases: List[Tuple[socket.AddressFamily, str]] = [
        (address_family, ip_address_hostname),
    ]
    if address_family == socket.AF_INET:
        cases.append((socket.AF_INET, "8.8.8.8"))
    if address_family == socket.AF_INET6:
        cases.append((socket.AF_INET6, "2001:4860:4860::8888"))

    for af, test_ip in cases:
        with contextlib.suppress(Exception):
            with socket.socket(af, socket.SOCK_DGRAM) as s:
                # the port is arbitrary since we won't actually send any data
                s.connect((test_ip, 1))
                local_ifname = _get_ifname_by_local_ip(s.getsockname()[0], af)
                if local_ifname is not None:
                    return local_ifname

    return None


def parse_port_range(port_range: str) -> Tuple[int, int]:
    """
    Parse the port range string to a tuple of start and end port.
    """

    start, end = port_range.split("-")
    return int(start), int(end)


def _normalize_port_range(port_range: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Accept either spelling of a range.

    Config carries it as the string "40000-40063"; callers that already
    parsed it hold the tuple. Taking both keeps every caller from having to
    re-parse a value it just parsed.
    """
    if isinstance(port_range, str):
        return parse_port_range(port_range)
    start, end = port_range
    return int(start), int(end)


class PortRangeExhaustedError(Exception):
    """No allocation of the requested width is left in the port range.

    Carries the arithmetic, not just the fact. The default pool is 64 ports
    (`service_port_range` = 40000-40063) and one TP8 Ascend instance takes
    1 + 8 of them, so exhaustion here is routinely a capacity-planning
    answer — "widen the range" — rather than a bug. The message says so
    instead of sending the operator to the source to work it out.
    """


def _describe_exhaustion(
    start: int,
    end: int,
    count: int,
    unavailable_ports: Set[int],
) -> str:
    pool = end - start + 1
    taken = len({port for port in unavailable_ports if start <= port <= end})
    free = pool - taken
    if free < count:
        reason = f"only {free} of the {pool} ports in the range are free"
    else:
        # Enough free ports overall, but no run of them: the caller asked for
        # a contiguous band and the pool is fragmented.
        reason = (
            f"{free} of the {pool} ports in the range are free, but no gap "
            f"of {count} consecutive ports is left"
        )
    return (
        f"No band of {count} consecutive free port(s) available in "
        f"{start}-{end}: {reason}. Widen the port range to at least "
        f"{pool + count} ports (e.g. {start}-{end + count})."
    )


def get_free_port(
    port_range: str,
    unavailable_ports: Optional[set[int]] = None,
    host: str = "127.0.0.1",
) -> int:
    start, end = parse_port_range(port_range)
    if unavailable_ports is None:
        unavailable_ports = set()

    if len(unavailable_ports) >= end - start + 1:
        raise PortRangeExhaustedError(
            _describe_exhaustion(start, end, 1, unavailable_ports)
        )

    while True:
        port = random.randint(start, end)
        if port in unavailable_ports:
            continue
        if is_port_available(port, host):
            return port
        else:
            unavailable_ports.add(port)
            if len(unavailable_ports) == end - start + 1:
                raise PortRangeExhaustedError(
                    _describe_exhaustion(start, end, 1, unavailable_ports)
                )
            continue


def get_free_band(
    port_range: Union[str, Tuple[int, int]],
    count: int = 1,
    unavailable_ports: Optional[Set[int]] = None,
    host: Optional[str] = None,
) -> int:
    """Return the *base* of a run of `count` consecutive free ports.

    Some connectors derive several ports from one base — Mooncake's kv_port
    on Ascend occupies TP-size consecutive ports (TP8 -> 41100-41107) — so
    allocation has to hand out bands, not points.

    This is deliberately not `get_free_port` generalized. That one picks a
    random port in the range, which is fine for single points but shatters a
    64-port pool into unusable one-port gaps: after a handful of random
    picks there is no run of 8 left anywhere, and the band allocation fails
    while the pool still looks two-thirds empty. First-fit contiguous
    scanning keeps the free space packed at the top of the range instead.

    `count=1` is therefore semantically "give me one free port", the same
    answer `get_free_port` gives, reached by scanning rather than sampling.

    `unavailable_ports` is read *and written*, matching `get_free_port`:
    ports found occupied by probing are memoized into it so a second call in
    the same allocation pass doesn't re-probe them. Callers who fence a whole
    band pass the same set back in.

    Raises:
        PortRangeExhaustedError: with the pool arithmetic and a concrete
            suggestion, never a bare "no free port".
    """
    if count < 1:
        raise ValueError(f"Port band count must be >= 1, got {count}.")

    start, end = _normalize_port_range(port_range)
    if unavailable_ports is None:
        unavailable_ports = set()
    probe_host = host or "127.0.0.1"

    base = start
    while base + count - 1 <= end:
        blocked: Optional[int] = None
        for port in range(base, base + count):
            if port in unavailable_ports:
                blocked = port
                break
            if not is_port_available(port, probe_host):
                # Memoize, so the restart of the scan below and any later
                # call in this pass skips it without another probe.
                unavailable_ports.add(port)
                blocked = port
                break
        if blocked is None:
            return base
        # First-fit: everything from `base` through the blocker can't start a
        # band, so resume right after it rather than at base + 1.
        base = blocked + 1

    raise PortRangeExhaustedError(
        _describe_exhaustion(start, end, count, unavailable_ports)
    )


def _bind_probe(port: int, host: str) -> Optional[bool]:
    """Try to bind `host:port`.

    Returns True if the bind succeeded (nothing holds the port), False if the
    port is taken or unusable by us, and None if the probe was inconclusive —
    e.g. `host` is not a local address on this machine, which says nothing
    about the port.

    SO_REUSEADDR is deliberately *not* set: with it Linux happily binds
    0.0.0.0:p alongside an existing 127.0.0.1:p, which is exactly the
    occupancy this probe exists to see.

    A consequence worth stating, because it looks like a bug from the outside:
    a port left in TIME_WAIT by a process that just exited reads as taken, for
    up to 2MSL (a minute on Linux). Retrying the bind with SO_REUSEADDR would
    tell that apart from a live listener, and is still not worth it. The
    asymmetry decides: calling a used port free leaves an engine wedged in
    `starting` forever, which is the failure this whole function was widened
    to prevent, while calling a free port used costs one more port out of a
    range far larger than the members on a host. If a range ever does get
    tight enough for TIME_WAIT to matter, widen the range -- do not loosen the
    probe.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
        return True
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES, errno.EPERM):
            # EADDRINUSE: somebody holds it. EACCES/EPERM: a privileged or
            # kernel-reserved port. Either way we can't have it.
            return False
        return None
    except Exception:
        return None


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Test if a port is available.

    Binds first, connects only as a fallback. A connect probe alone only sees
    a port that is *already LISTENing on `host`*:
    a server bound to 127.0.0.1, one that has bound but not yet listened, and
    (the common one) an engine that will bind seconds from now are all
    invisible to it. That last case is why a port collision shows up as an
    instance stuck forever in `starting` rather than a clean allocation
    failure.

    Both 0.0.0.0 and `host` are probed: a bind of 0.0.0.0:p fails if anything
    holds p on any address, and probing `host` too covers the case where
    0.0.0.0 could not be probed at all.

    **`host` has to be an address of the machine this runs on.** The answer is
    reached by binding, and binding only ever sees this machine -- the connect
    probe below is a fallback for when no bind was conclusive, and a local
    bind of 0.0.0.0 is always conclusive, so it never runs for a remote
    address. Every caller today passes the worker's own ip from the worker
    itself, which is why this holds; a caller that asked about another host
    would be told True for a port that host is listening on.

    Contract (several callers depend on it): returns a bool, never raises.

    Returns:
        True if the port is available, False otherwise.
    """

    probe_hosts = ["0.0.0.0"]
    if host and host not in probe_hosts:
        probe_hosts.append(host)

    conclusive = False
    for probe_host in probe_hosts:
        verdict = _bind_probe(port, probe_host)
        if verdict is False:
            return False
        if verdict is True:
            conclusive = True

    if conclusive:
        # A successful bind is stronger evidence than a refused connect;
        # nothing can be listening on a port we just bound.
        return True

    # Every bind probe was inconclusive (no local address to bind, a
    # sandboxed netns). Fall back to the connect probe: weaker, but better
    # than declaring the port free on no evidence.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.5)
            result = s.connect_ex((host, port))
            if result == 0:
                # Someone is listening, port is not available
                return False
        except Exception:
            pass

    return True


async def is_url_reachable(
    url: str, timeout_in_second: int = 10, retry_interval_in_second: int = 3
) -> bool:
    """Check if a url is reachable.

    Args:
        url (str): url to check.
        timeout (int): timeout in seconds. Defaults to 10.
        retry_interval_in_second (int, optional): retry inteval. Defaults to 3.
    Returns:
        bool: True if the url is reachable, False otherwise
    """
    end_time = time.time() + timeout_in_second
    while time.time() < end_time:
        try:
            use_proxy_env = use_proxy_env_for_url(url)
            async with aiohttp.ClientSession(trust_env=use_proxy_env) as session:
                async with session.get(url, timeout=2) as response:
                    if response.status == 200:
                        return True
        except Exception:
            await asyncio.sleep(retry_interval_in_second)
    return False


def is_offline(
    last_update: Optional[datetime],
    timeout_seconds: int,
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if the last_update time is offline based on the timeout_seconds.

    Args:
        last_update: The last update time (UTC datetime). If None, it means no record.
        timeout_seconds: The threshold in seconds to consider offline.
        now: The current time (UTC datetime), defaults to datetime.now(timezone.utc)

    Returns:
        Tuple[bool, Optional[str]]: (Whether offline, last_update readable string)
            - If last_update is None, returns "unknown"
            - Otherwise returns formatted time "%Y-%m-%d %H:%M:%S UTC"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if last_update is None:
        return True, "unknown"

    last_update_ts = int(last_update.timestamp())
    now_ts = int(now.timestamp())

    is_offline_flag = (now_ts - last_update_ts) > timeout_seconds
    last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
    return is_offline_flag, last_update_str


def check_registry_reachable(address: str) -> bool:
    """
    Check if the registry is reachable.
    To avoid frequent checks, cache the result for a short period via global lock.

    Returns:
        bool: True if the registry is reachable, False otherwise.
    """
    url = f"{address}/v2/"
    try:
        resp = requests.get(url, timeout=3)
        reachable = resp.status_code < 500
    except Exception:
        reachable = False
    return reachable


@lru_cache(maxsize=1)
def _get_no_proxy_cidrs() -> Tuple[ipaddress.IPv4Network, ...]:
    """
    Parse NO_PROXY environment variable to get a list of CIDR networks.
    """
    no_proxy = (os.getenv("NO_PROXY") or os.getenv("no_proxy") or "").strip()
    if not no_proxy:
        return ()
    cidrs = []
    for entry in no_proxy.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            net = ipaddress.IPv4Network(entry, strict=False)
            cidrs.append(net)
        except ValueError:
            # Ignore non-CIDR entries (including domain names, plain IPs, etc.)
            pass
    return tuple(cidrs)


def use_proxy_env_for_url(url: str) -> bool:
    """
    Determine if proxy environment variables (HTTP_PROXY, HTTPS_PROXY, etc.)
    should be used for the given URL.

    This is a workaround for the fact that current HTTP clients (e.g., httpx)
    do not support CIDR notation in NO_PROXY.
    Ref: https://github.com/encode/httpx/issues/1536

    - If the host is an IP address:
        Do **not** use proxy if it falls within any CIDR defined in NO_PROXY.
        -> Return False in that case.
    - If the host is a domain name:
        Defer to the HTTP client's standard NO_PROXY logic (which doesn't support CIDR),
        so assume proxy **should** be used unless explicitly overridden elsewhere.
        -> Return True.

    Args:
        url (str): Full URL (e.g., 'http://192.168.1.10:8080/path')

    Returns:
        bool: True if proxy environment variables should be used, False if the request
              should bypass the proxy (e.g., due to NO_PROXY CIDR match).
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return True

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # It's a domain name -> defer to standard NO_PROXY logic (no CIDR support)
            return True

        # Check against user-defined CIDRs in NO_PROXY
        for net in _get_no_proxy_cidrs():
            if ip in net:
                # Host is in a NO_PROXY CIDR -> bypass proxy
                return False

        return True
    except Exception:
        # On any error (e.g., malformed URL), default to using proxy
        return True
