import asyncio
import contextlib
import random
import socket
import time
from typing import Optional, Tuple, List
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


def get_first_non_loopback_ip() -> str:
    """
    Get the first non-loopback IPv4 address of the machine.

    Returns:
        The IPv4 address as a string.
    """

    # Fallback to scanning all interfaces
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                return addr.address

    raise Exception("No non-loopback IPv4 address found.")


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
                return _get_ifname_by_local_ip(s.getsockname()[0], af)

    return None


def parse_port_range(port_range: str) -> Tuple[int, int]:
    """
    Parse the port range string to a tuple of start and end port.
    """

    start, end = port_range.split("-")
    return int(start), int(end)


def get_free_port(port_range: str, unavailable_ports: Optional[set[int]] = None) -> int:
    start, end = parse_port_range(port_range)
    if unavailable_ports is None:
        unavailable_ports = set()

    if len(unavailable_ports) >= end - start + 1:
        raise Exception("No free port available in the port range.")

    while True:
        port = random.randint(start, end)
        if port in unavailable_ports:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                unavailable_ports.add(port)
                if len(unavailable_ports) == end - start + 1:
                    raise Exception("No free port available in the port range.")
                continue


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
            async with aiohttp.ClientSession(trust_env=True) as session:
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
