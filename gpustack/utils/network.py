import asyncio
import contextlib
import random
import socket
import time
from typing import Optional, Tuple, List
import aiohttp
import psutil


def normalize_route_path(path: str) -> str:
    """
    Normalize the route path by adding / at the beginning if not present.
    """

    if not path.startswith("/"):
        path = "/" + path
    return path


def get_first_non_loopback_ip() -> Tuple[str, str]:
    """
    Get the first non-loopback IPv4 address and the interface name of the machine.

    Returns:
        A tuple containing the IPv4 address and the interface name.
    """

    target_af = socket.AF_INET

    # First try to get the preferred outbound IP
    outbound_ip, _ = _get_preferred_outbound_ip(target_af)
    if outbound_ip:
        ifname = get_ifname_by_ip(outbound_ip, target_af)
        return outbound_ip, ifname

    # Fallback to scanning all interfaces
    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == target_af and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                return addr.address, ifname

    raise Exception("No non-loopback IPv4 address found.")


def _get_preferred_outbound_ip(
    address_family: Optional[socket.AddressFamily] = None,
) -> Tuple[Optional[str], Optional[socket.AddressFamily]]:
    """
    Get the preferred outbound IP address of the machine.

    Args:
        address_family:
            The address family to use (socket.AF_INET or socket.AF_INET6).
            If None, try both.
    """

    cases: List[Tuple[socket.AddressFamily, str]] = []
    if address_family is None or address_family == socket.AF_INET:
        cases.append((socket.AF_INET, "8.8.8.8"))
    if address_family is None or address_family == socket.AF_INET6:
        cases.append((socket.AF_INET6, "2001:4860:4860::8888"))

    for af, test_ip in cases:
        with contextlib.suppress(Exception):
            with socket.socket(af, socket.SOCK_DGRAM) as s:
                s.connect((test_ip, 80))
                return s.getsockname()[0], af

    return None, None


def get_ifname_by_ip(
    ip_address: str,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> str:
    """
    Get the interface name by IP address using psutil.

    Args:
        ip_address:
            The IP address to look for.
        address_family:
            The address family (default is socket.AF_INET).

    Returns:
        The interface name associated with the given IP address.
    """

    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == address_family and addr.address == ip_address:
                return ifname

    raise Exception(f"No interface found for IP address {ip_address}.")


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
