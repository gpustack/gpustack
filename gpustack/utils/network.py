import asyncio
import random
import socket
import time
from typing import Optional, Tuple
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
    Get the first non-loopback IP address of the machine using psutil.

    Returns:
        A tuple containing the IP address and the interface name.
    """

    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                return addr.address, ifname

    raise Exception("No non-loopback IP address found.")


def get_ifname_by_ip(ip_address: str) -> str:
    """
    Get the interface name by IP address using psutil.

    Args:
        ip_address: The IP address to look for.

    Returns:
        The interface name associated with the given IP address.
    """

    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip_address:
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
