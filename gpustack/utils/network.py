import random
import socket
import netifaces


def normalize_route_path(path: str) -> str:
    """
    Normalize the route path by adding / at the beginning if not present.
    """

    if not path.startswith("/"):
        path = "/" + path
    return path


def get_first_non_loopback_ip():
    """
    Get the first non-loopback IP address of the machine.
    """

    for interface in netifaces.interfaces():
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for link in addresses[netifaces.AF_INET]:
                ip_address = link["addr"]
                if not ip_address.startswith("127.") and not ip_address.startswith(
                    "169.254."
                ):
                    return ip_address

    raise Exception("No non-loopback IP address found.")


def get_free_port(start=40000, end=41024) -> int:
    while True:
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
