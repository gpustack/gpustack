import shutil
import socket


def is_command_available(command_name):
    """
    Use `shutil.which` to determine whether a command is available.

    Args:
    command_name (str): The name of the command to check.

    Returns:
    bool: True if the command is available, False otherwise.
    """

    return shutil.which(command_name) is not None


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

    ip_addresses = socket.getaddrinfo(socket.gethostname(), None)

    for ip in ip_addresses:
        if ip[0] == socket.AF_INET:  # Check if the address family is IPv4
            ip_address = ip[4][0]
            if not ip_address.startswith("127.") and not ip_address.startswith(
                "169.254."
            ):  # Exclude loopback IP addresses and 169.254 addresses
                return ip_address

    return "No non-loopback IP address found."
