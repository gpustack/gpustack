from packaging import version


def in_range(version_str: str, range_str: str) -> bool:
    conditions = [cond.strip() for cond in range_str.split(",")]
    ver = version.parse(version_str)
    for cond in conditions:
        if cond.startswith(">="):
            if ver < version.parse(cond[2:]):
                return False
        elif cond.startswith("<="):
            if ver > version.parse(cond[2:]):
                return False
        elif cond.startswith(">"):
            if ver <= version.parse(cond[1:]):
                return False
        elif cond.startswith("<"):
            if ver >= version.parse(cond[1:]):
                return False
        else:  # exact match
            if ver != version.parse(cond):
                return False
    return True


def is_valid_version_str(version_str: str) -> bool:
    """
    Check if the version string is valid and can be parsed.
    Returns True if valid, False otherwise.
    """
    try:
        version.parse(version_str)
        return True
    except Exception:
        return False


def is_worker_version_compatible(
    worker_version: str,
    server_version: str,
) -> bool:
    """
    Check if worker and server versions are compatible.

    Args:
        worker_version: The version string of the worker.
        server_version: The version string of the server.

    Returns:
        bool: is_compatible
    """
    # Skip development version
    if worker_version == "0.0.0" or server_version == "0.0.0":
        return True

    try:
        worker_ver = version.parse(worker_version)
        server_ver = version.parse(server_version)
    except Exception:
        return False

    return worker_ver == server_ver
