from packaging import version
from typing import Tuple


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
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    Check if worker and server versions are compatible.

    Args:
        worker_version: The version string of the worker.
        server_version: The version string of the server.
        strict: If True, require exact version match. If False (default), only require major.minor match.

    Returns:
        A tuple of (is_compatible: bool, reason: str).
    """
    # Skip development version
    if worker_version == "0.0.0" or server_version == "0.0.0":
        return True, "Development version detected, skipping check"

    try:
        worker_ver = version.parse(worker_version)
        server_ver = version.parse(server_version)
    except Exception as e:
        return False, f"Invalid version format: {e}"

    if strict:
        if worker_ver != server_ver:
            return (
                False,
                f"Strict mode: versions must match exactly (worker: {worker_version}, server: {server_version})",
            )
        return True, "Versions match exactly"
    else:
        if worker_ver.major != server_ver.major or worker_ver.minor != server_ver.minor:
            return (
                False,
                f"Incompatible versions: worker {worker_version} cannot connect to server {server_version}",
            )
        return True, "Versions are compatible"
