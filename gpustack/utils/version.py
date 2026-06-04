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

    Worker and server are built from the same release tag, so their
    version strings are byte-equal in practice. A plain string compare
    is therefore sufficient — and avoids PEP 440 parsing, which would
    reject otherwise valid release-tag forms (e.g. a trailing build
    suffix) and surface a spurious "incompatible" warning.

    Args:
        worker_version: The version string of the worker.
        server_version: The version string of the server.

    Returns:
        bool: is_compatible
    """
    # Skip development version
    if worker_version == "0.0.0" or server_version == "0.0.0":
        return True

    # An unresolved version on either side can't be confirmed as
    # compatible — surface a warning instead of silently matching two
    # empty / placeholder strings (e.g. when the /version response is
    # missing the field, worker_manager defaults it to "unknown").
    if not worker_version or worker_version == "unknown":
        return False
    if not server_version or server_version == "unknown":
        return False

    return worker_version == server_version
