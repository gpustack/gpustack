from functools import cmp_to_key

from gpustack_runtime.deployer.__utils__ import compare_versions
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


def version_in_range(version_str: str, range_str: str):
    """
    Tolerant ``in_range``: returns True/False when both sides parse,
    None when either side cannot be parsed — enforcement built on this
    fails open, so an exotic version string never blocks a deployment.
    """
    try:
        return in_range(version_str, range_str)
    except Exception:
        return None


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


def major_version(version_str):
    """Major segment of a version string, e.g. "12" from "v12.8"."""
    if not version_str:
        return None
    return version_str.removeprefix("v").split(".", 1)[0]


def pick_runtime_version(candidates, host_version):
    """Pick which declared runtime version an image build should target
    for a host, shared by inference-backend runner resolution and cache
    provider ``runtime_images``:

    - no host runtime detected -> the newest candidate;
    - otherwise the newest candidate <= the host version (an image built
      against an older runtime runs on a newer host, not vice versa);
    - every candidate newer than the host -> the newest one sharing the
      host's major (same-major minor compatibility holds), else the
      oldest overall;
    - no candidates -> None.

    Comparisons use gpustack_runtime's tolerant compare_versions, not
    PEP 440 parsing: runner and runtime version strings (e.g. CANN's
    "8.1.RC1.alpha003") are not guaranteed to be PEP 440.
    """
    ordered = sorted(
        candidates,
        key=cmp_to_key(compare_versions),
        reverse=True,
    )
    if not ordered:
        return None
    if not host_version:
        return ordered[0]
    for candidate in ordered:
        if compare_versions(candidate, host_version) <= 0:
            return candidate
    host_major = major_version(host_version)
    return next(
        (candidate for candidate in ordered if major_version(candidate) == host_major),
        ordered[-1],
    )
