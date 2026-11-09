import hashlib
import random
import string
from typing import Optional

# Reserve four characters for the common -f<index> workload suffix.
INSTANCE_NAME_MAX_LENGTH = 59
WORKLOAD_NAME_MAX_LENGTH = 63


def bounded_instance_name(model_name: str, role_name: Optional[str] = None) -> str:
    """Generate an instance name within the workload label length budget.

    Args:
        model_name: Model or cache service name; character validation is separate.
        role_name: Optional role or cache service component name.

    Returns:
        A name with the full random suffix and at most 59 characters.
    """
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    budget = INSTANCE_NAME_MAX_LENGTH - len(suffix) - 1
    if role_name:
        # Leave room for a model character and the role separator.
        role_name = role_name[: budget - 2]
        budget -= len(role_name) + 1
        return f"{model_name[:budget]}-{role_name}-{suffix}"
    return f"{model_name[:budget]}-{suffix}"


def bounded_follower_name(instance_name: str, index: int) -> str:
    """Append a follower index, bounding length with a stable instance digest.

    Args:
        instance_name: Persisted instance name.
        index: Zero-based follower index.

    Returns:
        A deterministic workload name of at most 63 characters.
    """
    suffix = f"-f{index}"
    name = f"{instance_name}{suffix}"
    if len(name) <= WORKLOAD_NAME_MAX_LENGTH:
        return name
    # Include the entire instance identity, including its random suffix, so
    # truncating the readable prefix does not discard the distinguishing part.
    digest = hashlib.sha256(instance_name.encode()).hexdigest()[:16]
    budget = WORKLOAD_NAME_MAX_LENGTH - len(suffix) - len(digest) - 1
    return f"{instance_name[:budget]}-{digest}{suffix}"


def metric_name(name: str) -> str:
    METRIC_PREFIX = "gpustack:"
    return f"{METRIC_PREFIX}{name}"
