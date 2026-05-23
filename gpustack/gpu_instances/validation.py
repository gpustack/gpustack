import re

from gpustack.api.exceptions import InvalidException

# Names must fit DNS-label rules so they can be safely projected into
# Kubernetes object names: up to 63 chars, lowercase alphanumerics and
# '-', starting with a letter and ending with a letter or digit.
_NAME_PATTERN = re.compile(r"[a-z]([a-z0-9-]{0,61}[a-z0-9])?")


def validate_k8s_object_name(name: str):
    """
    Validate that the given name is a valid Kubernetes object name.
    Raises an InvalidException if the name is invalid.
    """

    if not _NAME_PATTERN.fullmatch(name or ""):
        raise InvalidException(
            message=(
                "name must be at most 63 characters, "
                "contain only lowercase letters, digits, and '-', "
                "start with a lowercase letter, "
                "and end with a letter or digit"
            ),
        )
