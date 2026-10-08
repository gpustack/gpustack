from typing import Optional

from gpustack.security import API_KEY_PREFIX


def mask_credential(value: Optional[str]) -> str:
    """Return a log-safe representation of a credential.

    Empty values get an explicit placeholder, and short values are fully
    masked so logging never reveals a complete secret. Longer values retain
    only their final four characters for correlation during troubleshooting.
    """
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * min(len(value) - 4, 64)}{value[-4:]}"


def get_masked_api_key_value(value: str, is_custom: bool = False) -> str:
    """Return masked API key value with partial access key visible."""
    if is_custom:
        return "Custom API Key"

    masked_value = "***"
    if len(value) >= 8:
        masked_value = f"{value[:4]}***"
    return f"{API_KEY_PREFIX}_{masked_value}"
