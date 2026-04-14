from gpustack.security import API_KEY_PREFIX


def get_masked_api_key_value(value: str, is_custom: bool = False) -> str:
    """Return masked API key value with partial access key visible."""
    if is_custom:
        return "Custom API Key"

    masked_value = "***"
    if len(value) >= 8:
        masked_value = f"{value[:4]}***"
    return f"{API_KEY_PREFIX}_{masked_value}"
