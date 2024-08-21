from urllib.parse import urlparse


def url(value: str) -> bool:
    """
    Validates whether the given string is a properly formatted URL.

    This function checks if the provided string has a valid URL scheme
    (e.g., 'http', 'https') and a valid hostname (e.g., 'localhost', 'example.com').

    Args:
        value (str): The URL string to be validated.

    Returns:
        bool: True if the string is a valid URL with a non-empty scheme and hostname,
              False otherwise.
    """
    parsed_url = urlparse(value)
    if parsed_url.scheme and parsed_url.hostname:
        return True

    return False
