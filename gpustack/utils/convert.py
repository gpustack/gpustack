import logging
import re
import zlib
import base64

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    return safe_convert(value, float, default=default)


def safe_int(value, default=0):
    return safe_convert(value, int, default=default)


def safe_convert(value, target_type, default):
    """
    Safely converts a value to the specified target type.
    If conversion fails, returns the default value.
    """
    try:
        return target_type(value)
    except Exception:
        return default


def parse_duration(duration_str: str, default: int = 0) -> int:
    duration_str = duration_str.strip()

    if not re.fullmatch(r'^(\d+[hms])+$', duration_str):
        logger.warning(
            f"Invalid duration format: {duration_str}. Using default value: {default}"
        )
        return default

    try:
        matches = re.findall(r'(\d+)([hms])', duration_str)
        time_units = {'h': 3600, 'm': 60, 's': 1}
        total_seconds = 0

        for value, unit in matches:
            total_seconds += int(value) * time_units[unit]

        return total_seconds

    except Exception as e:
        logger.error(f"Error parsing duration: {e}")
        return default


def safe_b64decode(data):
    # Remove illegal characters and add equal signs
    cleaned = re.sub(r'[^A-Za-z0-9+/=]', '', data)
    padding = '=' * (-len(cleaned) % 4)
    return base64.b64decode(cleaned + padding)


def inflate_data(compressed):
    return zlib.decompress(compressed, -15)
