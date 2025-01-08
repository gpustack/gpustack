import logging


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
