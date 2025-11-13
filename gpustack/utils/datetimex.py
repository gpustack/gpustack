from datetime import datetime, timezone
from dateutil import parser


def parse_iso8601_to_utc(dt_str: str) -> datetime:
    """
    Parse an ISO8601 datetime string (with optional timezone) into a datetime
    object in UTC timezone.

    Handles nanoseconds by truncating to microseconds.

    Args:
        - dt_str: The ISO8601 datetime string to parse.
        Example:
            - UTC with Z: '2025-11-11T04:08:35.882997794Z'
            - With timezone offset: '2025-11-11T12:08:35.882997+08:00'
            - Without timezone (assumed UTC): '2025-11-11T04:08:35.882997'
    """
    dt = parser.isoparse(dt_str)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt
