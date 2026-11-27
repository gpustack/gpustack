"""Request accounting for measured points and their sampling boundaries."""

from typing import Optional


def cancelled_at_boundary(point: dict) -> int:
    """Return verified sampling cancellations without excluding server errors."""
    raw = point.get("raw_metrics")
    termination = raw.get("termination") if isinstance(raw, dict) else None
    if not isinstance(termination, dict) or termination.get("reason") != "min_requests":
        return 0
    cancelled = termination.get("cancelled_at_boundary")
    if not isinstance(cancelled, int) or isinstance(cancelled, bool):
        return 0
    incomplete = point.get("request_incomplete") or 0
    available = (point.get("request_total") or 0) - (
        (point.get("request_successful") or 0) + (point.get("request_errored") or 0)
    )
    return cancelled if 0 <= cancelled <= min(incomplete, available) else 0


def unplanned_incomplete(point: dict) -> int:
    """Count incomplete requests that still indicate a partial failure."""
    return max(0, (point.get("request_incomplete") or 0) - cancelled_at_boundary(point))


def success_rate(point: dict) -> Optional[float]:
    """Return evaluated success fraction, or None when no requests are evaluable."""
    total = (point.get("request_total") or 0) - cancelled_at_boundary(point)
    if total <= 0:
        return None
    return (point.get("request_successful") or 0) / total
