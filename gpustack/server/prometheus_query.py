"""Talking to Prometheus, for whoever needs to.

Extracted when a second consumer appeared. The pieces here are the ones that
have nothing to do with what is being measured: issuing a query, deciding
whether the answer is usable, and turning a label value into something safe to
put inside a PromQL matcher. Leaving a second copy of the response parsing in
the tree is how two callers end up disagreeing about what "no data" means.

Everything semantic — which metrics, what they mean, how to judge them — stays
with its own feature.
"""

import json
import logging
import re
from typing import List, Optional

import aiohttp

from gpustack.config.config import get_global_config

logger = logging.getLogger(__name__)

_WINDOW_PATTERN = re.compile(r"^(\d+)([mhd])$")
_WINDOW_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}
MIN_WINDOW_SECONDS = 5 * 60
MAX_WINDOW_SECONDS = 7 * 86400

QUERY_TIMEOUT_SECONDS = 10.0


def api_base(base_url: str) -> str:
    """The API root under `base_url`.

    The built-in Prometheus is served under a `/prometheus` route prefix,
    mirroring the admin proxy route; a Prometheus somebody else runs is not.
    Hardcoding the prefix made every query against an external one 404, and a
    404 comes back as an empty result — which reads exactly like "no data",
    not like "wrong URL". Found by pointing at a real external Prometheus that
    was serving the series perfectly.

    Decided by whether the URL is the embedded one rather than by probing:
    a probe would add a round trip to every query to answer a question the
    configuration already answers.
    """
    if base_url.startswith("http://127.0.0.1:") and _is_builtin(base_url):
        return f"{base_url}/prometheus/api/v1"
    return f"{base_url}/api/v1"


def _is_builtin(base_url: str) -> bool:
    config = get_global_config()
    builtin = config.get_builtin_prometheus_url()
    return bool(builtin) and base_url.rstrip("/") == builtin.rstrip("/")


def prometheus_url() -> Optional[str]:
    """Where to send PromQL, or None when no Prometheus is reachable.

    None is a first-class answer and must never be read as "nothing is wrong":
    a deployment can legitimately disable the embedded Prometheus, or point at
    its own stack, and a caller that treats absence as a healthy zero would
    report every metric-backed check as passing.
    """
    return get_global_config().get_prometheus_url()


def promql_regex_literal(value: str) -> str:
    """A label value -> a safe literal inside a =~"..." matcher.

    Two escaping layers stack: RE2 metacharacters for the regex itself,
    then the PromQL string literal around it — its lexer follows Go and
    errors on unknown escape sequences, so re.escape's lone \\- would
    be a parse error rather than a literal dash. Backslashes double
    before quotes are escaped, or the added quote-escapes would double
    again."""
    return re.escape(value).replace("\\", "\\\\").replace('"', '\\"')


def parse_window(window: str) -> int:
    """A chart window like "30m" / "6h" / "7d" -> seconds.
    Raises ValueError outside [5m, 7d] or on an unknown format."""
    match = _WINDOW_PATTERN.match(window or "")
    if not match:
        raise ValueError(f"Invalid window '{window}': expected e.g. 30m, 6h, 7d")
    seconds = int(match.group(1)) * _WINDOW_UNIT_SECONDS[match.group(2)]
    if not MIN_WINDOW_SECONDS <= seconds <= MAX_WINDOW_SECONDS:
        raise ValueError(f"Window '{window}' out of range (5m to 7d)")
    return seconds


async def query_range(
    client: aiohttp.ClientSession,
    base_url: str,
    query: str,
    start: float,
    end: float,
    step: int,
) -> List[dict]:
    url = f"{api_base(base_url)}/query_range"
    async with client.get(
        url,
        params={
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        },
        timeout=aiohttp.ClientTimeout(total=QUERY_TIMEOUT_SECONDS),
    ) as response:
        return await read_result(response)


async def query_instant(
    client: aiohttp.ClientSession,
    base_url: str,
    query: str,
    at: float,
) -> List[dict]:
    url = f"{api_base(base_url)}/query"
    async with client.get(
        url,
        params={"query": query, "time": at},
        timeout=aiohttp.ClientTimeout(total=QUERY_TIMEOUT_SECONDS),
    ) as response:
        return await read_result(response)


async def read_result(response: aiohttp.ClientResponse) -> List[dict]:
    """Parse a Prometheus API response body. Every failure mode — a
    non-200 status, a non-JSON body (a gateway error page), an
    API-level error — raises ValueError, so a bad response stays
    isolated to its own query instead of blanking the collection."""
    body = await response.text()
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    if (
        response.status != 200
        or not isinstance(payload, dict)
        or payload.get("status") != "success"
    ):
        error = payload.get("error") if isinstance(payload, dict) else None
        raise ValueError(
            f"Prometheus returned {response.status}: "
            f"{error or body[:200] or 'unknown error'}"
        )
    return (payload.get("data") or {}).get("result") or []


def instant_value(entry: dict) -> Optional[float]:
    try:
        value = float(entry["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value
