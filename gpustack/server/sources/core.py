"""The source layer's shared behaviour: fetching a document, ordering the enabled
sources into one merge input, and materializing it. The data model those act on is
``gpustack.schemas.source``.

Kept free of the FastAPI/auth stack, so ``probe`` (official fetch) and ``routes``
(HTTP) build on top of it.
"""

import asyncio
import hashlib
import ipaddress
import logging
import socket
from typing import Awaitable, Callable, Iterable, List, Optional, Type
from urllib.parse import urlparse

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import __version__
from gpustack.schemas.source import SourceContent, SourceMixin, SourceTypeEnum

logger = logging.getLogger(__name__)


# --- Source-type precedence & naming ---------------------------------------


# Written by seed/probe, never through a source API. In merge-precedence order:
# these baselines go in first.
PLATFORM_OWNED = (SourceTypeEnum.BUILTIN, SourceTypeEnum.OFFICIAL)

# Carry a kind's whole content: any in service replaces the packaged baseline
# rather than layering over it (see ``order_source_contents``).
REMOTE_OWNED = (SourceTypeEnum.OFFICIAL, SourceTypeEnum.FILE, SourceTypeEnum.URL)

# The single OFFICIAL row of every source table. A user source masks it by
# disabling it; reset re-enables it.
OFFICIAL_SOURCE_NAME = "official"


def sha256_of(text: Optional[str]) -> Optional[str]:
    """The sha256 of a source document, or ``None`` when there is none."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Fetching a source document by URL -------------------------------------
#
# Fetched once, at PUT time, by the instance handling the request; every other
# server reads the normalized text from the DB.

# So a slow/unreachable host can't hang the request path.
_FETCH_TIMEOUT_SECONDS = 60

# So a huge body can't exhaust memory. Shared with the probe.
MAX_SOURCE_BYTES = 4 * 1024 * 1024

# Followed (a raw link 302s to a CDN) but bounded against loops.
MAX_REDIRECTS = 5


def _validate_source_url(url: str) -> None:
    """Reject non-http(s) transports (file://, ssh://, ext::) and embedded
    credentials — ``ValueError`` → HTTP 400."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("url must be an http(s) URL")
    if parts.username or parts.password:
        raise ValueError("url must not embed credentials")
    if not parts.hostname:
        raise ValueError("url must include a host")


async def reject_a_forbidden_address(request: httpx.Request) -> None:
    """Refuse a request whose host resolves to link-local (``ValueError``). An
    httpx hook, so it runs on every redirect hop — a public host can't 302 into
    the cluster.

    - link-local: refused (cloud metadata, and the credentials it hands out)
    - private / loopback: allowed (internal artifact servers are documented)
    - unresolvable here: left alone (a proxy resolves it, not this process)
    """
    host = request.url.host
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        return
    for address in addresses:
        # Strip an IPv6 scope ("fe80::1%en0"), which ip_address won't parse.
        resolved = ipaddress.ip_address(address[4][0].split("%")[0])
        if resolved.is_link_local:
            raise ValueError(
                f"{host} resolves to {resolved}, which a source must not fetch from"
            )


async def fetch_source_text(url: Optional[str]) -> str:
    """Fetch a source document over http(s). ``ValueError`` (→ HTTP 400) on a bad
    URL, error status, oversized body, or an HTML page (a repo *page* link pasted
    instead of the raw file)."""
    if not url:
        raise ValueError("url is required for a URL source")
    _validate_source_url(url)

    headers = {"User-Agent": f"gpustack/{__version__}"}

    chunks, total = [], 0
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            event_hooks={"request": [reject_a_forbidden_address]},
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    # Cap while streaming — Content-Length may be absent or lie.
                    if total > MAX_SOURCE_BYTES:
                        raise ValueError(
                            f"source exceeds the "
                            f"{MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
                        )
                    chunks.append(chunk)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        raise ValueError(f"failed to fetch the source: {e}")

    # Strict: reject a mis-encoded document rather than store/hash U+FFFD.
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"the source is not valid UTF-8: {e}")
    if text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
        raise ValueError("the URL returned an HTML page — use the raw file URL")
    return text


# --- The ordered merge -----------------------------------------------------


# Materializes the derived table from the ordered contents. Owns write + commit.
ReconcileFn = Callable[[AsyncSession, List[SourceContent]], Awaitable[None]]


def order_source_contents(sources: Iterable[SourceMixin]) -> List[SourceContent]:
    """The given sources' contents in a stable total order:

    1. platform baselines, in ``PLATFORM_OWNED`` order
    2. custom sources by ``(name, id)``

    Empty ones drop out. Ranks by ``source_type`` not name, so custom always
    wins. Shared with the pre-write check, which must order identically.

    Whole-content replacement: a remote document carries a kind's *entire* content
    (the official one is maintained by editing the packaged baseline), so any in
    service takes that baseline out of the merge rather than layering over it per
    key — which is also what lets an entry be *withdrawn*. Applied after the empty
    ones drop out, so an official slot never yet fetched (air-gapped, or before the
    first refresh round) leaves the baseline serving.
    """
    ordered = sorted(
        sources,
        key=lambda source: (
            (
                PLATFORM_OWNED.index(source.source_type)
                if source.source_type in PLATFORM_OWNED
                else len(PLATFORM_OWNED)
            ),
            source.name or "",
            source.id or 0,
        ),
    )
    contents = [
        SourceContent(source.name, source.source_type, source.content)
        for source in ordered
        if source.content
    ]
    if any(content.source_type in REMOTE_OWNED for content in contents):
        return [
            content
            for content in contents
            if content.source_type is not SourceTypeEnum.BUILTIN
        ]
    return contents


async def gather_and_merge(
    session: AsyncSession,
    source_cls: Type[SourceMixin],
    reconcile_fn: ReconcileFn,
) -> None:
    """Collect all enabled sources of ``source_cls`` and materialize them.
    ``order_source_contents`` is what makes every HA leader compute the same
    input; ``content`` is already normalized, so ``reconcile_fn`` can trust it."""
    sources = await source_cls.all(session)
    contents = order_source_contents(source for source in sources if source.enabled)
    await reconcile_fn(session, contents)
