import os
import posixpath
import re
import stat
from mimetypes import guess_type
from typing import Optional, Tuple

import anyio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope

# One year, the conventional "forever" — the longest value HTTP caches are
# expected to honour.
IMMUTABLE_MAX_AGE = 365 * 24 * 60 * 60

# A build-injected cache buster: ``umi.530e136d.js``, ``bagel.16fb8279.png``,
# ``p__playground__index.1752132356288.chunk.css``. Its presence is what makes
# a URL content-addressed, and therefore safe to cache forever.
_CACHE_BUSTED_NAME = re.compile(r"\.[0-9a-f]{8,}\.")

CACHE_FOREVER = f"public, max-age={IMMUTABLE_MAX_AGE}, immutable"
CACHE_REVALIDATE = "public, no-cache"

# What Starlette's FileResponse settles on when `mimetypes` cannot guess. Kept
# in step with it deliberately: the gzip and plain answers to one URL have to
# agree on the type, including for extensions neither of us recognises.
FILE_RESPONSE_FALLBACK_MEDIA_TYPE = "application/octet-stream"


def cache_control_for(path: str) -> str:
    """The caching policy for a UI asset, decided by its file name.

    Content-addressed names can be cached forever: a new build produces a new
    name, so a stale entry is never *reachable* rather than merely unlikely.
    Everything else keeps a stable name across builds and must be revalidated
    — ``no-cache`` still lets the client store the body and settle the check
    with a 304, so the cost is one round trip, not a re-download.

    The split is per-file rather than per-directory because the build mixes
    both kinds in one place: ``static/`` holds cache-busted bundles and fonts
    next to ``catalog_icons/qwen.png``, which the backend ships under a fixed
    name and can replace in-place at any release.

    The two ways this can be wrong are not equally bad. Failing to recognise a
    hash costs a round trip per load; mistaking a stable name for one — a hex
    run of eight or more between dots that is not a cache buster — pins that
    asset in browsers for a year with no way to invalidate it. Widen the
    pattern only against real build output.
    """
    if _CACHE_BUSTED_NAME.search(posixpath.basename(path)):
        return CACHE_FOREVER
    return CACHE_REVALIDATE


def accepts_gzip(headers: Headers) -> bool:
    """Whether the client asked for gzip in ``Accept-Encoding``.

    Parses q-values rather than substring-matching, so an explicit refusal
    (``gzip;q=0``, which some proxies send to opt out) is honoured instead of
    being read as acceptance. A named coding beats ``*``, per RFC 9110 §12.5.3
    — ``gzip;q=0, *`` is a refusal of gzip, not an acceptance of everything.
    """
    qualities: dict[str, float] = {}
    for part in headers.get("accept-encoding", "").split(","):
        coding, _, params = part.partition(";")
        coding = coding.strip().lower()
        if coding not in ("gzip", "*"):
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value.strip())
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality

    if "gzip" in qualities:
        return qualities["gzip"] > 0
    return qualities.get("*", 0.0) > 0


class PrecompressedStaticFiles(StaticFiles):
    """Serve the UI build's ``.gz`` siblings to clients that accept gzip.

    The gpustack-ui build already emits ``<asset>.gz`` next to every asset over
    ~10 KiB — the bundles that dominate first-load time. Plain ``StaticFiles``
    ignores those files entirely and ships the uncompressed original, so the
    work the build did is thrown away on every request. Compressing on the fly
    instead would redo that work per request for a worse ratio (the build can
    afford a slower, higher level), so prefer the precompressed file and fall
    back to the plain asset when there is none — small chunks and images.

    Every answer also carries a ``Cache-Control`` from ``cache_control_for``
    and the ``Vary`` keys that make it safe to store — see ``_apply_policy``.
    Starlette sets an ETag and Last-Modified but no freshness lifetime at all,
    which leaves the browser guessing and in practice revalidating every asset
    on every load.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        if scope["method"] in ("GET", "HEAD") and accepts_gzip(Headers(scope=scope)):
            response = await self._precompressed_response(path, scope)
            if response is not None:
                return response

        response = await super().get_response(path, scope)
        self._apply_policy(response, path)
        return response

    @staticmethod
    def _apply_policy(response: Response, path: str) -> None:
        """The caching contract, applied to every answer including a 304.

        A Not Modified response is how a client refreshes an expired entry, so
        leaving the policy off it would give the refreshed copy no lifetime and
        force another revalidation on the very next load.

        Both ``Vary`` keys describe a way this URL's response changes with the
        request, and a cache that is not told will keep one entry and replay it
        for the rest. ``Accept-Encoding``, or it may hand a gzip body to a
        client that never asked for one. ``Origin``, because CORSMiddleware
        adds ``Access-Control-Allow-Origin`` only when the request carries an
        Origin, and with ``allow_origins=["*"]`` it does not declare that
        variance itself — so an entry stored from an ordinary same-origin load
        would come back, header missing, to a cross-origin one, and stay
        blocked for the whole immutable lifetime.
        """
        headers = response.headers
        headers.add_vary_header("Accept-Encoding")
        headers.add_vary_header("Origin")
        headers["cache-control"] = cache_control_for(path)

    def _lookup_precompressed(self, path: str) -> Optional[Tuple[str, os.stat_result]]:
        """Locate the ``.gz`` for ``path``, or None if there is nothing to serve.

        Requires the plain asset to exist too. A ``.gz`` on its own is not an
        encoding of anything, and serving it would make the URL resolve for
        clients that accept gzip and 404 for everyone else — whether the
        resource exists would depend on ``Accept-Encoding``, which is meant to
        select among representations, not decide existence.

        Both stats happen here, in whichever thread the caller dispatched to,
        rather than in a second ``run_sync``: the handoff costs ~80 µs against
        ~1.5 µs for the stat itself. The plain asset is only checked once the
        ``.gz`` is known to be there, so the common case of an asset with no
        ``.gz`` is no more expensive than before.
        """
        gz_path, gz_stat = self.lookup_path(f"{path}.gz")
        if gz_stat is None or not stat.S_ISREG(gz_stat.st_mode):
            return None

        _, plain_stat = self.lookup_path(path)
        if plain_stat is None or not stat.S_ISREG(plain_stat.st_mode):
            return None

        return gz_path, gz_stat

    async def _precompressed_response(
        self, path: str, scope: Scope
    ) -> Optional[Response]:
        """The ``.gz`` sibling of ``path``, or None to fall back to the plain file."""
        try:
            found = await anyio.to_thread.run_sync(self._lookup_precompressed, path)
        except (OSError, ValueError, HTTPException):
            # A .gz we cannot stat is not worth failing the request over — the
            # plain asset is a complete answer on its own, and falling through
            # hands the path to Starlette, which has its own answer for each of
            # these. ValueError is the one that bites: os.stat raises it, not
            # an OSError, for a path holding a null byte, so leaving it out
            # turned "/asset%00" into a 500 for gzip-accepting clients while
            # everyone else still got Starlette's 404. HTTPException is here
            # because the dependency range spans a major version of Starlette
            # and a build that signals a miss by raising must not turn every
            # asset without a .gz sibling into a 404.
            return None

        if found is None:
            return None

        full_path, stat_result = found
        response = self.file_response(full_path, stat_result, scope)
        # Keyed on the requested path, not the .gz we resolved to: the two
        # encodings are the same resource and must expire together.
        self._apply_policy(response, path)
        if response.status_code == 200:
            response.headers["content-encoding"] = "gzip"
            # Type the asset the browser ends up with, not the .gz wrapper it
            # arrives in: it has to parse this as JS/CSS once decompressed.
            # Derived from the original path so it never depends on how
            # mimetypes happens to treat a ``.gz`` suffix.
            response.headers["content-type"] = _content_type_for(path)
        return response


def _content_type_for(path: str) -> str:
    """The Content-Type ``path`` would get if it were served uncompressed.

    Mirrors what ``FileResponse`` puts on the plain asset, charset included —
    the two representations of a URL must not disagree about how to decode the
    bytes, or the same script parses differently depending on whether the
    client happened to accept gzip. That includes the fallback: an extension
    `mimetypes` does not know (a `.map`, say) has to land on the same guess
    `FileResponse` would have made for it.
    """
    media_type = guess_type(path)[0] or FILE_RESPONSE_FALLBACK_MEDIA_TYPE
    if media_type.startswith("text/"):
        media_type += "; charset=utf-8"
    return media_type


def register(app: FastAPI):
    ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
    if not os.path.isdir(ui_dir):
        raise RuntimeError(f"directory '{ui_dir}' does not exist")

    for name in ["css", "js", "static"]:
        # follow_symlink stays off, unlike the /static mount fastapi_cdn_host
        # would have made. install.sh materialises this tree as real files, so
        # resolving links buys nothing here and would let one placed under
        # ui/ serve a file from outside it.
        app.mount(
            f"/{name}",
            PrecompressedStaticFiles(directory=os.path.join(ui_dir, name)),
            name=name,
        )

    @app.get("/", include_in_schema=False)
    async def index():
        # Revalidate the entry point on every load. Its own name is fixed
        # while every asset it points at is content-addressed, so a copy
        # reused without asking is what pins a browser to the previous build's
        # bundles after an upgrade. no-cache still permits storing the body —
        # the check settles with a 304, costing a round trip rather than a
        # re-download.
        return FileResponse(
            os.path.join(ui_dir, "index.html"),
            headers={"Cache-Control": CACHE_REVALIDATE},
        )
