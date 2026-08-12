import os
import posixpath
import re
import stat
from mimetypes import guess_type
from typing import Optional

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

    Content negotiation here is per-file, so ``Vary: Accept-Encoding`` goes on
    both answers: without it a shared cache could hand a gzip body to a client
    that never asked for one.

    Both answers also carry a ``Cache-Control`` chosen by ``cache_control_for``.
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
        response.headers.add_vary_header("Accept-Encoding")
        self._set_cache_control(response, path)
        return response

    @staticmethod
    def _set_cache_control(response: Response, path: str) -> None:
        """Apply the freshness policy, including on a 304.

        A Not Modified response is how a client refreshes an expired entry, so
        omitting the header there would leave the refreshed copy with no
        lifetime and force another revalidation on the very next load.
        """
        response.headers["cache-control"] = cache_control_for(path)

    async def _precompressed_response(
        self, path: str, scope: Scope
    ) -> Optional[Response]:
        """The ``.gz`` sibling of ``path``, or None to fall back to the plain file."""
        try:
            full_path, stat_result = await anyio.to_thread.run_sync(
                self.lookup_path, f"{path}.gz"
            )
        except (OSError, HTTPException):
            # Covers the unreadable, the too-long and the merely absent. The
            # installed Starlette reports a miss by returning ``(\"\", None)``,
            # but the dependency range spans a major version of it, so a build
            # that signals the miss by raising must land here too rather than
            # turning every asset without a .gz sibling into a 404. Either way
            # a .gz we cannot stat is not worth failing the request over — the
            # plain asset is a complete answer on its own.
            return None

        if stat_result is None or not stat.S_ISREG(stat_result.st_mode):
            return None

        response = self.file_response(full_path, stat_result, scope)
        response.headers.add_vary_header("Accept-Encoding")
        # Keyed on the requested path, not the .gz we resolved to: the two
        # encodings are the same resource and must expire together.
        self._set_cache_control(response, path)
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
