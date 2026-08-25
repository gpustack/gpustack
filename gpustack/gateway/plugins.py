import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import url2pathname
from importlib.resources import files
from typing import Optional, List, Tuple

import aiohttp

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack_higress_plugins.server import router as higress_plugins_router

logger = logging.getLogger(__name__)

# Reuse the same prefix as the plugin server router
http_path_prefix = higress_plugins_router.prefix.removeprefix("/")

# Where the plugin package keeps the module bytes it also serves over HTTP.
_plugins_dir = Path(str(files("gpustack_higress_plugins").joinpath("plugins")))


@dataclass
class HigressPlugin:
    name: str
    version: str

    def get_path(self, cfg: Optional[Config] = None) -> str:
        path = "/".join(
            [quote(self.name, safe=""), quote(self.version, safe=""), "plugin.wasm"]
        )
        return f"{get_plugin_url_prefix(cfg)}/{path}"

    def get_local_path(self) -> Path:
        """Where this module sits on disk, served or not."""
        return _plugins_dir / self.name / self.version / "plugin.wasm"


def _load_plugins_from_manifest() -> List[HigressPlugin]:
    manifest_text = (
        files("gpustack_higress_plugins")
        .joinpath("manifest.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_text)
    return [
        HigressPlugin(name=name, version=info["latest"])
        for name, info in manifest["plugins"].items()
    ]


supported_plugins: List[HigressPlugin] = _load_plugins_from_manifest()


def resolve_plugin(name: str, version: str) -> HigressPlugin:
    target = next(
        (p for p in supported_plugins if p.name == name and p.version == version), None
    )
    if target is None:
        raise ValueError(f"Plugin {name} with version {version} is not supported.")
    return target


def use_local_plugin_modules(cfg: Optional[Config] = None) -> bool:
    """Whether Envoy reads the modules from disk instead of fetching them.

    Embedded only, and that is not a policy choice -- it is the only mode where
    Envoy runs in the same container as this process, so it is the only mode
    where a local path means the same thing on both sides. In ``external`` and
    ``incluster`` the gateway is a different pod with a different filesystem
    and a ``file://`` URL would resolve to nothing.
    """
    return cfg is not None and cfg.gateway_mode == GatewayModeEnum.embedded


def get_plugin_url_with_name_and_version(
    name: str, version: str, cfg: Optional[Config] = None
) -> str:
    """Where Envoy loads one plugin's module from.

    Two shapes, decided by :func:`use_local_plugin_modules`. In the embedded
    gateway it is a ``file://`` URL: the module bytes are already present,
    shipped inside ``gpustack_higress_plugins`` on the filesystem Envoy reads,
    so serving them over HTTP only to fetch them back over loopback buys
    nothing and costs the one thing that actually broke -- a dependency on this
    server being able to answer at the moment Envoy warms its listeners.
    Everywhere else the gateway is a different pod, and the module has to come
    from this server over HTTP.

    ``Path.as_uri`` rather than an f-string, so a path needing escaping
    produces a URL Go's ``url.Parse`` reads back as the same path.

    Falls back to HTTP when the local file is absent, rather than publishing a
    path Envoy can only fail on. That fallback is reported rather than relied
    on: :func:`verify_published_plugin_modules` is what says a module could not
    be found.
    """
    target = resolve_plugin(name, version)
    if use_local_plugin_modules(cfg):
        local_path = target.get_local_path()
        if local_path.is_file():
            return local_path.resolve().as_uri()
    return target.get_path(cfg)


def get_plugin_url_prefix(cfg: Optional[Config] = None) -> str:
    base_url = "http://127.0.0.1"
    if cfg is not None and cfg.gateway_plugin_server_url:
        base_url = cfg.gateway_plugin_server_url
    return f"{base_url}/{http_path_prefix}"


def published_plugin_urls(cfg: Optional[Config] = None) -> List[Tuple[str, str]]:
    """Every module URL the WasmPlugin CRs carry, as ``(plugin name, url)``.

    Derived from ``cfg`` rather than recorded while publishing: the same ``cfg``
    produces the same URLs, so there is nothing to keep in sync and no state to
    carry between ``initialize_gateway`` and whoever wants to check its work.

    Covers every plugin in the manifest, including any the current
    configuration does not deploy. One extra URL checked is cheap; keeping this
    in step with the deployment list in ``initialize_gateway`` -- which varies
    by server role -- is not.
    """
    return [
        (
            plugin.name,
            get_plugin_url_with_name_and_version(plugin.name, plugin.version, cfg),
        )
        for plugin in supported_plugins
    ]


async def verify_published_plugin_modules(
    cfg: Optional[Config] = None, timeout: float = 10.0
) -> bool:
    """Check that every published module URL resolves, and log what it found.

    Worth doing because a module Envoy cannot load is not a degradation: the
    filter is discovered over ECDS with ``initial_fetch_timeout: 0`` and no
    default config, so an unresolved one leaves the listener warming and its
    port unbound, and ``failStrategy: FAIL_OPEN`` does not cover it. Silence
    from the gateway is the only other symptom.

    ``file://`` URLs are stat'd -- under the embedded gateway that is every
    module, on the filesystem Envoy itself reads, so an existing readable file
    is as much proof as a fetch would be. HTTP URLs are fetched with a one-byte
    ``Range``, since the modules run to megabytes.

    An HTTP pass proves the URL is servable from *here*. Where the gateway is a
    different pod it leaves routing and firewalling between there and this
    address unproven, which is why the URL is logged verbatim -- enough for an
    operator to curl it from the gateway itself.

    Returns True when nothing was found broken.
    """
    plugin_urls = published_plugin_urls(cfg)
    if not plugin_urls:
        return True

    def probe_local(name: str, url: str) -> Tuple[bool, str]:
        path = Path(url2pathname(urlparse(url).path))
        try:
            return True, f"{name}=local {path.stat().st_size} bytes"
        except OSError as e:
            return False, f"{name}=UNREADABLE {path}: {type(e).__name__}: {e}"

    async def probe_http(
        session: aiohttp.ClientSession, name: str, url: str
    ) -> Tuple[bool, str]:
        started = time.monotonic()
        try:
            async with session.get(url, headers={"Range": "bytes=0-0"}) as resp:
                elapsed = time.monotonic() - started
                length = resp.headers.get("Content-Range") or resp.headers.get(
                    "Content-Length", "?"
                )
                return resp.status in (200, 206), (
                    f"{name}={resp.status} ({length}) in {elapsed:.2f}s"
                )
        except Exception as e:
            elapsed = time.monotonic() - started
            return (
                False,
                f"{name}=FAILED after {elapsed:.2f}s: {type(e).__name__}: {e}",
            )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:

        async def probe(name: str, url: str) -> Tuple[bool, str]:
            if url.startswith(("http://", "https://")):
                return await probe_http(session, name, url)
            if url.startswith("file://"):
                return probe_local(name, url)
            return True, f"{name}=unverified ({urlparse(url).scheme or 'no'} scheme)"

        results = await asyncio.gather(*(probe(name, url) for name, url in plugin_urls))

    broken = [text for ok, text in results if not ok]
    message = "Gateway plugin modules (%d): %s" % (
        len(plugin_urls),
        "; ".join(text for _, text in results),
    )
    if broken:
        logger.error(
            "%s -- Envoy cannot load these; the gateway listener will stay in "
            "warming and its port will not be bound.",
            message,
        )
        return False
    logger.info(message)
    return True
