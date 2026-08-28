import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import url2pathname
from importlib.resources import files
from typing import Any, Dict, Optional, List, Tuple

import aiohttp

from gpustack.config.config import Config, GatewayPluginEntry
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


def get_plugin_url_with_name_and_version(
    name: str, version: Optional[str] = None, cfg: Optional[Config] = None
) -> str:
    """The module URL for one plugin, resolved from the bundled plugin package.

    ``version`` is normally omitted, and callers here do omit it. The manifest
    carries one version per plugin -- the build actually shipped in the package,
    read as ``latest`` into ``supported_plugins`` -- so a version passed in can
    only assert what the package already decided; it cannot select an older
    build, because no older build is present to select. Leaving it out is what
    keeps a ``gpustack-higress-plugins`` bump from having to touch every call
    site, at the cost of that assertion: a plugin whose config schema changed
    across the bump is then caught by the gateway rejecting the config rather
    than by a startup error here.

    Pass a version only where following the package silently is the worse
    failure.
    """
    return resolve_plugin(name, version).get_path(cfg)


def resolve_plugin(name: str, version: Optional[str] = None) -> HigressPlugin:
    target = next(
        (
            p
            for p in supported_plugins
            if p.name == name and (version is None or p.version == version)
        ),
        None,
    )
    if target is None:
        wanted = name if version is None else f"{name} with version {version}"
        raise ValueError(f"Plugin {wanted} is not supported.")
    return target


def get_plugin_url_prefix(cfg: Optional[Config] = None) -> str:
    """Where Envoy fetches plugin modules from.

    This setting decides something the code cannot: whether a gateway pod can
    still load its modules while the GPUStack server is down. Unset, the server
    serves them itself, so a pod cold-starting during an outage gets nothing --
    and ``failStrategy: FAIL_OPEN`` turns a module that fails to load into
    inference routes served with no authentication at all. The chart points it
    at a separate deployment, which is what takes module distribution out of
    the server's availability domain; embedded mode does not, and relies on
    Envoy having cached the module for the life of the pod.

    Deliberately no ``sha256`` on what comes back, though the spec accepts one
    and an operator can supply it. The URL already carries the plugin version
    and a version's bytes do not change, so a digest adds no discrimination the
    path does not; and tampering with this fetch requires the same position on
    the network as tampering with the authorization calls to ``/token-auth``,
    which is the stronger attack of the two. It would harden a link that is not
    the weakest one.
    """
    base_url = "http://127.0.0.1"
    if cfg is not None and cfg.gateway_plugin_server_url:
        base_url = cfg.gateway_plugin_server_url
    return f"{base_url}/{http_path_prefix}"


def get_local_plugin_url(name: str, version: Optional[str] = None) -> Optional[str]:
    """A ``file://`` URL for a module Envoy can read off its own filesystem,
    or None when it cannot.

    Only meaningful where Envoy and this process share a filesystem, which is
    the embedded gateway and nothing else -- see
    :func:`use_local_plugin_modules`. There the module bytes are already
    present, shipped inside ``gpustack_higress_plugins``, so serving them over
    HTTP only to fetch them back over loopback buys nothing and costs the one
    thing that actually broke: a dependency on this server being able to
    answer at the moment Envoy warms its listeners.

    ``Path.as_uri`` rather than an f-string, so a path needing escaping
    produces a URL Go's ``url.Parse`` reads back as the same path.

    Returns None if the file is absent, which lets the caller fall back to
    HTTP instead of publishing a CR Envoy can only fail on.
    """
    path = resolve_plugin(name, version).get_local_path()
    if not path.is_file():
        return None
    return path.resolve().as_uri()


def use_local_plugin_modules(cfg: Optional[Config] = None) -> bool:
    """Whether Envoy reads the modules from disk instead of fetching them.

    Embedded only, and that is not a policy choice -- it is the only mode
    where Envoy runs in the same container as this process, so it is the only
    mode where a local path means the same thing on both sides. In
    ``external`` and ``incluster`` the gateway is a different pod with a
    different filesystem and a ``file://`` URL would resolve to nothing.
    """
    return cfg is not None and cfg.gateway_mode == GatewayModeEnum.embedded


def plugin_entry(name: str, cfg: Optional[Config]) -> Optional[GatewayPluginEntry]:
    """The operator's overrides for one plugin, keyed by its *manifest* name.

    That is the name in ``supported_plugins`` and in the module URL, which for
    five of the eight plugins is not the name of the WasmPlugin resource they
    end up as -- ``gpustack-ext-auth`` is deployed as ``gpustack-llm-ext-auth``,
    ``transformer`` as ``gpustack-header-transformer``, and so on. Keying by
    the resource name instead would silently match nothing.
    """
    if cfg is None:
        return None
    return (cfg.gateway_plugin or {}).get(name)


def plugin_spec_overrides(
    name: str, version: Optional[str] = None, cfg: Optional[Config] = None
) -> Dict[str, Any]:
    """The distribution half of a ``WasmPluginSpec``: where the module comes
    from and how it is verified.

    Splat into the spec (``**plugin_spec_overrides(...)``) instead of setting
    ``url=`` directly, so every plugin honours ``gateway_plugin.<name>.url``
    the same way. Overriding the URL without the matching ``sha256`` is
    accepted -- Envoy simply does not verify the module then -- but the pair is
    what an operator pointing at a private registry normally wants.

    Three sources, in descending order of how specific they are: an operator's
    explicit URL, the module on Envoy's own filesystem where there is one, and
    otherwise this server over HTTP.
    """
    entry = plugin_entry(name, cfg)
    local_url = (
        get_local_plugin_url(name, version) if use_local_plugin_modules(cfg) else None
    )
    if entry is not None and entry.url:
        spec: Dict[str, Any] = {"url": entry.url}
    elif local_url is not None:
        spec = {"url": local_url}
    else:
        spec = {"url": get_plugin_url_with_name_and_version(name, version, cfg)}
    if entry is not None and entry.sha256:
        spec["sha256"] = entry.sha256
    if entry is not None and entry.image_pull_policy:
        spec["imagePullPolicy"] = entry.image_pull_policy
    return spec


def published_plugin_urls(cfg: Optional[Config] = None) -> List[Tuple[str, str]]:
    """Every module URL the WasmPlugin CRs carry, as ``(plugin name, url)``.

    Derived from ``cfg`` rather than recorded while publishing: the same
    ``cfg`` produces the same URLs, so there is nothing to keep in sync and no
    state to carry between ``initialize_gateway`` and whoever wants to check
    its work.

    Covers every plugin in the manifest, including any the current
    configuration does not deploy. One extra URL checked is cheap; keeping
    this in step with the deployment list in ``initialize_gateway`` -- which
    varies by server role -- is not.
    """
    return [
        (plugin.name, plugin_spec_overrides(plugin.name, cfg=cfg).get("url", ""))
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
    is as much proof as a fetch would be. HTTP URLs are fetched with a
    one-byte ``Range``, since the modules run to megabytes. Anything else is
    reported unverified rather than broken: a plugin pointed at an OCI
    registry cannot be reached from here and does not depend on this server.

    An HTTP pass proves the URL is servable from *here*. Where the gateway is
    a different pod it leaves routing and firewalling between there and this
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
