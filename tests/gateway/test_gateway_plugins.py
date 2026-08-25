import logging
import pytest
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from unittest.mock import MagicMock, patch
from gpustack.gateway.plugins import (
    HigressPlugin,
    get_plugin_url_prefix,
    get_plugin_url_with_name_and_version,
    published_plugin_urls,
    supported_plugins,
    http_path_prefix,
    verify_published_plugin_modules,
)
from gpustack.gateway import ext_auth_plugin
from gpustack.schemas.config import GatewayModeEnum


def make_cfg(plugin_server_url: str, gateway_mode=GatewayModeEnum.external):
    cfg = MagicMock()
    cfg.gateway_plugin_server_url = plugin_server_url
    # Explicit, because the mode decides whether modules are fetched over HTTP
    # or read off Envoy's own filesystem. A MagicMock here would compare
    # unequal to every mode and silently pick HTTP.
    cfg.gateway_mode = gateway_mode
    return cfg


def _a_plugin():
    return supported_plugins[0]


class TestGetPluginUrlPrefix:
    def test_no_cfg_returns_localhost(self):
        assert get_plugin_url_prefix() == f"http://127.0.0.1/{http_path_prefix}"

    def test_cfg_none_returns_localhost(self):
        assert get_plugin_url_prefix(None) == f"http://127.0.0.1/{http_path_prefix}"

    def test_cfg_with_url(self):
        cfg = make_cfg("http://192.168.1.1:8080")
        assert (
            get_plugin_url_prefix(cfg) == f"http://192.168.1.1:8080/{http_path_prefix}"
        )

    def test_cfg_with_https_url(self):
        cfg = make_cfg("https://example.com")
        assert get_plugin_url_prefix(cfg) == f"https://example.com/{http_path_prefix}"


class TestHigressPluginGetPath:
    def test_path_without_cfg(self):
        plugin = HigressPlugin(name="ai-proxy", version="2.0.0")
        assert (
            plugin.get_path()
            == f"http://127.0.0.1/{http_path_prefix}/ai-proxy/2.0.0/plugin.wasm"
        )

    def test_path_with_cfg(self):
        plugin = HigressPlugin(name="ai-proxy", version="2.0.0")
        cfg = make_cfg("http://10.0.0.1:9000")
        assert (
            plugin.get_path(cfg)
            == f"http://10.0.0.1:9000/{http_path_prefix}/ai-proxy/2.0.0/plugin.wasm"
        )

    def test_path_uses_forward_slash(self):
        plugin = HigressPlugin(name="model-router", version="2.0.0")
        path = plugin.get_path()
        assert "\\" not in path

    def test_name_with_special_chars_is_encoded(self):
        plugin = HigressPlugin(name="plugin name", version="1.0.0")
        path = plugin.get_path()
        assert "plugin%20name" in path
        assert " " not in path

    def test_version_with_special_chars_is_encoded(self):
        plugin = HigressPlugin(name="my-plugin", version="1.0.0+build")
        path = plugin.get_path()
        assert "1.0.0%2Bbuild" in path


class TestGetPluginUrlWithNameAndVersion:
    def test_known_plugin(self):
        # Resolve from the manifest so version bumps don't break this test.
        known = supported_plugins[0]
        cfg = make_cfg("http://127.0.0.1:8080")
        url = get_plugin_url_with_name_and_version(known.name, known.version, cfg)
        assert (
            url
            == f"http://127.0.0.1:8080/{http_path_prefix}/{known.name}/{known.version}/plugin.wasm"
        )

    def test_unknown_plugin_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version("nonexistent-plugin", "1.0.0")

    def test_wrong_version_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version(supported_plugins[0].name, "9.9.9")


class TestSupportedPlugins:
    def test_plugins_loaded(self):
        assert len(supported_plugins) > 0

    def test_all_plugins_have_name_and_version(self):
        for plugin in supported_plugins:
            assert plugin.name
            assert plugin.version

    def test_known_plugins_present(self):
        names = {p.name for p in supported_plugins}
        for expected in [
            "gpustack-ai-proxy",
            "ai-statistics",
            "ext-auth",
            "gpustack-generic-proxy-router",
            "gpustack-model-mapper",
            "transformer",
            "gpustack-token-usage",
            "gpustack-set-header-pre-route",
        ]:
            assert expected in names, f"{expected} not found in supported_plugins"


class TestExtAuthPlugin:
    def _build(self, namespace, gateway_namespace):
        registry = MagicMock()
        registry.get_service_name.return_value = "gpustack.static"
        registry.port = 80
        cfg = MagicMock()
        cfg.get_derived_gateway_token.return_value = "token"
        cfg.get_namespace.return_value = namespace
        cfg.gateway_namespace = gateway_namespace
        with (
            patch(
                "gpustack.gateway.get_gpustack_higress_registry", return_value=registry
            ),
            patch(
                "gpustack.gateway.get_plugin_url_with_name_and_version",
                return_value="http://127.0.0.1/wasm/ext-auth/2.0.0/plugin.wasm",
            ),
        ):
            return ext_auth_plugin(cfg=cfg)

    def _rule(self, spec):
        rules = spec.defaultConfig["_rules_"]
        assert len(rules) == 1
        return rules[0]

    def test_scopes_to_model_route_prefix(self):
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        # auth is applied via defaultConfig _rules_ matched by route prefix,
        # not a global catch-all, and there are no separate matchRules.
        assert spec.defaultConfigDisable is False
        assert not spec.matchRules
        rule = self._rule(spec)
        assert rule["_match_route_prefix_"] == ["default/ai-route-route-"]
        assert rule["match_type"] == "blacklist"

    def test_no_namespace_prefix_when_same_namespace(self):
        _, spec = self._build(
            namespace="higress-system", gateway_namespace="higress-system"
        )
        rule = self._rule(spec)
        assert rule["_match_route_prefix_"] == ["ai-route-route-"]


class TestLocalPluginModules:
    """Embedded reads modules off Envoy's own filesystem instead of fetching
    them from this server -- the one mode where the two share one."""

    def test_embedded_resolves_to_a_file_url(self):
        plugin = _a_plugin()
        cfg = make_cfg("http://127.0.0.1:30080", GatewayModeEnum.embedded)
        url = get_plugin_url_with_name_and_version(plugin.name, plugin.version, cfg)
        assert url.startswith("file:///")
        assert url.endswith(f"/{plugin.name}/{plugin.version}/plugin.wasm")

    def test_every_module_resolves_to_a_file_that_exists(self):
        cfg = make_cfg("http://127.0.0.1:30080", GatewayModeEnum.embedded)
        for plugin in supported_plugins:
            url = get_plugin_url_with_name_and_version(plugin.name, plugin.version, cfg)
            assert Path(url2pathname(urlparse(url).path)).is_file()

    @pytest.mark.parametrize(
        "mode", [GatewayModeEnum.external, GatewayModeEnum.incluster]
    )
    def test_other_modes_keep_fetching_over_http(self, mode):
        # Envoy is a different pod there, so a local path would resolve to
        # nothing on its side.
        plugin = _a_plugin()
        cfg = make_cfg("http://192.168.1.1:8080", mode)
        url = get_plugin_url_with_name_and_version(plugin.name, plugin.version, cfg)
        assert url.startswith("http://192.168.1.1:8080/")

    def test_no_cfg_keeps_fetching_over_http(self):
        plugin = _a_plugin()
        url = get_plugin_url_with_name_and_version(plugin.name, plugin.version)
        assert url.startswith("http://")

    def test_missing_module_falls_back_to_http_not_a_dead_path(self):
        plugin = _a_plugin()
        cfg = make_cfg("http://127.0.0.1:30080", GatewayModeEnum.embedded)
        with patch("gpustack.gateway.plugins.Path.is_file", return_value=False):
            url = get_plugin_url_with_name_and_version(plugin.name, plugin.version, cfg)
        assert url.startswith("http://127.0.0.1:30080/")


class TestPublishedPluginUrls:
    def test_pairs_every_manifest_plugin_with_the_url_in_its_cr(self):
        cfg = make_cfg("http://127.0.0.1:30080")
        pairs = published_plugin_urls(cfg)
        assert len(pairs) == len(supported_plugins)
        for name, url in pairs:
            plugin = next(p for p in supported_plugins if p.name == name)
            assert url == get_plugin_url_with_name_and_version(
                plugin.name, plugin.version, cfg
            )


class TestVerifyPublishedPluginModules:
    """A module Envoy cannot load takes the listener down silently, so the
    check has to distinguish "broken" from "cannot be checked from here"."""

    @pytest.mark.asyncio
    async def test_local_modules_pass_without_any_fetch(self, caplog):
        # The embedded default. aiohttp cannot open a file:// URL, so treating
        # these as HTTP would report every module broken on every start.
        cfg = make_cfg("http://127.0.0.1:30080", GatewayModeEnum.embedded)
        with caplog.at_level(logging.INFO, logger="gpustack.gateway.plugins"):
            assert await verify_published_plugin_modules(cfg) is True
        assert f"{_a_plugin().name}=local" in caplog.text

    @pytest.mark.asyncio
    async def test_a_missing_local_module_is_reported_broken(self, tmp_path, caplog):
        # Fed in directly: every shipped module is present, and this branch has
        # no per-plugin URL override to point one somewhere else.
        missing = (tmp_path / "gone.wasm").as_uri()
        cfg = make_cfg("http://127.0.0.1:30080", GatewayModeEnum.embedded)
        with patch(
            "gpustack.gateway.plugins.published_plugin_urls",
            return_value=[("transformer", missing)],
        ):
            with caplog.at_level(logging.INFO, logger="gpustack.gateway.plugins"):
                assert await verify_published_plugin_modules(cfg) is False
        assert "transformer=UNREADABLE" in caplog.text
        assert "will not be bound" in caplog.text

    @pytest.mark.asyncio
    async def test_an_unreachable_http_module_is_reported_broken(self, caplog):
        # Port 1 on loopback: refused fast, no network needed.
        cfg = make_cfg("http://127.0.0.1:1")
        with caplog.at_level(logging.INFO, logger="gpustack.gateway.plugins"):
            assert await verify_published_plugin_modules(cfg) is False
        assert "=FAILED" in caplog.text
