import pytest
from unittest.mock import MagicMock, patch
from gpustack.gateway.plugins import (
    HigressPlugin,
    get_plugin_url_prefix,
    get_plugin_url_with_name_and_version,
    supported_plugins,
    http_path_prefix,
)
from gpustack.gateway import ext_auth_plugin


def make_cfg(plugin_server_url: str):
    cfg = MagicMock()
    cfg.gateway_plugin_server_url = plugin_server_url
    return cfg


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
        cfg = make_cfg("http://127.0.0.1:8080")
        url = get_plugin_url_with_name_and_version("ai-proxy", "2.0.0", cfg)
        assert (
            url
            == f"http://127.0.0.1:8080/{http_path_prefix}/ai-proxy/2.0.0/plugin.wasm"
        )

    def test_unknown_plugin_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version("nonexistent-plugin", "1.0.0")

    def test_wrong_version_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version("ai-proxy", "9.9.9")


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
            "ai-proxy",
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
