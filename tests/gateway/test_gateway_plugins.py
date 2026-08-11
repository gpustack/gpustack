import pytest
from unittest.mock import MagicMock, patch
from gpustack.gateway.plugins import (
    HigressPlugin,
    get_plugin_url_prefix,
    get_plugin_url_with_name_and_version,
    supported_plugins,
    http_path_prefix,
)
from gpustack.config.config import GatewayPluginEntry
from gpustack.gateway import ai_statistics_plugin, ext_auth_plugin, transformer_plugin


def make_cfg(plugin_server_url: str, gateway_plugin=None):
    cfg = MagicMock()
    cfg.gateway_plugin_server_url = plugin_server_url
    # Real entries, not raw dicts: a live Config validates this section into
    # GatewayPluginEntry, so a dict here would exercise a shape production
    # never sees -- and would skip the extra="forbid" on url / sha256.
    cfg.gateway_plugin = {
        name: GatewayPluginEntry.model_validate(entry)
        for name, entry in (gateway_plugin or {}).items()
    }
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
        cfg = make_cfg("http://127.0.0.1")
        cfg.get_derived_gateway_token.return_value = "token"
        cfg.get_namespace.return_value = namespace
        cfg.gateway_namespace = gateway_namespace
        with patch(
            "gpustack.gateway.get_gpustack_higress_registry", return_value=registry
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


class TestGatewayPluginOverrides:
    """``gateway_plugin.<manifest name>`` in config.yaml.

    Two properties matter. The key is the plugin's *manifest* name, not the
    WasmPlugin resource name it is deployed as -- five of the eight differ, and
    keying by the wrong one matches nothing at all, silently. And ``config`` is
    validated against a model the plugin module owns, so a field that plugin
    does not declare is a startup error rather than a value quietly ignored.
    """

    def test_module_source_is_overridable(self):
        cfg = make_cfg(
            "http://127.0.0.1",
            {"transformer": {"url": "oci://example.com/transformer:9", "sha256": "ab"}},
        )

        _, spec = transformer_plugin(cfg=cfg)

        assert spec.url == "oci://example.com/transformer:9"
        assert spec.sha256 == "ab"

    def test_the_manifest_url_is_used_when_nothing_is_overridden(self):
        cfg = make_cfg("http://127.0.0.1")

        _, spec = transformer_plugin(cfg=cfg)

        assert spec.url.endswith("/transformer/2.0.0/plugin.wasm")
        assert spec.sha256 is None

    def test_the_key_is_the_manifest_name_not_the_resource_name(self):
        # transformer is deployed as gpustack-header-transformer; keying by the
        # latter must not take effect, or an operator would silently get the
        # manifest URL while believing otherwise.
        cfg = make_cfg(
            "http://127.0.0.1",
            {"gpustack-header-transformer": {"url": "oci://example.com/nope:1"}},
        )

        _, spec = transformer_plugin(cfg=cfg)

        assert spec.url.endswith("/transformer/2.0.0/plugin.wasm")

    def test_plugin_config_reaches_the_default_config(self):
        cfg = make_cfg(
            "http://127.0.0.1",
            {
                "ai-statistics": {
                    "config": {"enable_content_types": ["application/json"]}
                }
            },
        )

        _, spec = ai_statistics_plugin(cfg=cfg)

        assert spec.defaultConfig["enable_content_types"] == ["application/json"]

    def test_plugin_config_defaults_apply_when_absent(self):
        _, spec = ai_statistics_plugin(cfg=make_cfg("http://127.0.0.1"))

        assert spec.defaultConfig["enable_content_types"] == [
            "application/json",
            "text/event-stream",
        ]

    def test_a_field_the_plugin_does_not_declare_is_refused(self):
        # Silently ignoring it would leave the plugin on a default the operator
        # believes they changed. ``attributes`` in particular carries the
        # consumer identity into the access log.
        cfg = make_cfg(
            "http://127.0.0.1", {"ai-statistics": {"config": {"attributes": []}}}
        )

        with pytest.raises(ValueError, match="Invalid gateway_plugin"):
            ai_statistics_plugin(cfg=cfg)
