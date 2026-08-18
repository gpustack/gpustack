import pytest
import logging
from unittest.mock import MagicMock, patch
from gpustack.gateway.plugins import (
    HigressPlugin,
    get_plugin_url_prefix,
    get_plugin_url_with_name_and_version,
    supported_plugins,
    http_path_prefix,
)
from gpustack.config.config import GatewayPluginEntry
from gpustack.api.auth import (
    GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
    GATEWAY_ASSERTED_KEY_REF_HEADER,
    GATEWAY_AUTH_TOKEN_HEADER,
)
from gpustack.gateway import (
    ai_statistics_override,
    ai_statistics_plugin,
    ext_auth_plugin,
    transformer_plugin,
)
from gpustack.gateway.ext_auth import ext_auth_override


def _manifest_suffix(name: str) -> str:
    """The tail of a module URL for the version the plugin package ships, so a
    bump does not need this file edited."""
    version = next(p.version for p in supported_plugins if p.name == name)
    return f"/{name}/{version}/plugin.wasm"


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

    def test_the_version_is_optional_and_resolves_to_the_shipped_one(self):
        """What every caller in gpustack.gateway does. The manifest carries one
        version per plugin, so a pinned version could only assert what the
        package already shipped -- and pinning meant editing every call site on
        a gpustack-higress-plugins bump."""
        known = supported_plugins[0]
        cfg = make_cfg("http://127.0.0.1:8080")

        assert get_plugin_url_with_name_and_version(
            known.name, cfg=cfg
        ) == get_plugin_url_with_name_and_version(known.name, known.version, cfg)

    def test_unknown_plugin_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version("nonexistent-plugin", "1.0.0")

    def test_unknown_plugin_raises_without_a_version_too(self):
        # The name is still checked when the version is left out; a plugin the
        # package does not carry must not resolve to some other plugin's module.
        with pytest.raises(ValueError, match="not supported"):
            get_plugin_url_with_name_and_version("nonexistent-plugin")

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
            "gpustack-ext-auth",
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
        # The plugin package carries no entry for this plugin, so its URL can
        # only come from config -- see test_a_missing_module_url_is_refused.
        cfg.gateway_plugin = {
            "gpustack-ext-auth": GatewayPluginEntry.model_validate(
                {"url": "oci://example.com/ext-auth:1", "sha256": "ab"}
            )
        }
        cfg.get_derived_gateway_token.return_value = "token"
        cfg.get_derived_auth_cache_key.return_value = "auth-cache-key"
        cfg.get_namespace.return_value = namespace
        cfg.gateway_namespace = gateway_namespace
        with patch(
            "gpustack.gateway.get_gpustack_higress_registry", return_value=registry
        ):
            return ext_auth_plugin(cfg=cfg)

    def test_scopes_to_model_route_names(self):
        """The plugin's own gate, not a rule. The SDK's matcher falls back to
        the global config when nothing matches, so "no rule matched" cannot
        mean "not ours" -- without this every route of every other tenant on a
        shared gateway would have this plugin calling gpustack's /token-auth.

        Anchored explicitly: the plugin does not anchor for you, and an
        unanchored prefix matches a route that merely contains it."""
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        assert spec.defaultConfigDisable is False
        assert spec.defaultConfig["route_match_regexes"] == [
            r"^default/ai\-route\-route\-"
        ]
        # No hand-written _rules_ anywhere: it cannot coexist with matchRules,
        # which the controller overwrites it with as soon as they are non-empty.
        assert "_rules_" not in spec.defaultConfig

    def test_no_namespace_prefix_when_same_namespace(self):
        _, spec = self._build(
            namespace="higress-system", gateway_namespace="higress-system"
        )
        assert spec.defaultConfig["route_match_regexes"] == [r"^ai\-route\-route\-"]

    def test_access_policy_never_appears_globally(self):
        """In defaultConfig it would declare every route public -- the exact
        opposite of a per-route grant."""
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        assert "access_policy" not in spec.defaultConfig

    def test_ships_the_dedicated_signing_key_not_the_jwt_secret(self):
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        auth_cache = spec.defaultConfig["auth_cache"]
        assert auth_cache["signing_key"] == "auth-cache-key"
        assert auth_cache["header"] == "x-gpustack-auth-cache"

    def test_local_auth_starts_empty_and_enabled(self):
        # The tables are the reconciler's to fill; the static base only
        # declares the switch.
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        assert spec.defaultConfig["local_auth"] == {
            "enabled": True,
            "keys": {},
            "refs": {},
        }

    def test_the_module_source_comes_from_config(self):
        _, spec = self._build(namespace="default", gateway_namespace="higress-system")
        assert spec.url == "oci://example.com/ext-auth:1"
        assert spec.sha256 == "ab"

    def test_a_module_the_package_does_not_carry_is_refused(self, monkeypatch):
        """Nothing is defaulted when the plugin package carries no ext-auth
        module. The alternative failure is Envoy unable to pull the module: the
        filter then does not load, and under FAIL_OPEN every inference route is
        served unauthenticated, silently. Refusing to start is the loud version
        of the same problem, and the message names the config key that
        overrides it."""
        monkeypatch.setattr(
            "gpustack.gateway.plugins.supported_plugins",
            [p for p in supported_plugins if p.name != "gpustack-ext-auth"],
        )
        registry = MagicMock()
        registry.get_service_name.return_value = "gpustack.static"
        registry.port = 80
        cfg = make_cfg("http://127.0.0.1")
        cfg.get_namespace.return_value = "default"
        cfg.gateway_namespace = "higress-system"

        with patch(
            "gpustack.gateway.get_gpustack_higress_registry", return_value=registry
        ):
            with pytest.raises(
                ValueError, match="gateway_plugin.gpustack-ext-auth.url"
            ):
                ext_auth_plugin(cfg=cfg)


class TestTransformerPlugin:
    """The transformer is the only thing standing between a client and a
    self-asserted identity: it strips the gateway-only headers at priority 810,
    ahead of ext-auth's injection at 360 in the same AUTHN phase. With
    authentication moved to the edge there is no second gate behind it, so the
    remove rule is load-bearing rather than hygienic."""

    def _rules(self):
        cfg = MagicMock()
        cfg.gateway_plugin = {}
        cfg.gateway_plugin_server_url = "http://127.0.0.1"
        _, spec = transformer_plugin(cfg=cfg)
        return spec

    def test_strips_a_client_supplied_consumer(self):
        """ext-auth removes this too, but only on routes it owns -- its route
        gate returns first for everything else. This plugin has no gate, so it
        covers what ext-auth declines: the control-plane mirror ingress, and
        other tenants on a shared gateway. ai-statistics and token-usage read
        the header globally and write it into the access log and the usage
        records, so leaving it there lets a client name itself on those routes
        and be believed."""
        spec = self._rules()

        removed = {
            header["key"]
            for rule in spec.defaultConfig["reqRules"]
            if rule["operate"] == "remove"
            for header in rule["headers"]
        }

        assert "x-mse-consumer" in removed

    def test_strips_the_gateway_only_headers(self):
        spec = self._rules()
        removed = {
            header["key"]
            for rule in spec.defaultConfig["reqRules"]
            if rule["operate"] == "remove"
            for header in rule["headers"]
        }
        assert {
            GATEWAY_AUTH_TOKEN_HEADER,
            GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
            GATEWAY_ASSERTED_KEY_REF_HEADER,
        } <= removed

    def test_runs_before_ext_auth(self):
        # Higress runs higher priority first within a phase, so removal must
        # outrank ext-auth's injection.
        spec = self._rules()
        assert spec.phase == "AUTHN"
        assert spec.priority > 360


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

        assert spec.url.endswith(_manifest_suffix("transformer"))
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

        assert spec.url.endswith(_manifest_suffix("transformer"))

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


class TestExtAuthPluginOverrides:
    """``gateway_plugin.<manifest name>`` in config.yaml.

    Two properties matter here. The key is the *manifest* name, not the
    WasmPlugin resource name -- five of the eight differ, and keying by the
    wrong one matches nothing silently. And ``config`` is a whitelist: the
    fields that would let a config file grant access rather than configure a
    plugin are absent from the model, so writing one is a startup error.
    """

    def _cfg(self, gateway_plugin):
        # Real entries, not raw dicts: a live Config validates this section into
        # GatewayPluginEntry, so a dict here would test a shape production
        # never sees -- and would skip the extra="forbid" on url / sha256.
        entries = {
            "url": "oci://example.com/ext-auth:1",
            **gateway_plugin.pop("gpustack-ext-auth", {}),
        }
        cfg = MagicMock()
        cfg.gateway_plugin = {
            "gpustack-ext-auth": GatewayPluginEntry.model_validate(entries),
            **{
                name: GatewayPluginEntry.model_validate(entry)
                for name, entry in gateway_plugin.items()
            },
        }
        cfg.gateway_plugin_server_url = "http://127.0.0.1"
        cfg.get_derived_gateway_token.return_value = "token"
        cfg.get_derived_auth_cache_key.return_value = "auth-cache-key"
        cfg.get_namespace.return_value = "default"
        cfg.gateway_namespace = "higress-system"
        return cfg

    def _ext_auth(self, gateway_plugin):
        registry = MagicMock()
        registry.get_service_name.return_value = "gpustack.static"
        registry.port = 80
        with patch(
            "gpustack.gateway.get_gpustack_higress_registry", return_value=registry
        ):
            return ext_auth_plugin(cfg=self._cfg(gateway_plugin))[1]

    def test_config_knobs_reach_the_default_config(self):
        spec = self._ext_auth(
            {
                "gpustack-ext-auth": {
                    "config": {
                        "local_auth": {"enabled": False},
                        "authz": {"timeout": 5000},
                        "failure_mode_allow_authenticated": False,
                    }
                }
            }
        )

        assert spec.defaultConfig["local_auth"]["enabled"] is False
        assert spec.defaultConfig["authz"]["timeout"] == 5000
        assert spec.defaultConfig["failure_mode_allow_authenticated"] is False

    def test_defaults_apply_when_the_section_is_absent(self):
        spec = self._ext_auth({})

        assert spec.defaultConfig["local_auth"]["enabled"] is True
        # Empirical, and deliberately generous: 1s shipped once and was
        # reverted, 3s was no better. A saturated Python server routinely takes
        # seconds to answer, and treating that as an outage would skip
        # authorization for every caller the gateway can name.
        assert spec.defaultConfig["authz"]["timeout"] == 30000
        assert spec.defaultConfig["failure_mode_allow_authenticated"] is True

    @pytest.mark.parametrize(
        "config,why",
        [
            (
                {"access_policy": "public"},
                "globally it would make every route locally allowed",
            ),
            (
                {"local_auth": {"keys": {"ak": {"digest": "x", "user_id": 1}}}},
                "keys is the authentication table",
            ),
            (
                {"route_match_regexes": ["^"]},
                "the gate that scopes this plugin to gpustack's own routes",
            ),
            (
                {"auth_cache": {"signing_key": "attacker-chosen"}},
                "the marker signing key is derived from jwt_secret_key",
            ),
            (
                {"upstream_request": {"headers_to_remove": []}},
                "not stripping cookie leaks the session to the model",
            ),
            (
                {"failure_mode_allow": True},
                "admits requests carrying no credential at all",
            ),
            (
                {"authz": {"endpoint": {"path": "/elsewhere"}}},
                "the endpoint is gpustack's own /token-auth",
            ),
        ],
    )
    def test_fields_that_would_grant_access_are_refused(self, config, why):
        with pytest.raises(ValueError, match="Invalid gateway_plugin"):
            self._ext_auth({"gpustack-ext-auth": {"config": config}})

    def test_a_typo_in_a_knob_is_refused_too(self):
        # Silently ignoring it would leave the plugin on a default the operator
        # believes they changed.
        with pytest.raises(ValueError, match="Invalid gateway_plugin"):
            self._ext_auth(
                {"gpustack-ext-auth": {"config": {"local_auth": {"enable": False}}}}
            )


class TestDeprecatedGatewayEnvs:
    """The two settings that predate ``gateway_plugin`` still work.

    Dropping them outright would have failed silently rather than loudly: a
    deployment that had raised the ext-auth timeout would quietly get the stock
    one, and one that had added a content type would quietly stop metering it --
    which surfaces as a wrong bill, not an error.
    """

    def test_the_old_env_supplies_the_default(self, monkeypatch):
        monkeypatch.setenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", "60000")
        monkeypatch.setenv(
            "GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES",
            "application/json, text/plain",
        )

        cfg = make_cfg("http://127.0.0.1")

        assert ext_auth_override(cfg).authz.timeout == 60000
        assert ai_statistics_override(cfg).enable_content_types == [
            "application/json",
            "text/plain",
        ]

    def test_the_config_file_wins_over_the_old_env(self, monkeypatch):
        monkeypatch.setenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", "60000")

        cfg = make_cfg(
            "http://127.0.0.1",
            {"gpustack-ext-auth": {"config": {"authz": {"timeout": 5000}}}},
        )

        assert ext_auth_override(cfg).authz.timeout == 5000

    def test_an_unparseable_old_env_falls_back_rather_than_failing(self, monkeypatch):
        """It governs a running gateway, so a bad value is worth a log line and
        the stock default -- not a server that refuses to start."""
        monkeypatch.setenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", "not-a-number")

        assert ext_auth_override(make_cfg("http://127.0.0.1")).authz.timeout == 30000

    def test_the_deprecation_is_warned_once_not_per_render(self, monkeypatch, caplog):
        """It is read from a pydantic ``default_factory``, so it runs every time
        the plugin config is rendered -- which the reconciler does on a timer."""
        from gpustack import envs

        monkeypatch.setattr(envs, "_warned_deprecated_envs", set())
        monkeypatch.setenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", "60000")
        cfg = make_cfg("http://127.0.0.1")

        with caplog.at_level(logging.WARNING, logger="gpustack.envs"):
            for _ in range(3):
                ext_auth_override(cfg)

        assert caplog.text.count("is deprecated") == 1
