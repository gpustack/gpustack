from gpustack.gateway.utils import (
    RoutePrefix,
    cleanup_generic_proxy_router_spec_diff,
    generate_model_ingress,
    generic_proxy_router_diff_spec,
    provider_registry,
)
from gpustack.gateway.client.extensions_higress_io_v1_api import WasmPluginSpec
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
    OpenAIConfig,
    OllamaConfig,
)
from gpustack.gateway.client.networking_higress_io_v1_api import McpBridgeRegistry


def test_flattened_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"],
        support_legacy=True,
    ).flattened_prefixes() == [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1-openai/chat/completions",
        "/v1-openai/completions",
        "/v1-openai/responses",
    ]
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"]
    ).flattened_prefixes() == [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
    ]


def test_regex_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"],
        support_legacy=True,
    ).regex_prefixes() == [
        r"/(v1)(-openai)?(/chat/completions)",
        r"/(v1)(-openai)?(/completions)",
        r"/(v1)(-openai)?(/responses)",
    ]
    assert RoutePrefix(["/chat/completions", "/completions"]).regex_prefixes() == [
        r"/(v1)()(/chat/completions)",
        r"/(v1)()(/completions)",
    ]


def test_v2_prefixes():
    rerank = RoutePrefix(
        ["/rerank"],
        support_legacy=False,
        additional_versions=["/v2"],
    )
    assert rerank.regex_prefixes() == [
        r"/(v1)()(/rerank)",
        r"/(v2)()(/rerank)",
    ]
    assert rerank.flattened_prefixes() == [
        "/v1/rerank",
        "/v2/rerank",
    ]


def test_provider_registry_static_ip():
    provider = ModelProvider(
        id=1,
        name="provider-1",
        config=OpenAIConfig(
            type=ModelProviderTypeEnum.OPENAI, openaiCustomUrl="http://1.2.3.4/v1"
        ),
        proxy_url="http://proxy.example.com:8080",
    )
    reg = provider_registry(provider)
    assert isinstance(reg, McpBridgeRegistry)
    assert reg.domain == "1.2.3.4:80"
    assert reg.port == 80
    assert reg.protocol == "http"
    assert reg.type == "static"
    assert reg.name == "provider-1"
    assert reg.proxyName is not None
    assert reg.proxyName == "provider-1-proxy"


def test_provider_registry_dns():
    provider = ModelProvider(
        id=2,
        name="provider-2",
        config=OpenAIConfig(
            type=ModelProviderTypeEnum.OPENAI,
            openaiCustomUrl="https://provider.example.com:8443/v1",
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "provider.example.com"
    assert reg.port == 8443
    assert reg.protocol == "https"
    assert reg.type == "dns"
    assert reg.name == "provider-2"
    assert reg.proxyName is None


def test_ollama_registry():
    provider = ModelProvider(
        id=3,
        name="provider-3",
        config=OllamaConfig(
            type=ModelProviderTypeEnum.OLLAMA,
            ollamaServerHost="localhost",
            ollamaServerPort=8080,
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "localhost"
    assert reg.port == 8080
    assert reg.protocol == "http"
    assert reg.type == "dns"
    assert reg.name == "provider-3"

    provider = ModelProvider(
        id=3,
        name="provider-3",
        config=OllamaConfig(
            type=ModelProviderTypeEnum.OLLAMA,
            ollamaServerHost="1.2.3.4",
            ollamaServerPort=8080,
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "1.2.3.4:8080"
    assert reg.port == 80
    assert reg.type == "static"
    assert reg.protocol == "http"


# --- Generic proxy router -------------------------------------------------


def _empty_router_spec() -> WasmPluginSpec:
    """Match the shape produced by generic_proxy_router_plugin(cfg)."""
    return WasmPluginSpec(
        defaultConfig={
            "prefix": "/model/proxy/",
            "targetHeader": "x-higress-llm-model",
            "aliasNameMapping": {},
        },
        defaultConfigDisable=False,
    )


def _alias_mapping(spec: WasmPluginSpec):
    return spec.defaultConfig.get("aliasNameMapping") or {}


def test_diff_spec_add_first_route():
    spec = _empty_router_spec()

    spec = generic_proxy_router_diff_spec(spec, route_id=1, route_name="route-one")

    assert spec.defaultConfigDisable is False
    assert _alias_mapping(spec) == {"1": "route-one"}


def test_diff_spec_preserves_other_routes():
    spec = _empty_router_spec()

    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    spec = generic_proxy_router_diff_spec(spec, 2, "route-two")

    assert _alias_mapping(spec) == {"1": "route-one", "2": "route-two"}


def test_diff_spec_update_in_place():
    """Changing a route's name replaces its entry, not appends a duplicate."""
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one-renamed")

    assert _alias_mapping(spec) == {"1": "route-one-renamed"}


def test_diff_spec_remove_route_keeps_siblings():
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    spec = generic_proxy_router_diff_spec(spec, 2, "route-two")
    # route 1's generic_proxy turned off → route_name=None
    spec = generic_proxy_router_diff_spec(spec, 1, None)

    assert _alias_mapping(spec) == {"2": "route-two"}


def test_diff_spec_remove_missing_route_is_noop():
    """Removing a route that isn't in the mapping must not raise or affect others."""
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    spec = generic_proxy_router_diff_spec(spec, 99, None)

    assert _alias_mapping(spec) == {"1": "route-one"}


def test_diff_spec_does_not_flip_default_config_disable():
    """
    Toggling defaultConfigDisable rewrites Envoy's filter chain and tears down
    in-flight connections, so the diff must leave the flag alone regardless of
    whether any entries remain.
    """
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    assert spec.defaultConfigDisable is False
    spec = generic_proxy_router_diff_spec(spec, 1, None)
    assert spec.defaultConfigDisable is False
    assert _alias_mapping(spec) == {}


def test_diff_spec_passes_through_none():
    """Plugin doesn't exist yet → diff returns None so ensure_wasm_plugin can skip."""
    assert generic_proxy_router_diff_spec(None, 1, "route-one") is None
    assert generic_proxy_router_diff_spec(None, 1, None) is None


def test_diff_spec_preserves_unrelated_default_config_keys():
    """
    Diff must coexist with other defaultConfig keys (prefix, targetHeader, and
    any future config a contributor adds). Only aliasNameMapping is mutated.
    """
    spec = _empty_router_spec()
    spec.defaultConfig["customKey"] = "preserve-me"

    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")

    assert spec.defaultConfig["prefix"] == "/model/proxy/"
    assert spec.defaultConfig["targetHeader"] == "x-higress-llm-model"
    assert spec.defaultConfig["customKey"] == "preserve-me"
    assert _alias_mapping(spec) == {"1": "route-one"}


def test_cleanup_spec_diff_prunes_orphans():
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")
    spec = generic_proxy_router_diff_spec(spec, 2, "route-two")

    spec = cleanup_generic_proxy_router_spec_diff(spec, expected_route_ids={2})

    assert _alias_mapping(spec) == {"2": "route-two"}
    assert spec.defaultConfigDisable is False


def test_cleanup_spec_diff_empties_when_no_routes_remain():
    spec = _empty_router_spec()
    spec = generic_proxy_router_diff_spec(spec, 1, "route-one")

    spec = cleanup_generic_proxy_router_spec_diff(spec, expected_route_ids=set())

    assert _alias_mapping(spec) == {}
    assert spec.defaultConfig["prefix"] == "/model/proxy/"
    assert spec.defaultConfigDisable is False


# --- Main ingress path rules ----------------------------------------------


def test_included_proxy_route_adds_id_variant_before_legacy():
    """
    When generic_proxy is enabled, the ingress must carry both a /model/proxy/<id>/*
    rule (for URL-based routing) and the legacy /model/proxy/* rule (header-based).
    The id-based rule must list first so Higress tries the more specific match.
    """
    ingress = generate_model_ingress(
        ingress_name="ai-route-route-42.internal",
        namespace="default",
        route_name="my-route",
        destinations="100% svc.default.svc.cluster.local:80",
        included_proxy_route=True,
    )
    paths = [p.path for p in ingress.spec.rules[0].http.paths]

    id_rule = r"/()model/proxy/\d+(/|$)(.*)"
    legacy_rule = r"/()model/proxy(/|$)(.*)"
    assert id_rule in paths
    assert legacy_rule in paths
    assert paths.index(id_rule) < paths.index(
        legacy_rule
    ), "id-based rule must precede legacy rule for specificity-first matching"


def test_included_proxy_route_off_has_no_proxy_paths():
    ingress = generate_model_ingress(
        ingress_name="ai-route-route-42.internal",
        namespace="default",
        route_name="my-route",
        destinations="100% svc.default.svc.cluster.local:80",
        included_proxy_route=False,
    )
    paths = [p.path for p in ingress.spec.rules[0].http.paths]
    assert not any("model/proxy" in p for p in paths)
