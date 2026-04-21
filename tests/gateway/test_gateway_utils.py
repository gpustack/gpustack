import re

from gpustack.gateway.utils import (
    RoutePrefix,
    build_generic_route_header_rule,
    build_generic_route_path_pattern,
    cleanup_generic_route_transformer_spec_diff,
    generate_model_ingress,
    generic_route_transformer_diff_spec,
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


# --- Generic route transformer --------------------------------------------


def test_generic_route_path_pattern_boundary():
    """
    The path pattern must anchor after the id's last digit so that id=1 does
    not spuriously match /model/proxy/10 or /model/proxy/100/foo.
    """
    pat_1 = build_generic_route_path_pattern(1)
    pat_10 = build_generic_route_path_pattern(10)
    assert pat_1 == r"^/model/proxy/1(/.*)?$"
    assert pat_10 == r"^/model/proxy/10(/.*)?$"

    matches_for_1 = [
        "/model/proxy/1",
        "/model/proxy/1/",
        "/model/proxy/1/pooling",
        "/model/proxy/1/v1/models",
        "/model/proxy/1/v1/chat/completions",
    ]
    non_matches_for_1 = [
        "/model/proxy/10",
        "/model/proxy/10/foo",
        "/model/proxy/100/foo",
        "/model/proxy/2/foo",
        "/model/proxy/1bar",
        "/v1/chat/completions",
    ]
    for path in matches_for_1:
        assert re.match(pat_1, path), f"expected {path!r} to match id=1"
    for path in non_matches_for_1:
        assert not re.match(pat_1, path), f"expected {path!r} to NOT match id=1"


def test_generic_route_header_value_after_substitution():
    """
    Higress transformer's `add` with path_pattern substitutes the match with
    `value` inside the full :path. We must ensure the resulting header value is
    the route name alone — not contaminated with the untouched path tail.
    """
    rule = build_generic_route_header_rule(1, "qwen3-0.6b")
    assert rule == {
        "key": "x-higress-llm-model",
        "value": "qwen3-0.6b",
        "path_pattern": r"^/model/proxy/1(/.*)?$",
    }
    for path in [
        "/model/proxy/1",
        "/model/proxy/1/",
        "/model/proxy/1/pooling",
        "/model/proxy/1/v1/models",
    ]:
        header_value = re.sub(rule["path_pattern"], rule["value"], path)
        assert (
            header_value == "qwen3-0.6b"
        ), f"path {path!r} must reduce to route name; got {header_value!r}"


def _empty_transformer_spec() -> WasmPluginSpec:
    """Match the shape produced by generic_route_transformer_plugin(cfg)."""
    return WasmPluginSpec(
        defaultConfig={"reqRules": []},
        defaultConfigDisable=False,
    )


def _first_add_headers(spec: WasmPluginSpec):
    rules = spec.defaultConfig.get("reqRules", [])
    add_block = next((r for r in rules if r.get("operate") == "add"), None)
    return add_block.get("headers", []) if add_block else []


def test_diff_spec_add_first_route():
    spec = _empty_transformer_spec()
    rule = build_generic_route_header_rule(1, "route-one")

    spec = generic_route_transformer_diff_spec(
        spec,
        expected_header_rules=[rule],
        operating_path_pattern=build_generic_route_path_pattern(1),
    )

    assert spec.defaultConfigDisable is False
    assert _first_add_headers(spec) == [rule]


def test_diff_spec_preserves_other_routes():
    spec = _empty_transformer_spec()
    rule_1 = build_generic_route_header_rule(1, "route-one")
    rule_2 = build_generic_route_header_rule(2, "route-two")

    spec = generic_route_transformer_diff_spec(
        spec, [rule_1], build_generic_route_path_pattern(1)
    )
    spec = generic_route_transformer_diff_spec(
        spec, [rule_2], build_generic_route_path_pattern(2)
    )

    headers = _first_add_headers(spec)
    assert rule_1 in headers
    assert rule_2 in headers
    # Sort is deterministic by path_pattern so diff-equal checks are stable.
    assert headers == sorted(headers, key=lambda h: h["path_pattern"])


def test_diff_spec_update_in_place():
    """Changing a route's name replaces its rule, not appends a duplicate."""
    spec = _empty_transformer_spec()
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one-renamed")],
        build_generic_route_path_pattern(1),
    )

    headers = _first_add_headers(spec)
    assert len(headers) == 1
    assert headers[0]["value"] == "route-one-renamed"


def test_diff_spec_remove_route_keeps_siblings():
    spec = _empty_transformer_spec()
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(2, "route-two")],
        build_generic_route_path_pattern(2),
    )
    # route 1's generic_proxy turned off → expected_header_rules is empty
    spec = generic_route_transformer_diff_spec(
        spec, [], build_generic_route_path_pattern(1)
    )

    headers = _first_add_headers(spec)
    assert len(headers) == 1
    assert headers[0]["value"] == "route-two"


def test_diff_spec_does_not_flip_default_config_disable():
    """
    Toggling defaultConfigDisable rewrites Envoy's filter chain and tears down
    in-flight connections, so the diff must leave the flag alone regardless of
    whether any rules remain.
    """
    spec = _empty_transformer_spec()
    # Add then remove everything — flag must stay False the whole way.
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )
    assert spec.defaultConfigDisable is False
    spec = generic_route_transformer_diff_spec(
        spec, [], build_generic_route_path_pattern(1)
    )
    assert spec.defaultConfigDisable is False
    assert spec.defaultConfig == {"reqRules": []}


def test_diff_spec_passes_through_none():
    """Plugin doesn't exist yet → diff returns None so ensure_wasm_plugin can skip."""
    assert (
        generic_route_transformer_diff_spec(
            None, [], build_generic_route_path_pattern(1)
        )
        is None
    )


def test_diff_spec_preserves_unrelated_req_rules():
    """
    Diff must coexist with foreign reqRules — a future contributor may add
    another `operate: rename` block or a separate `add` block with non-generic
    headers to the same plugin. Our logic identifies generic-route rules by
    path_pattern shape and leaves everything else alone.
    """
    spec = _empty_transformer_spec()
    foreign_rename_block = {
        "operate": "rename",
        "headers": [{"oldKey": "a", "newKey": "b"}],
    }
    foreign_add_header = {
        "key": "x-other",
        "value": "v",
        "path_pattern": "^/other/path",
    }
    spec.defaultConfig = {
        "reqRules": [
            foreign_rename_block,
            {"operate": "add", "headers": [foreign_add_header]},
        ],
    }

    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )

    rules = spec.defaultConfig["reqRules"]
    # Foreign rename block untouched.
    assert foreign_rename_block in rules
    # Foreign add header preserved (may be in its own block).
    assert any(
        r.get("operate") == "add" and foreign_add_header in r.get("headers", [])
        for r in rules
    )
    # Generic-route rule landed in an add block of its own.
    assert any(
        r.get("operate") == "add"
        and any(h.get("value") == "route-one" for h in r.get("headers", []))
        for r in rules
    )


def test_cleanup_spec_diff_prunes_orphans():
    spec = _empty_transformer_spec()
    # Seed with two routes, then run cleanup expecting only route 2 to survive.
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(2, "route-two")],
        build_generic_route_path_pattern(2),
    )

    spec = cleanup_generic_route_transformer_spec_diff(
        spec, expected_path_patterns={build_generic_route_path_pattern(2)}
    )

    headers = _first_add_headers(spec)
    assert len(headers) == 1
    assert headers[0]["value"] == "route-two"
    assert spec.defaultConfigDisable is False


def test_cleanup_spec_diff_empties_when_no_routes_remain():
    spec = _empty_transformer_spec()
    spec = generic_route_transformer_diff_spec(
        spec,
        [build_generic_route_header_rule(1, "route-one")],
        build_generic_route_path_pattern(1),
    )
    spec = cleanup_generic_route_transformer_spec_diff(
        spec, expected_path_patterns=set()
    )
    assert spec.defaultConfig == {"reqRules": []}
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
