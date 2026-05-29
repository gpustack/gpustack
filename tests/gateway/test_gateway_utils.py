from gpustack.gateway import generic_proxy_router_spec_diff
import re

import pytest
from fastapi import HTTPException

from gpustack.api.exceptions import NotFoundException
from gpustack.gateway.utils import (
    RoutePrefix,
    cleanup_generic_proxy_router_spec_diff,
    generate_model_ingress,
    generic_proxy_router_diff_spec,
    get_instance_id_from_header,
    lora_registry_name_suffix,
    model_instance_registry,
    model_instances_registry_list,
    provider_registry,
    router_header_key,
)
from gpustack.schemas.models import ModelInstance
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


# --- Generic proxy router init-time spec diff -----------------------------


def _expected_router_spec() -> WasmPluginSpec:
    """Match the shape produced by generic_proxy_router_plugin(cfg)."""
    return WasmPluginSpec(
        defaultConfig={
            "prefix": "/model/proxy/",
            "targetHeader": "x-higress-llm-model",
            "enableOnPathSuffix": ["/v1/chat/completions", "/v1/messages"],
            "aliasNameMapping": {},
        },
        defaultConfigDisable=False,
        phase="AUTHN",
        priority=900,
        url="http://plugin-server/gpustack-generic-proxy-router/1.0.0/plugin.wasm",
    )


def test_init_diff_passes_through_when_plugin_missing():
    """No live spec → factory expected_spec is returned untouched (no live state to merge)."""
    expected = _expected_router_spec()
    result = generic_proxy_router_spec_diff(None, expected)
    assert result is expected


def test_init_diff_preserves_live_alias_name_mapping():
    """Controller-managed mapping must survive a restart even though every other field is refreshed."""
    expected = _expected_router_spec()
    live = _expected_router_spec()
    live.defaultConfig = {
        # stale defaultConfig as if an older release wrote it
        "prefix": "/old/prefix/",
        "targetHeader": "x-old-header",
        "aliasNameMapping": {"1": "route-one", "2": "route-two"},
    }

    result = generic_proxy_router_spec_diff(live, expected)

    # refreshed from expected
    assert result.defaultConfig["prefix"] == "/model/proxy/"
    assert result.defaultConfig["targetHeader"] == "x-higress-llm-model"
    assert result.defaultConfig["enableOnPathSuffix"] == [
        "/v1/chat/completions",
        "/v1/messages",
    ]
    # preserved from live
    assert result.defaultConfig["aliasNameMapping"] == {
        "1": "route-one",
        "2": "route-two",
    }


def test_init_diff_preserves_operator_max_body_bytes():
    """``maxBodyBytes`` isn't set by the factory; if the operator set it on the live spec, keep it."""
    expected = _expected_router_spec()
    live = _expected_router_spec()
    live.defaultConfig = {
        "aliasNameMapping": {"1": "route-one"},
        "maxBodyBytes": 10 * 1024 * 1024,
    }

    result = generic_proxy_router_spec_diff(live, expected)

    assert result.defaultConfig["maxBodyBytes"] == 10 * 1024 * 1024
    assert result.defaultConfig["aliasNameMapping"] == {"1": "route-one"}


def test_init_diff_drops_live_default_config_keys_outside_preserve_list():
    """
    Anything in the live defaultConfig that isn't on the preserve list (e.g. an
    operator's ad-hoc ``enableOnPathSuffix`` override) gets overwritten by the
    factory's value on restart.
    """
    expected = _expected_router_spec()
    live = _expected_router_spec()
    live.defaultConfig = {
        "enableOnPathSuffix": ["/operator-tweaked"],
        "autoRouting": {"enable": True},
        "aliasNameMapping": {"1": "route-one"},
    }

    result = generic_proxy_router_spec_diff(live, expected)

    assert result.defaultConfig["enableOnPathSuffix"] == [
        "/v1/chat/completions",
        "/v1/messages",
    ]
    assert "autoRouting" not in result.defaultConfig
    assert result.defaultConfig["aliasNameMapping"] == {"1": "route-one"}


def test_init_diff_treats_none_values_as_unset():
    """A live key explicitly set to None falls through to the factory's value."""
    expected = _expected_router_spec()
    live = _expected_router_spec()
    live.defaultConfig = {
        "aliasNameMapping": None,
        "maxBodyBytes": None,
    }

    result = generic_proxy_router_spec_diff(live, expected)

    # factory's {} wins because live value was None
    assert result.defaultConfig["aliasNameMapping"] == {}
    # maxBodyBytes wasn't in the factory either → stays unset
    assert "maxBodyBytes" not in result.defaultConfig


def test_init_diff_does_not_mutate_expected_spec():
    """
    The diff is built with ``partial(spec_diff, expected_spec=plugin_spec)`` —
    if it mutated ``expected_spec.defaultConfig`` in place, state would leak
    across hypothetical re-invocations.
    """
    expected = _expected_router_spec()
    expected_config_before = dict(expected.defaultConfig)
    live = _expected_router_spec()
    live.defaultConfig = {"aliasNameMapping": {"1": "route-one"}}

    generic_proxy_router_spec_diff(live, expected)

    assert expected.defaultConfig == expected_config_before
    assert expected.defaultConfig["aliasNameMapping"] == {}


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


def test_lora_registry_name_suffix_stable_distinct_and_safe():
    suffix = lora_registry_name_suffix("qwen3-0.6b:french")
    # Stable: same LoRA route name always yields the same suffix.
    assert lora_registry_name_suffix("qwen3-0.6b:french") == suffix
    # Distinct: different LoRA route names do not collide.
    assert lora_registry_name_suffix("qwen3-0.6b:german") != suffix
    # RFC1123 label safe (lowercase alphanumeric + hyphen, starts/ends alnum).
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", suffix)


def test_model_instance_registry_name_suffix_static():
    instance = ModelInstance(id=12, model_id=5, worker_ip="1.2.3.4", port=8000)
    base = model_instance_registry(instance)
    suffix = lora_registry_name_suffix("qwen3-0.6b:french")
    alias = model_instance_registry(instance, name_suffix=suffix)
    # Distinct service name, same upstream address/port/type.
    assert base.name == "model-5-12"
    assert alias.name == f"model-5-12-{suffix}"
    assert alias.get_service_name() != base.get_service_name()
    assert alias.domain == base.domain == "1.2.3.4:8000"
    assert alias.port == base.port == 80
    assert alias.type == base.type == "static"


def test_model_instance_registry_name_suffix_dns():
    instance = ModelInstance(
        id=7, model_id=3, worker_ip="worker.example.com", port=8001
    )
    base = model_instance_registry(instance)
    alias = model_instance_registry(instance, name_suffix="labcdef12")
    assert base.type == "dns"
    assert alias.name == "model-3-7-labcdef12"
    assert alias.domain == base.domain == "worker.example.com"
    assert alias.port == base.port == 8001
    assert alias.type == "dns"


def test_get_instance_id_from_header_base_and_lora_alias():
    # Base service name: instance id is the second numeric segment.
    assert get_instance_id_from_header({router_header_key: "model-1-2.static"}) == 2
    # LoRA alias service name (extra -l<hash> segment) must still resolve to the
    # underlying instance id, not the alias.
    suffix = lora_registry_name_suffix("qwen3-0.6b:french")
    assert (
        get_instance_id_from_header({router_header_key: f"model-1-2-{suffix}.static"})
        == 2
    )
    # Multi-digit model/instance ids.
    assert (
        get_instance_id_from_header({router_header_key: "model-10-234-labcdef12.dns"})
        == 234
    )


def test_get_instance_id_from_header_invalid():
    with pytest.raises(HTTPException):
        get_instance_id_from_header({})
    with pytest.raises(NotFoundException):
        get_instance_id_from_header({router_header_key: "cluster-gateway.static"})


def test_model_instances_registry_list_threads_suffix():
    instances = [
        ModelInstance(id=1, model_id=5, worker_ip="1.2.3.4", port=8000),
        ModelInstance(id=2, model_id=5, worker_ip="1.2.3.5", port=8000),
    ]
    suffix = lora_registry_name_suffix("qwen3-0.6b:french")
    dests = model_instances_registry_list(
        instances,
        downstream_model_name="qwen3-0.6b:french",
        registry_name_suffix=suffix,
    )
    assert [model_name for _, model_name, _ in dests] == [
        "qwen3-0.6b:french",
        "qwen3-0.6b:french",
    ]
    assert [registry.name for _, _, registry in dests] == [
        f"model-5-1-{suffix}",
        f"model-5-2-{suffix}",
    ]
