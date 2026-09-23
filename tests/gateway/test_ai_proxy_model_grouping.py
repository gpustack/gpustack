from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
    WasmPluginSpec,
)
from gpustack.gateway.utils import (
    ModelAIProxyGroup,
    _ai_proxy_anthropic_capabilities,
    ai_proxy_diff_spec,
    ai_proxy_model_provider_config,
    cleanup_ai_proxy_config,
    model_ai_proxy_plugin_spec,
    model_ai_proxy_provider_id,
    provider_id_prefix,
)
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model

ROUTE_A = "default/ai-route-route-1.internal"
ROUTE_A_FALLBACK = "default/ai-route-route-1.fallback.internal"
ROUTE_B = "default/ai-route-route-2.internal"
ROUTE_B_FALLBACK = "default/ai-route-route-2.fallback.internal"

LEGACY_A = "ai-route-route-1"


def _spec(
    providers: Optional[List[Dict[str, Any]]] = None,
    match_rules: Optional[List[WasmPluginMatchRule]] = None,
) -> WasmPluginSpec:
    return WasmPluginSpec(
        defaultConfig={"providers": providers or []},
        matchRules=match_rules or [],
    )


def _rule(provider_id: str, ingress: str, services: List[str]) -> WasmPluginMatchRule:
    # The legacy per-route rule form: ingress-keyed. Fresh rules are built by
    # model_ai_proxy_plugin_spec and carry no ingress.
    return WasmPluginMatchRule(
        config={"activeProviderId": provider_id},
        configDisable=False,
        service=services,
        ingress=[ingress],
    )


def _service_rule(provider_id: str, services: List[str]) -> WasmPluginMatchRule:
    """The current rule form: service-only, no ingress."""
    return WasmPluginMatchRule(
        config={"activeProviderId": provider_id},
        configDisable=False,
        service=services,
    )


def _provider_ids(spec: WasmPluginSpec) -> List[str]:
    return [p["id"] for p in spec.defaultConfig["providers"]]


def _rule_keys(spec: WasmPluginSpec) -> List[tuple]:
    return [
        (
            (r.config or {}).get("activeProviderId"),
            tuple(r.ingress or []),
            tuple(r.service or []),
        )
        for r in spec.matchRules or []
    ]


def test_provider_id_is_per_deployment():
    assert model_ai_proxy_provider_id(7) == "gpustack-model-7"


def test_api_tokens_only_present_when_supplied():
    with_token = ai_proxy_model_provider_config("gpustack-model-7", ["gpustack_ak_sk"])
    assert with_token["apiTokens"] == ["gpustack_ak_sk"]
    # Absent rather than empty: an empty list would be a CR change with no
    # meaning, and ai-proxy's fallback to the inbound header is what we want.
    assert "apiTokens" not in ai_proxy_model_provider_config("gpustack-model-7")


def test_capabilities_only_present_when_anthropic_is_native():
    plain = ai_proxy_model_provider_config("gpustack-model-7")
    # The openai provider's own defaults already cover the OpenAI surface, so
    # the common case must not write a capabilities map at all -- an empty one
    # would be a CR change with no meaning.
    assert "capabilities" not in plain
    assert plain["type"] == "openai"

    native = ai_proxy_model_provider_config(
        "gpustack-model-7", native_anthropic_api=True
    )
    # Still an openai provider: capabilities merge over its defaults, so this is
    # the OpenAI surface *plus* Anthropic rather than a swap.
    assert native["type"] == "openai"
    assert native["capabilities"] == {
        "anthropic/v1/messages": "/v1/messages",
        "anthropic/v1/messages/count_tokens": "/v1/messages/count_tokens",
    }


def test_capabilities_do_not_alias_the_module_level_map():
    """Every provider entry owns its capabilities dict.

    Nothing in the builder copies ``_ai_proxy_anthropic_capabilities``
    explicitly -- pydantic validation and ``model_dump`` each construct a fresh
    dict on the way through. That makes the isolation an implementation detail
    of pydantic rather than something the code states, so assert it: a provider
    entry mutated by a later reconcile step must not reach the shared map or
    another deployment's entry.
    """
    before = dict(_ai_proxy_anthropic_capabilities)
    first = ai_proxy_model_provider_config(
        "gpustack-model-1", native_anthropic_api=True
    )
    second = ai_proxy_model_provider_config(
        "gpustack-model-2", native_anthropic_api=True
    )

    assert first["capabilities"] is not _ai_proxy_anthropic_capabilities
    assert first["capabilities"] is not second["capabilities"]

    first["capabilities"]["anthropic/v1/messages"] = "/mutated"
    assert _ai_proxy_anthropic_capabilities == before
    assert second["capabilities"] == before


def test_native_anthropic_api_reaches_the_provider_entry():
    providers, _ = model_ai_proxy_plugin_spec(
        [
            ModelAIProxyGroup(
                model_id=5,
                api_tokens=["t"],
                service_names={"model-5-1.static"},
                native_anthropic_api=True,
            )
        ],
    )
    (provider,) = providers
    assert "anthropic/v1/messages" in provider["capabilities"]


def test_plugin_spec_groups_by_deployment():
    groups = [
        ModelAIProxyGroup(
            model_id=9,
            api_tokens=["token-cluster-2"],
            service_names={"model-9-2.static", "model-9-1.static"},
        ),
        ModelAIProxyGroup(
            model_id=5,
            api_tokens=["token-cluster-1"],
            service_names={"model-5-1.static"},
        ),
    ]

    providers, rules = model_ai_proxy_plugin_spec(groups)

    # One provider per deployment, each holding its own cluster's token.
    assert [p["id"] for p in providers] == ["gpustack-model-5", "gpustack-model-9"]
    assert providers[0]["apiTokens"] == ["token-cluster-1"]
    assert providers[1]["apiTokens"] == ["token-cluster-2"]
    # One service-only rule per deployment: service names are globally unique,
    # so the rule selects the provider on the main ingress, the fallback
    # ingress and the LB-selected cluster alike. Services are sorted so an
    # unchanged deployment yields a byte-identical CR.
    assert [(r.config["activeProviderId"], r.ingress, r.service) for r in rules] == [
        ("gpustack-model-5", [], ["model-5-1.static"]),
        ("gpustack-model-9", [], ["model-9-1.static", "model-9-2.static"]),
    ]


def test_plugin_spec_skips_deployment_without_services():
    providers, rules = model_ai_proxy_plugin_spec(
        [ModelAIProxyGroup(model_id=5, api_tokens=["t"])]
    )
    assert providers == []
    assert rules == []


def test_reconcile_replaces_stale_rules_of_the_same_provider():
    """A deployment's rules have exactly one owner form: reconciling a route
    that points at the deployment replaces every rule of that provider with
    the fresh service-only one -- a stale ingress-keyed rule another route
    wrote retires in the same write, whatever ingress it carries."""
    live = _spec(
        providers=[ai_proxy_model_provider_config("gpustack-model-5", ["token"])],
        match_rules=[_rule("gpustack-model-5", ROUTE_B, ["model-5-1.static"])],
    )
    providers, rules = model_ai_proxy_plugin_spec(
        [
            ModelAIProxyGroup(
                model_id=5, api_tokens=["token"], service_names={"model-5-1.static"}
            )
        ]
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=providers,
        expected_match_rules=rules,
        owned_provider_ids={"gpustack-model-5", LEGACY_A},
    )

    assert _provider_ids(result) == ["gpustack-model-5"], "provider must not duplicate"
    assert _rule_keys(result) == [("gpustack-model-5", (), ("model-5-1.static",))]


def test_route_removal_retires_only_the_legacy_per_route_entry():
    live = _spec(
        providers=[
            ai_proxy_model_provider_config("gpustack-model-5", ["token"]),
            {"id": LEGACY_A, "type": "openai"},
        ],
        match_rules=[
            _service_rule("gpustack-model-5", ["model-5-1.static"]),
            _rule(LEGACY_A, ROUTE_A, ["model-5-1.static"]),
        ],
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=[],
        expected_match_rules=[],
        owned_provider_ids={LEGACY_A},
    )

    # Deployment entries are model-scoped: deleting the route keeps them until
    # the deployment goes; only the route's own legacy entry retires.
    assert _rule_keys(result) == [("gpustack-model-5", (), ("model-5-1.static",))]
    assert _provider_ids(result) == ["gpustack-model-5"]


def test_deployment_provider_survives_last_route_removal():
    """Deployment entries are garbage collected with the deployment, not with
    the last route pointing at it: a rule a deleted route leaves behind on a
    live deployment is inert (no route leads traffic to those services) and
    retires when the deployment does."""
    live = _spec(
        providers=[ai_proxy_model_provider_config("gpustack-model-5", ["token"])],
        match_rules=[_service_rule("gpustack-model-5", ["model-5-1.static"])],
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=[],
        expected_match_rules=[],
        owned_provider_ids=set(),
    )

    assert _rule_keys(result) == [("gpustack-model-5", (), ("model-5-1.static",))]
    assert _provider_ids(result) == ["gpustack-model-5"]


def test_token_change_updates_the_existing_provider_entry():
    live = _spec(
        providers=[ai_proxy_model_provider_config("gpustack-model-5", ["old-token"])],
        match_rules=[_service_rule("gpustack-model-5", ["model-5-1.static"])],
    )
    providers, rules = model_ai_proxy_plugin_spec(
        [
            ModelAIProxyGroup(
                model_id=5, api_tokens=["new-token"], service_names={"model-5-1.static"}
            )
        ]
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=providers,
        expected_match_rules=rules,
        owned_provider_ids={"gpustack-model-5"},
    )

    assert _provider_ids(result) == ["gpustack-model-5"]
    assert result.defaultConfig["providers"][0]["apiTokens"] == ["new-token"]


def test_external_provider_reconcile_leaves_deployment_entries_alone():
    """``ModelProviderController`` owns ``provider-<id>`` entries and reconciles
    them by id prefix; its rules match on service only, so they carry no ingress.
    """
    live = _spec(
        providers=[
            ai_proxy_model_provider_config("gpustack-model-5", ["token"]),
            {"id": "provider-3", "type": "openai", "apiTokens": ["stale"]},
        ],
        match_rules=[
            _service_rule("gpustack-model-5", ["model-5-1.static"]),
            WasmPluginMatchRule(
                config={"activeProviderId": "provider-3"},
                configDisable=False,
                service=["provider-3.dns"],
            ),
        ],
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=[
            {"id": "provider-3", "type": "openai", "apiTokens": ["ok"]}
        ],
        expected_match_rules=[
            WasmPluginMatchRule(
                config={"activeProviderId": "provider-3"},
                configDisable=False,
                service=["provider-3.dns"],
            )
        ],
        operating_id_prefix=provider_id_prefix,
    )

    assert _provider_ids(result) == ["gpustack-model-5", "provider-3"]
    assert [p for p in result.defaultConfig["providers"] if p["id"] == "provider-3"][0][
        "apiTokens"
    ] == ["ok"]
    assert _rule_keys(result) == [
        ("gpustack-model-5", (), ("model-5-1.static",)),
        ("provider-3", (), ("provider-3.dns",)),
    ]


def test_diff_is_idempotent():
    groups = [
        ModelAIProxyGroup(
            model_id=5,
            api_tokens=["token"],
            service_names={"model-5-2.static", "model-5-1.static"},
        )
    ]
    providers, rules = model_ai_proxy_plugin_spec(groups)

    first = ai_proxy_diff_spec(
        _spec(),
        expected_providers=providers,
        expected_match_rules=rules,
        owned_provider_ids={"gpustack-model-5", LEGACY_A},
    )
    providers, rules = model_ai_proxy_plugin_spec(groups)
    second = ai_proxy_diff_spec(
        first,
        expected_providers=providers,
        expected_match_rules=rules,
        owned_provider_ids={"gpustack-model-5", LEGACY_A},
    )

    assert second.model_dump(exclude_none=True) == first.model_dump(exclude_none=True)


def test_diff_returns_none_when_plugin_absent():
    assert (
        ai_proxy_diff_spec(None, expected_providers=[], expected_match_rules=[]) is None
    )


def test_legacy_route_entry_migrates_in_one_write():
    """A route retires its own legacy entry in the same write that adds the
    deployment entry replacing it, so it is never left without a provider.
    Ownership is the provider id -- the legacy one derives from the route id
    -- so the legacy rules drop whatever ingress they carry.
    """
    live = _spec(
        providers=[{"id": LEGACY_A, "type": "openai"}],
        match_rules=[
            _rule(LEGACY_A, ROUTE_A, ["model-5-1.static"]),
            _rule(LEGACY_A, ROUTE_A_FALLBACK, ["model-5-1.static"]),
        ],
    )
    providers, rules = model_ai_proxy_plugin_spec(
        [
            ModelAIProxyGroup(
                model_id=5,
                api_tokens=["token"],
                service_names={"model-5-1.static"},
            )
        ]
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=providers,
        expected_match_rules=rules,
        owned_provider_ids={"gpustack-model-5", LEGACY_A},
    )

    assert _provider_ids(result) == ["gpustack-model-5"]
    assert _rule_keys(result) == [("gpustack-model-5", (), ("model-5-1.static",))]


def test_another_routes_legacy_entry_is_left_alone():
    """Ownership is by provider id, so reconciling one route must not retire
    the legacy entry of a route that has not reconciled yet.
    """
    live = _spec(
        providers=[{"id": "ai-route-route-2", "type": "openai"}],
        match_rules=[_rule("ai-route-route-2", ROUTE_B, ["model-9-1.static"])],
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=[],
        expected_match_rules=[],
        owned_provider_ids={LEGACY_A},
    )

    assert _provider_ids(result) == ["ai-route-route-2"]
    assert _rule_keys(result) == [
        ("ai-route-route-2", (ROUTE_B,), ("model-9-1.static",))
    ]


@pytest.mark.asyncio
async def test_startup_cleanup_drops_only_entries_nothing_will_reconcile():
    """The startup pass keeps live providers and live deployments, and
    retires every legacy per-route entry wholesale — running this pass
    means control is already on this server version, so the per-route
    entries are dropped in one deterministic write instead of racing
    per-route retirements against sibling reconciles; the replay
    rewrites each deployment's own entry moments later."""
    live_plugin = {
        "metadata": {"name": "gpustack-ai-proxy"},
        "spec": {
            "defaultConfig": {
                "providers": [
                    {"id": LEGACY_A, "type": "openai"},
                    {"id": "ai-route-route-404", "type": "openai"},
                    {"id": "gpustack-model-5", "type": "openai"},
                    {"id": "gpustack-model-404", "type": "openai"},
                    {"id": "provider-3", "type": "openai"},
                    {"id": "provider-404", "type": "openai"},
                ]
            },
            "matchRules": [
                # legacy entry of a LIVE route — retired here too
                _rule(LEGACY_A, ROUTE_A, ["model-5-1.static"]).model_dump(),
                _service_rule("gpustack-model-5", ["model-5-1.static"]).model_dump(),
                _service_rule(
                    "gpustack-model-404", ["model-404-1.static"]
                ).model_dump(),
            ],
        },
    }
    api = MagicMock()
    api.get_wasmplugin = AsyncMock(return_value=live_plugin)
    api.edit_wasmplugin = AsyncMock()

    with (
        patch("gpustack.gateway.utils.ExtensionsHigressIoV1Api", return_value=api),
        patch("gpustack.gateway.utils.k8s_client.ApiClient"),
    ):
        await cleanup_ai_proxy_config(
            providers=[ModelProvider(id=3, name="p3")],
            models=[Model(id=5, name="m5")],
            k8s_config=MagicMock(),
            namespace="higress-system",
        )

    body = api.edit_wasmplugin.await_args.kwargs["body"]
    assert _provider_ids(body.spec) == [
        "gpustack-model-5",
        "provider-3",
    ]
    assert _rule_keys(body.spec) == [
        ("gpustack-model-5", (), ("model-5-1.static",)),
    ]


@pytest.mark.asyncio
async def test_startup_cleanup_drops_only_dead_providers():
    """Retention is judged by provider id: entries of a deployment or provider
    that no longer exists are dropped, a rule a deleted route left behind on a
    live deployment is kept (inert — no route leads traffic to those services
    — and retired with the deployment), and the legacy per-route entry of a
    route that no longer exists is dropped.
    """
    live_plugin = {
        "metadata": {"name": "gpustack-ai-proxy"},
        "spec": {
            "defaultConfig": {
                "providers": [
                    {"id": "gpustack-model-5", "type": "openai"},
                    {"id": "provider-3", "type": "openai"},
                ]
            },
            "matchRules": [
                _service_rule("gpustack-model-5", ["model-5-1.static"]).model_dump(),
                # Route 2 is gone, but the deployment provider is alive.
                _rule("ai-route-route-2", ROUTE_B, ["model-5-1.static"]).model_dump(),
                _service_rule("provider-3", ["provider-3.dns"]).model_dump(),
            ],
        },
    }
    api = MagicMock()
    api.get_wasmplugin = AsyncMock(return_value=live_plugin)
    api.edit_wasmplugin = AsyncMock()

    with (
        patch("gpustack.gateway.utils.ExtensionsHigressIoV1Api", return_value=api),
        patch("gpustack.gateway.utils.k8s_client.ApiClient"),
    ):
        await cleanup_ai_proxy_config(
            providers=[ModelProvider(id=3, name="p3")],
            models=[Model(id=5, name="m5")],
            k8s_config=MagicMock(),
            namespace="higress-system",
        )

    body = api.edit_wasmplugin.await_args.kwargs["body"]
    assert _rule_keys(body.spec) == [
        ("gpustack-model-5", (), ("model-5-1.static",)),
        ("provider-3", (), ("provider-3.dns",)),
    ]
    assert _provider_ids(body.spec) == ["gpustack-model-5", "provider-3"]


def test_none_config_and_default_config_are_tolerated():
    """``config`` and ``defaultConfig`` are Optional on the CR models, so a
    hand-edited ``null`` must not crash the reconcile.
    """
    live = WasmPluginSpec(
        defaultConfig=None,
        matchRules=[WasmPluginMatchRule(config=None, ingress=[ROUTE_B])],
    )

    result = ai_proxy_diff_spec(
        live,
        expected_providers=[ai_proxy_model_provider_config("gpustack-model-5", ["t"])],
        expected_match_rules=[_service_rule("gpustack-model-5", ["model-5-1.static"])],
        owned_provider_ids={"gpustack-model-5"},
    )

    assert _provider_ids(result) == ["gpustack-model-5"]
    assert _rule_keys(result) == [
        (None, (ROUTE_B,), ()),
        ("gpustack-model-5", (), ("model-5-1.static",)),
    ]
