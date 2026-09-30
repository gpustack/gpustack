import pytest
from pydantic import ValidationError

from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
    WasmPluginSpec,
)
from gpustack.routes.plugins.capability_policy import CapabilityPolicy
from gpustack.routes.plugins.decision_service.config import DecisionServiceRouteConfig
from gpustack.routes.plugins.decision_service.plugin import decision_service_plugin
from gpustack.routes.plugins.decision_service.providers import (
    decision_provider_entries,
)
from gpustack.routes.plugins.decision_service.spec_diff import decision_static_spec_diff
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
    ProviderModel,
    TypesafeConfig,
)
from gpustack.schemas.model_routes import ModelRoute


def _route(meta=None):
    return ModelRoute(id=1, name="r", targets=0, ready_targets=0, meta=meta)


def _provider(provider_id=2, endpoint=None, api_tokens=None):
    return ModelProvider(
        id=provider_id,
        name=f"jev-{provider_id}",
        config=TypesafeConfig.model_validate(
            {
                "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
                **({"endpoint": endpoint} if endpoint else {}),
            }
        ),
        api_tokens=api_tokens or [],
    )


class TestDecisionServiceRouteConfig:
    def test_criteria_required_when_model_selection_present(self):
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate(
                {"providerId": 2, "modelSelection": {}}
            )

    def test_empty_criteria_rejected(self):
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate(
                {"providerId": 2, "modelSelection": {"criteria": {}}}
            )

    def test_weight_range(self):
        base = {
            "providerId": 2,
            "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
        }
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate({**base, "weight": 0})
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate({**base, "weight": -1})
        assert DecisionServiceRouteConfig.model_validate({**base, "weight": 0.5})

    def test_gateway_rule_shape(self):
        rule = DecisionServiceRouteConfig.model_validate(
            {
                "providerId": 2,
                "weight": 3,
                "decisionModel": "jev-preview",
                "modelSelection": {
                    "instructions": "pick wisely",
                    "criteria": {"m1": "fast", "m2": "smart"},
                },
            }
        ).to_gateway_rule()
        assert rule == {
            "enabled": True,
            "activeProviderId": "provider-2",
            "decisionModel": "jev-preview",
            "rankWeight": 3,
            "modelSelection": {
                "instructions": "pick wisely",
                "criteria": {"m1": "fast", "m2": "smart"},
            },
        }

    def test_gateway_rule_omits_unset_optionals(self):
        rule = DecisionServiceRouteConfig.model_validate(
            {
                "providerId": 2,
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            }
        ).to_gateway_rule()
        assert rule == {
            "enabled": True,
            "activeProviderId": "provider-2",
            "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
        }


class _MemoryStore:
    """Stands in for the shared capability policy table."""

    def __init__(self, cls=CapabilityPolicy):
        self.cls = cls
        self.rows = {}
        self.delete_calls = []

    def install(self, monkeypatch):
        store = self

        async def one_by_fields(cls, session, fields, **kw):
            return store.rows.get((fields.get("capability"), fields.get("route_id")))

        async def all_by_fields(cls, session, fields=None, extra_conditions=None, **kw):
            capability = (fields or {}).get("capability")
            return [r for (cap, _), r in store.rows.items() if cap == capability]

        async def create(cls, session, source, **kw):
            row = cls(**source)
            store.rows[(source["capability"], source["route_id"])] = row
            return row

        async def _delete(row, session=None, **kw):
            store.delete_calls.append(kw)
            store.rows.pop((row.capability, row.route_id), None)

        async def _update(row, session=None, source=None, **kw):
            for k, v in (source or {}).items():
                setattr(row, k, v)
            return row

        monkeypatch.setattr(self.cls, "one_by_fields", classmethod(one_by_fields))
        monkeypatch.setattr(self.cls, "all_by_fields", classmethod(all_by_fields))
        monkeypatch.setattr(self.cls, "create", classmethod(create))
        monkeypatch.setattr(self.cls, "delete", _delete)
        monkeypatch.setattr(self.cls, "update", _update)

    def row(self, capability="decision-service", route_id=1):
        return self.rows.get((capability, route_id))


class TestHooks:
    @pytest.mark.asyncio
    async def test_section_lands_in_shared_table(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        await decision_service_plugin.on_route_write(
            "update",
            _route(),
            {
                "providerId": 2,
                "weight": 5,
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
            session=None,
        )
        row = store.row()
        assert row.weight == 5
        assert row.config == {
            "enabled": True,
            "providerId": 2,
            "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
        }
        # the route row itself is untouched — storage is the plugin's own
        assert _route().meta is None

    @pytest.mark.asyncio
    async def test_removal_deletes_the_row(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service", route_id=1, config={"enabled": True}
        )
        await decision_service_plugin.on_route_write(
            "update", _route(), None, session=None, removed=True
        )
        assert store.row() is None

    @pytest.mark.asyncio
    async def test_untouched_section_keeps_stored_policy(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service", route_id=1, config={"enabled": True}
        )
        await decision_service_plugin.on_route_write(
            "update", _route(), None, session=None
        )
        assert store.row().config == {"enabled": True}

    @pytest.mark.asyncio
    async def test_unknown_provider_id_rejected(self, monkeypatch):
        async def one_by_id(cls, session, provider_id):
            return None

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        with pytest.raises(ValueError, match="decision-service"):
            await decision_service_plugin.on_route_write(
                "update",
                _route(),
                {
                    "providerId": 9,
                    "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
                },
                session=object(),
            )

    @pytest.mark.asyncio
    async def test_cross_org_provider_id_rejected(self, monkeypatch):
        # providerId is a cross-resource reference from a tenant-owned
        # route; another Org's decision provider must not be selectable,
        # the same alignment the route-target path enforces.
        from gpustack.api.exceptions import InvalidException

        provider = _provider(provider_id=9, endpoint="http://jev.internal:8010")
        provider.owner_principal_id = 42

        async def one_by_id(cls, session, provider_id):
            return provider

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        route = _route()
        route.owner_principal_id = 7
        with pytest.raises(InvalidException, match="same Org"):
            await decision_service_plugin.on_route_write(
                "update",
                route,
                {
                    "providerId": 9,
                    "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
                },
                session=object(),
            )

    @pytest.mark.asyncio
    async def test_soft_deleted_provider_id_rejected(self, monkeypatch):
        provider = _provider(provider_id=9, endpoint="http://jev.internal:8010")
        provider.deleted_at = 1

        async def one_by_id(cls, session, provider_id):
            return provider

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        with pytest.raises(ValueError, match="decision-service"):
            await decision_service_plugin.on_route_write(
                "update",
                _route(),
                {
                    "providerId": 9,
                    "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
                },
                session=object(),
            )


class TestIsEffectiveOn:
    @pytest.mark.asyncio
    async def test_absent_policy_reports_false(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        assert not await decision_service_plugin.is_effective_on(_route(), None)

    @pytest.mark.asyncio
    async def test_enabled_with_model_selection_reports_true(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service",
            route_id=1,
            config={
                "enabled": True,
                "providerId": 7,
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
        )
        assert await decision_service_plugin.is_effective_on(_route(), None)

    @pytest.mark.asyncio
    async def test_enabled_without_model_selection_reports_false(self, monkeypatch):
        # modelSelection absent = the feature is off for the route; the
        # stored policy alone must not count as an LB opinion.
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service",
            route_id=1,
            config={"enabled": True, "providerId": 7},
        )
        assert not await decision_service_plugin.is_effective_on(_route(), None)

    @pytest.mark.asyncio
    async def test_unresolvable_provider_id_is_inert(self, monkeypatch):
        # The provider went away; the stored policy stays (criteria and
        # all), but the feature is off for the route until the reference
        # is fixed — no broken activeProviderId is rendered.
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service",
            route_id=1,
            config={
                "enabled": True,
                "providerId": 7,
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
        )

        async def one_by_id(cls, session, provider_id):
            return None

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        assert not await decision_service_plugin.is_effective_on(_route(), object())

        async def one_by_id_found(cls, session, provider_id):
            return _provider(provider_id=7, endpoint="http://jev.internal:8010")

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id_found))
        assert await decision_service_plugin.is_effective_on(_route(), object())

    @pytest.mark.asyncio
    async def test_decision_model_dropped_from_cache_is_inert(self, monkeypatch):
        # The route pins decisionModel="jev-x"; a models refresh that no
        # longer lists it means the service retired the alias — same
        # treatment as a removed provider: rule stripped, policy kept.
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("decision-service", 1)] = CapabilityPolicy(
            capability="decision-service",
            route_id=1,
            config={
                "enabled": True,
                "providerId": 7,
                "decisionModel": "jev-x",
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
        )
        session = object()

        def one_by_id_with_models(names):
            provider = _provider(provider_id=7, endpoint="http://jev.internal:8010")

            async def one_by_id(cls, sess, provider_id):
                provider.models = [
                    ProviderModel(name=n, category="decision") for n in names
                ]
                return provider

            return classmethod(one_by_id)

        monkeypatch.setattr(
            ModelProvider, "one_by_id", one_by_id_with_models(["jev-latest"])
        )
        assert not await decision_service_plugin.is_effective_on(_route(), session)

        monkeypatch.setattr(
            ModelProvider, "one_by_id", one_by_id_with_models(["jev-x"])
        )
        assert await decision_service_plugin.is_effective_on(_route(), session)

        # cache never pulled: absence is not evidence the alias is gone
        monkeypatch.setattr(ModelProvider, "one_by_id", one_by_id_with_models([]))
        assert await decision_service_plugin.is_effective_on(_route(), session)

    def test_referencing_route_ids_selects_policies_pointing_at_the_provider(self):
        from gpustack.routes.plugins.decision_service.plugin import (
            decision_referencing_route_ids,
        )

        policies = [
            CapabilityPolicy(
                capability="decision-service",
                route_id=1,
                config={"providerId": 7, "enabled": True},
            ),
            CapabilityPolicy(
                capability="decision-service",
                route_id=2,
                config={"providerId": 8, "enabled": True},
            ),
            CapabilityPolicy(capability="decision-service", route_id=3, config=None),
        ]
        assert decision_referencing_route_ids(policies, 7) == {1}
        assert decision_referencing_route_ids(policies, 8) == {2}
        assert decision_referencing_route_ids(None, 7) == set()

    def test_blank_criteria_description_rejected(self):
        # Descriptions are what the decision service reasons over: a
        # name-only question must not persist.
        with pytest.raises(ValidationError, match="capability description"):
            DecisionServiceRouteConfig.model_validate(
                {
                    "providerId": 2,
                    "modelSelection": {"criteria": {"m1": "  ", "m2": "fast"}},
                }
            )

    def test_blank_criteria_key_rejected(self):
        with pytest.raises(ValidationError, match="must not be blank"):
            DecisionServiceRouteConfig.model_validate(
                {"providerId": 2, "modelSelection": {"criteria": {" ": "fast"}}}
            )


class TestProviderEntries:
    def test_no_synthetic_hosted_default_entry(self):
        # No synthetic entry is invented for the hosted API: the decision
        # callout forwards the request body to the service, so a target the
        # operator never wrote down must not appear in the catalogue.
        assert decision_provider_entries([]) == []

    def test_provider_entry_uses_provider_registry_cluster(self):
        # A self-hosted endpoint renders the generic systemone type.
        entries = decision_provider_entries(
            [
                _provider(
                    provider_id=2,
                    endpoint="http://jev.tenant-a.internal:8010",
                    api_tokens=["tenant-a-key"],
                )
            ]
        )
        assert entries == [
            {
                "id": "provider-2",
                "type": "systemone",
                "endpoint": "http://jev.tenant-a.internal:8010",
                "apiToken": "tenant-a-key",
                "cluster": "outbound|8010||provider-2.dns",
            }
        ]

    def test_hosted_endpoint_used_when_no_custom_base_url(self):
        # One provider type covers both flavors: endpoint omitted = the
        # TypeSafe hosted default, written into the entry explicitly; the
        # plugin-side type is uniformly systemone.
        entries = decision_provider_entries(
            [_provider(provider_id=3, api_tokens=["k"])]
        )
        assert entries == [
            {
                "id": "provider-3",
                "type": "systemone",
                "endpoint": "https://api.typesafe.ai",
                "apiToken": "k",
                "cluster": "outbound|443||provider-3.dns",
            }
        ]

    def test_cluster_field_rejected(self):
        # No cluster knob: the registry-derived cluster is the only truth,
        # so an explicit one would be a second way to point at a backend.
        with pytest.raises(ValidationError):
            TypesafeConfig.model_validate(
                {
                    "type": "gpustack-lb-typesafe",
                    "endpoint": "http://jev.internal:8010",
                    "cluster": "outbound|9||custom.dns",
                }
            )

    def test_provider_entry_carries_default_model(self):
        config = TypesafeConfig.model_validate(
            {
                "type": "gpustack-lb-typesafe",
                "endpoint": "http://jev.internal:8010",
                "model": "jev-preview",
            }
        )
        provider = ModelProvider(id=6, name="jev-6", config=config, api_tokens=[])
        entry = decision_provider_entries([provider])[0]
        assert entry["model"] == "jev-preview"

    def test_non_systemone_providers_skipped(self):
        from gpustack.schemas.model_provider import OpenAIConfig

        other = ModelProvider(
            id=5, name="openai", config=OpenAIConfig.model_validate({"type": "openai"})
        )
        entries = decision_provider_entries(
            [
                other,
                _provider(provider_id=2, endpoint="http://jev.internal:8010"),
            ]
        )
        assert [e["id"] for e in entries] == ["provider-2"]


class TestStaticSpecDiff:
    def _expected(self):
        return WasmPluginSpec(
            phase="AUTHN",
            priority=328,
            url="http://plugins/gpustack-lb-decision-service/0.1.0/plugin.wasm",
            defaultConfig={"enabled": False, "decisionTimeoutMs": 3000},
            matchRules=[],
        )

    def test_absent_cr_creates_from_expected(self):
        diff = decision_static_spec_diff(self._expected())
        assert diff(None) == self._expected()

    def test_catalogue_diff_invoked_positionally(self):
        # ensure_wasm_plugin calls spec_diff(current_spec) positionally; a
        # keyword-bound first argument in the partial collides with it
        # (TypeError: got multiple values). Guard the wiring, not just the
        # function: call the partial exactly the way the caller does.
        from functools import partial

        from gpustack.routes.plugins.decision_service.providers import (
            _carry_match_rules,
        )

        spec_diff = partial(_carry_match_rules, {"providers": [], "enabled": False})
        # absent CR -> None: this sync never creates it (the static half is
        # the init pass's to write); ensure_wasm_plugin treats None as no-op
        assert spec_diff(None) is None

    def test_catalogue_diff_preserves_the_static_half(self):
        # The sync owns defaultConfig only. Rebuilding the spec from it alone
        # would strip url/sha256/phase/priority and leave Envoy no module to
        # fetch -- everything else must carry over from the live spec.
        from gpustack.routes.plugins.decision_service.providers import (
            _carry_match_rules,
        )

        current = WasmPluginSpec(
            phase="AUTHN",
            priority=328,
            url="http://plugins/gpustack-lb-decision-service/0.1.0/plugin.wasm",
            sha256="abc123",
            defaultConfig={"enabled": True, "providers": [{"id": "old"}]},
            matchRules=[
                WasmPluginMatchRule(
                    ingress=["ns/ai-route-route-1.internal"], configDisable=False
                )
            ],
        )
        merged = _carry_match_rules({"enabled": False, "providers": []}, current)
        assert merged.url == current.url
        assert merged.sha256 == current.sha256
        assert merged.phase == current.phase
        assert merged.priority == current.priority
        assert merged.defaultConfig == {"enabled": False, "providers": []}
        assert merged.matchRules == current.matchRules

    def test_match_rules_and_catalogue_carried_over(self):
        current = WasmPluginSpec(
            phase="AUTHN",
            priority=328,
            url="http://old/plugin.wasm",
            defaultConfig={
                "providers": [{"id": "provider-2", "type": "typesafe"}],
                # a stale value from before the synthetic default entry was
                # removed: it names a provider the catalogue no longer has,
                # and the plugin's config parse fails on that -- init must
                # let it be dropped, not preserve it
                "activeProviderId": "default",
            },
            matchRules=[
                WasmPluginMatchRule(
                    ingress=["ns/ai-route-route-1.internal"], configDisable=False
                )
            ],
        )
        merged = decision_static_spec_diff(self._expected())(current)
        assert merged.url.endswith("0.1.0/plugin.wasm")
        assert merged.defaultConfig["providers"] == [
            {"id": "provider-2", "type": "typesafe"}
        ]
        assert "activeProviderId" not in merged.defaultConfig
        assert merged.defaultConfig["enabled"] is False
        assert merged.matchRules == current.matchRules


class TestAiproxyExclusion:
    def test_systemone_providers_never_ride_the_aiproxy_catalogue(self):
        from gpustack.gateway.utils import provider_proxy_plugin_spec

        providers, match_rules = provider_proxy_plugin_spec(
            _provider(
                provider_id=2,
                endpoint="http://jev.tenant-a.internal:8010",
                api_tokens=["k"],
            ),
            _provider(provider_id=3, api_tokens=["k"]),
        )
        assert providers == []
        assert match_rules == []


class TestReviewHardening:
    def test_malformed_endpoint_rejected_at_validation(self):
        # A bad url used to surface only inside registry generation, where
        # one malformed endpoint blocked the whole catalogue sync.
        for bad in (
            "https://decision.example.com:abc",
            "not-a-url",
            "ftp://decision.example.com",
        ):
            with pytest.raises(ValidationError, match="endpoint"):
                TypesafeConfig.model_validate(
                    {
                        "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
                        "endpoint": bad,
                    }
                )

    def test_one_broken_entry_does_not_block_the_catalogue(self, monkeypatch):
        # The catalogue spans every organization: a provider whose entry
        # cannot be built is skipped with a warning, the rest still sync.
        from gpustack.routes.plugins.decision_service import providers as mod

        good = _provider(provider_id=2, endpoint="http://jev.internal:8010")
        bad = _provider(provider_id=1, endpoint="http://jev.other:8010")
        real_entry = mod.decision_provider_entry

        def flaky(provider):
            if provider.id == 1:
                raise ValueError("registry boom")
            return real_entry(provider)

        monkeypatch.setattr(mod, "decision_provider_entry", flaky)
        entries = decision_provider_entries([good, bad])
        assert [e["id"] for e in entries] == ["provider-2"]

    def test_decision_payload_detection(self):
        from gpustack.server.controllers import _is_decision_payload

        assert _is_decision_payload(
            {"type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value}
        )
        assert _is_decision_payload(ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE)
        assert _is_decision_payload(
            TypesafeConfig.model_validate(
                {"type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value}
            )
        )
        assert not _is_decision_payload({"type": "openai"})
        assert not _is_decision_payload(None)


class TestChangedFieldsShapes:
    @pytest.mark.parametrize(
        "config_change",
        [
            # cross-instance detect_changes: flat (old, new)
            ({"type": "openai"}, {"type": "gpustack-lb-typesafe"}),
            # local find_history: nested ([old], [new])
            ([{"type": "openai"}], [{"type": "gpustack-lb-typesafe"}]),
            # None side of a None<->value transition arrives as an empty seq
            ((), [{"type": "gpustack-lb-typesafe"}]),
        ],
        ids=["flat", "nested", "none-side"],
    )
    def test_change_shapes_decode_without_raising(self, config_change, monkeypatch):
        # The decode runs before the notify try-block: on the flat
        # cross-instance shape the old code raised TypeError there, so the
        # notification never went out. Every shape must reach the notify
        # path, observed via the policy query it issues.
        import asyncio
        from contextlib import asynccontextmanager

        from gpustack.server.coordinator.base import Event, EventType
        from gpustack.server.controllers import ModelProviderController
        import gpustack.server.controllers as controllers_mod

        controller = ModelProviderController.__new__(ModelProviderController)
        queried = {}

        class _Policies:
            @staticmethod
            async def all_by_fields(session, fields=None, **kw):
                queried["capability"] = (fields or {}).get("capability")
                return []

        @asynccontextmanager
        async def _session():
            yield None

        monkeypatch.setattr(controllers_mod, "CapabilityPolicy", _Policies)
        monkeypatch.setattr(controllers_mod, "async_session", _session)
        event = Event(
            type=EventType.UPDATED,
            data=None,
            id=9,
            changed_fields={"config": config_change},
        )
        asyncio.run(controller._notify_decision_service_routes(event))
        assert queried.get("capability") == "decision-service"
