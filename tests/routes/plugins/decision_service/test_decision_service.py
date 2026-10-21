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
    decision_model_entries,
    decision_model_entry,
    decision_provider_entries,
)
from gpustack.routes.plugins.decision_service.spec_diff import decision_static_spec_diff
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
    ProviderModel,
    TypesafeConfig,
)
from gpustack.schemas.model_routes import ModelRoute


def _route(meta=None):
    return ModelRoute(id=1, name="r", targets=0, ready_targets=0, meta=meta)


def _provider(provider_id=2, endpoint=None, api_tokens=None, model=None):
    return ModelProvider(
        id=provider_id,
        name=f"jev-{provider_id}",
        config=TypesafeConfig.model_validate(
            {
                "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
                **({"endpoint": endpoint} if endpoint else {}),
                **({"model": model} if model else {}),
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
            "decisionModel": "jev-latest",
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
                "decisionModel": "jev-latest",
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            }
        ).to_gateway_rule()
        assert rule == {
            "enabled": True,
            "activeProviderId": "provider-2",
            "decisionModel": "jev-latest",
            "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
        }

    def test_decision_model_required(self):
        # A blank decision model would ship the decision request without a
        # model field, so the service performs no routing decision and the
        # policy silently does not run.
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate(
                {
                    "providerId": 2,
                    "modelSelection": {"criteria": {"m1": "fast"}},
                }
            )
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate(
                {
                    "providerId": 2,
                    "decisionModel": "   ",
                    "modelSelection": {"criteria": {"m1": "fast"}},
                }
            )


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
                "decisionModel": "jev-latest",
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
            session=None,
        )
        row = store.row()
        assert row.weight == 5
        assert row.config == {
            "enabled": True,
            "providerId": 2,
            "decisionModel": "jev-latest",
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
        async def one_by_id(cls, session, provider_id, for_update=False):
            return None

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        with pytest.raises(ValueError, match="decision-service"):
            await decision_service_plugin.on_route_write(
                "update",
                _route(),
                {
                    "providerId": 9,
                    "decisionModel": "jev-latest",
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

        async def one_by_id(cls, session, provider_id, for_update=False):
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
                    "decisionModel": "jev-latest",
                    "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
                },
                session=object(),
            )

    @pytest.mark.asyncio
    async def test_soft_deleted_provider_id_rejected(self, monkeypatch):
        provider = _provider(provider_id=9, endpoint="http://jev.internal:8010")
        provider.deleted_at = 1

        async def one_by_id(cls, session, provider_id, for_update=False):
            return provider

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        with pytest.raises(ValueError, match="decision-service"):
            await decision_service_plugin.on_route_write(
                "update",
                _route(),
                {
                    "providerId": 9,
                    "decisionModel": "jev-latest",
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
                "decisionModel": "jev-latest",
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
            config={"enabled": True, "providerId": 7, "decisionModel": "jev-latest"},
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
                "decisionModel": "jev-latest",
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            },
        )

        async def one_by_id(cls, session, provider_id, for_update=False):
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

    @pytest.mark.asyncio
    async def test_legacy_section_without_decision_model_degrades_to_inert(
        self, monkeypatch
    ):
        # Rows written before decisionModel became required carry none
        # (exclude_none dropped it at write time). The read path keeps the
        # old precedence -- the provider's own model -- and degrades to
        # inert instead of raising out of reconciliation.
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

        def one_by_id_with_model(model):
            provider = _provider(
                provider_id=7,
                endpoint="http://jev.internal:8010",
                model=model,
            )

            async def one_by_id(cls, sess, provider_id):
                return provider

            return classmethod(one_by_id)

        # provider default backfills the missing decisionModel
        monkeypatch.setattr(
            ModelProvider, "one_by_id", one_by_id_with_model("jev-latest")
        )
        assert await decision_service_plugin.is_effective_on(_route(), object())

        # no provider default either: the policy is inert, not broken
        monkeypatch.setattr(ModelProvider, "one_by_id", one_by_id_with_model(None))
        assert not await decision_service_plugin.is_effective_on(_route(), object())

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
                "apiTokens": ["tenant-a-key"],
                "clusters": ["outbound|8010||provider-2.dns"],
            }
        ]

    def test_provider_entry_carries_full_token_list_for_failover(self):
        # All of the provider's api_tokens ship as the canonical apiTokens
        # list so the wasm plugin can retry the decision callout with the
        # next key on 401/403/429; only the first key must not be emitted.
        entries = decision_provider_entries(
            [
                _provider(
                    provider_id=4,
                    endpoint="http://jev.tenant-a.internal:8010",
                    api_tokens=["primary-key", "fallback-key"],
                )
            ]
        )
        assert entries[0]["apiTokens"] == ["primary-key", "fallback-key"]
        assert "apiToken" not in entries[0]

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
                "apiTokens": ["k"],
                "clusters": ["outbound|443||provider-3.dns"],
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


class TestScopedCatalogueMerge:
    """Per-entry catalogue maintenance: a reconcile owns exactly its entry
    id, merges it into the live providers list, and reports whether
    anything changed — the gate for the route re-notification."""

    def _spec(self, providers):
        return WasmPluginSpec(
            phase="AUTHN",
            priority=328,
            url="http://plugins/gpustack-lb-decision-service/0.2.1/plugin.wasm",
            sha256="abc123",
            defaultConfig={"enabled": False, "providers": list(providers)},
            matchRules=[
                WasmPluginMatchRule(
                    ingress=["ns/ai-route-route-1.internal"], configDisable=False
                )
            ],
        )

    def test_replace_keeps_every_other_entry(self):
        # The scoped sync owns its entry id only: the static half
        # (url/sha256/phase/priority), the matchRules and every other
        # provider entry carry over — a rebuild from defaultConfig alone
        # would strip the module URL and leave Envoy nothing to fetch.
        from gpustack.routes.plugins.decision_service.providers import (
            merge_decision_catalogue_entry,
        )

        current = self._spec(
            [
                {"id": "provider-2", "type": "systemone"},
                {"id": "gpustack-model-12", "clusters": ["old"]},
            ]
        )
        merged, changed = merge_decision_catalogue_entry(
            current,
            "gpustack-model-12",
            {"id": "gpustack-model-12", "clusters": ["new"]},
        )
        assert changed is True
        assert merged.url == current.url
        assert merged.sha256 == current.sha256
        assert merged.phase == current.phase
        assert merged.priority == current.priority
        assert merged.matchRules == current.matchRules
        assert merged.defaultConfig["providers"] == [
            {"id": "provider-2", "type": "systemone"},
            {"id": "gpustack-model-12", "clusters": ["new"]},
        ]
        assert "activeProviderId" not in merged.defaultConfig

    def test_removal_and_absent_no_op_report_change(self):
        from gpustack.routes.plugins.decision_service.providers import (
            merge_decision_catalogue_entry,
        )

        removed, changed = merge_decision_catalogue_entry(
            self._spec([{"id": "gpustack-model-12"}]), "gpustack-model-12", None
        )
        assert changed is True
        assert removed.defaultConfig["providers"] == []

        # Removing an id the catalogue never held is a no-op — what keeps
        # an id-only model DELETE at one entry instead of a whole-catalogue
        # rebuild over the surviving rows.
        _, unchanged = merge_decision_catalogue_entry(
            self._spec([{"id": "provider-2"}]), "gpustack-model-12", None
        )
        assert unchanged is False

    def test_identical_entry_reports_unchanged(self):
        # The frequent ready-replica ticks re-render a byte-identical
        # entry; reporting it unchanged is what stops the per-tick route
        # re-notification churn.
        from gpustack.routes.plugins.decision_service.providers import (
            merge_decision_catalogue_entry,
        )

        current = self._spec([{"id": "gpustack-model-12", "clusters": ["c"]}])
        _, changed = merge_decision_catalogue_entry(
            current,
            "gpustack-model-12",
            {"id": "gpustack-model-12", "clusters": ["c"]},
        )
        assert changed is False

    def test_merges_converge_on_the_full_render_order(self):
        # Provider entries before model entries, each by numeric id, so a
        # scoped write and the route plugin's full-render recreate base
        # agree on the list instead of fighting over its order.
        from gpustack.routes.plugins.decision_service.providers import (
            merge_decision_catalogue_entry,
        )

        current = self._spec(
            [
                {"id": "gpustack-model-12"},
                {"id": "provider-10"},
                {"id": "provider-2"},
            ]
        )
        merged, _ = merge_decision_catalogue_entry(
            current, "gpustack-model-9", {"id": "gpustack-model-9"}
        )
        assert [e["id"] for e in merged.defaultConfig["providers"]] == [
            "provider-2",
            "provider-10",
            "gpustack-model-9",
            "gpustack-model-12",
        ]


class TestScopedEntrySync:
    """The scoped syncs read only their own row's data and merge only their
    own entry id — the ai-proxy per-deployment discipline, without the
    whole-catalogue loads."""

    def _cfg(self, monkeypatch):
        # The module gate reads the real plugin manifest; it is not what
        # these tests exercise, so it is forced on and the cfg carries only
        # the namespace the ensure call reads.
        from types import SimpleNamespace

        from gpustack.routes.plugins.decision_service import providers as providers_mod

        monkeypatch.setattr(
            providers_mod, "decision_module_available", lambda cfg: True
        )
        return SimpleNamespace(gateway_namespace="higress-system")

    def _capture_ensure(self, monkeypatch, captured, providers):
        import gpustack.gateway.utils as gateway_utils

        async def fake_ensure(api, name, namespace, spec_diff, extra_labels=None):
            current = WasmPluginSpec(
                phase="AUTHN",
                priority=328,
                url="http://x/plugin.wasm",
                defaultConfig={"enabled": False, "providers": list(providers)},
                matchRules=[],
            )
            captured["spec"] = spec_diff(current)

        monkeypatch.setattr(gateway_utils, "ensure_wasm_plugin", fake_ensure)

    @pytest.mark.asyncio
    async def test_model_entry_sync_merges_its_own_id(self, monkeypatch):
        from gpustack.routes.plugins.decision_service import providers as providers_mod
        from gpustack.schemas.clusters import Cluster as ClusterRow
        from gpustack.schemas.workers import Worker

        model = Model(id=12, name="d3-flash", categories=["decision"], cluster_id=1)
        served = ModelInstance(
            id=35,
            model_id=12,
            name="model-instance-35",
            worker_ip="192.168.1.20",
            port=40201,
            state=ModelInstanceStateEnum.RUNNING,
        )

        async def model_one(cls, session, model_id, for_update=False):
            return model

        async def instances(cls, session, fields=None, extra_conditions=None, **kw):
            return [served]

        async def workers(cls, session, fields=None, extra_conditions=None, **kw):
            return []

        async def clusters(cls, session, fields=None, extra_conditions=None, **kw):
            return [ClusterRow(id=1, name="c1", registration_token="reg-token")]

        monkeypatch.setattr(Model, "one_by_id", classmethod(model_one))
        monkeypatch.setattr(ModelInstance, "all_by_fields", classmethod(instances))
        monkeypatch.setattr(Worker, "all_by_fields", classmethod(workers))
        monkeypatch.setattr(ClusterRow, "all_by_fields", classmethod(clusters))

        captured = {}
        self._capture_ensure(
            monkeypatch, captured, [{"id": "provider-2", "type": "systemone"}]
        )

        changed = await providers_mod.sync_decision_model_entry(
            cfg=self._cfg(monkeypatch), session=None, extensions_api=None, model_id=12
        )
        assert changed is True
        providers = captured["spec"].defaultConfig["providers"]
        assert [e["id"] for e in providers] == ["provider-2", "gpustack-model-12"]
        assert providers[1]["apiTokens"] == ["reg-token"]

    @pytest.mark.asyncio
    async def test_remote_cluster_entry_rides_the_cluster_gateway(self, monkeypatch):
        # A remote deployment's callouts ride the cluster gateway entrypoint
        # — the same destination its inference traffic is routed through —
        # not the workers' unreachable private addresses, and authenticate
        # with the cluster's registration token.
        from gpustack.routes.plugins.decision_service import providers as providers_mod
        from gpustack.schemas.clusters import Cluster as ClusterRow
        from gpustack.schemas.users import Principal, PrincipalType

        model = Model(id=12, name="d3-remote", categories=["decision"], cluster_id=5)
        cluster = ClusterRow(
            id=5,
            name="remote",
            system_principal_id=9,
            gateway_endpoint="10.9.8.7:9090",
            registration_token="remote-reg-token",
        )
        principal = Principal(id=9, name="remote-cluster", kind=PrincipalType.SYSTEM)

        async def model_one(cls, session, model_id, for_update=False):
            return model

        async def clusters_all(cls, session, fields=None, extra_conditions=None, **kw):
            return [cluster]

        async def principals_all(
            cls, session, fields=None, extra_conditions=None, **kw
        ):
            return [principal]

        monkeypatch.setattr(Model, "one_by_id", classmethod(model_one))
        monkeypatch.setattr(ClusterRow, "all_by_fields", classmethod(clusters_all))
        monkeypatch.setattr(Principal, "all_by_fields", classmethod(principals_all))

        entry = await providers_mod.render_decision_model_entry(None, 12)
        assert entry == {
            "id": "gpustack-model-12",
            "type": "systemone",
            "endpoint": "http://10.9.8.7:9090",
            "model": "d3-remote",
            "clusters": ["outbound|80||cluster-gateway.static"],
            "apiTokens": ["remote-reg-token"],
        }

    @pytest.mark.asyncio
    async def test_gone_model_merges_to_removal(self, monkeypatch):
        # An id-only DELETE: the row is gone, so the entry merges to
        # removal — no decision-model knowledge needed up front.
        from gpustack.routes.plugins.decision_service import providers as providers_mod

        async def model_one(cls, session, model_id, for_update=False):
            return None

        monkeypatch.setattr(Model, "one_by_id", classmethod(model_one))

        captured = {}
        self._capture_ensure(
            monkeypatch,
            captured,
            [{"id": "gpustack-model-12", "clusters": ["dead"]}, {"id": "provider-2"}],
        )

        changed = await providers_mod.sync_decision_model_entry(
            cfg=self._cfg(monkeypatch), session=None, extensions_api=None, model_id=12
        )
        assert changed is True
        assert [e["id"] for e in captured["spec"].defaultConfig["providers"]] == [
            "provider-2"
        ]


class TestRenderAllEntries:
    """The recreate base's full render: bounded queries (one batch per
    data source, however many decision models exist) with each entry
    rendered through the same helpers the scoped syncs use."""

    def _install(self, monkeypatch):
        from gpustack.schemas.clusters import Cluster as ClusterRow
        from gpustack.schemas.users import Principal, PrincipalType
        from gpustack.schemas.workers import Worker

        local = Model(id=12, name="d3-local", categories=["decision"])
        remote = Model(id=13, name="d3-remote", categories=["decision"], cluster_id=5)
        other = Model(id=14, name="qwen", categories=["llm"])
        served = ModelInstance(
            id=35,
            model_id=12,
            name="model-instance-35",
            worker_ip="192.168.1.20",
            port=40201,
            state=ModelInstanceStateEnum.RUNNING,
        )
        remote_cluster = ClusterRow(
            id=5,
            name="remote",
            system_principal_id=9,
            gateway_endpoint="10.9.8.7:9090",
            registration_token="remote-reg-token",
        )
        local_cluster = ClusterRow(id=6, name="default", registration_token="tok")
        principal = Principal(id=9, name="remote-cluster", kind=PrincipalType.SYSTEM)
        calls = {"count": 0}

        def _tick():
            calls["count"] += 1

        async def models_all(cls, session, field, value):
            _tick()
            return [local, remote, other]

        async def instances_all(cls, session, fields=None, extra_conditions=None, **kw):
            _tick()
            return [served]

        async def workers_all(cls, session, fields=None, extra_conditions=None, **kw):
            _tick()
            return []

        async def clusters_all(cls, session, fields=None, extra_conditions=None, **kw):
            _tick()
            return [remote_cluster, local_cluster]

        async def principals_all(
            cls, session, fields=None, extra_conditions=None, **kw
        ):
            _tick()
            return [principal]

        monkeypatch.setattr(Model, "all_by_field", classmethod(models_all))
        monkeypatch.setattr(ModelInstance, "all_by_fields", classmethod(instances_all))
        monkeypatch.setattr(Worker, "all_by_fields", classmethod(workers_all))
        monkeypatch.setattr(ClusterRow, "all_by_fields", classmethod(clusters_all))
        monkeypatch.setattr(Principal, "all_by_fields", classmethod(principals_all))
        return calls

    @pytest.mark.asyncio
    async def test_mixed_local_and_remote_in_bounded_queries(self, monkeypatch):
        from gpustack.routes.plugins.decision_service.providers import (
            render_all_decision_model_entries,
        )

        calls = self._install(monkeypatch)
        entries = await render_all_decision_model_entries(None)
        assert [e["id"] for e in entries] == ["gpustack-model-12", "gpustack-model-13"]
        # The local deployment rides its instance registry; the remote one
        # the cluster gateway with the registration token.
        assert entries[0]["clusters"] == ["outbound|80||model-12-35.static"]
        assert "apiTokens" not in entries[0]
        assert entries[1]["clusters"] == ["outbound|80||cluster-gateway.static"]
        assert entries[1]["apiTokens"] == ["remote-reg-token"]
        # One batch per data source regardless of the decision model count:
        # models, clusters + principals (one context read), instances. No
        # worker read here — these instances carry no worker_id — and 100
        # decision models would still be 4-5 reads. The remote model's
        # instances are not fetched at all: its entry renders the gateway.
        assert calls["count"] == 4

    @pytest.mark.asyncio
    async def test_default_cluster_principal_is_not_a_gateway(self, monkeypatch):
        # The local (default) cluster's principal is not a remote gateway:
        # its token still resolves (worker-proxy auth) but no registry
        # does, so its deployments render through their instances.
        from gpustack.routes.plugins.decision_service.providers import (
            decision_cluster_context,
        )
        from gpustack.schemas.clusters import Cluster as ClusterRow
        from gpustack.schemas.users import (
            Principal,
            PrincipalType,
            default_cluster_principal_name,
        )

        default_cluster = ClusterRow(
            id=6, name="default", system_principal_id=8, registration_token="tok"
        )
        default_principal = Principal(
            id=8, name=default_cluster_principal_name, kind=PrincipalType.SYSTEM
        )

        async def clusters_all(cls, session, fields=None, extra_conditions=None, **kw):
            return [default_cluster]

        async def principals_all(
            cls, session, fields=None, extra_conditions=None, **kw
        ):
            return [default_principal]

        monkeypatch.setattr(ClusterRow, "all_by_fields", classmethod(clusters_all))
        monkeypatch.setattr(Principal, "all_by_fields", classmethod(principals_all))
        tokens, registries = await decision_cluster_context(None, {6})
        assert tokens == {6: ["tok"]}
        assert registries == {}


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
        controller._decision_notify_owed = set()
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


class TestLocalDecisionModelConfig:
    """The modelId half of the section: exactly-one target, its catalogue
    id, and the route-id lookup a model event uses to find its routes."""

    def test_model_id_replaces_provider_id(self):
        config = DecisionServiceRouteConfig.model_validate(
            {
                "modelId": 12,
                "decisionModel": "d3-flash",
                "modelSelection": {"criteria": {"m1": "fast", "m2": "smart"}},
            }
        )
        rule = config.to_gateway_rule()
        assert rule["activeProviderId"] == "gpustack-model-12"

    def test_both_targets_rejected(self):
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate(
                {
                    "providerId": 7,
                    "modelId": 12,
                    "decisionModel": "jev-latest",
                }
            )

    def test_neither_target_rejected(self):
        with pytest.raises(ValidationError):
            DecisionServiceRouteConfig.model_validate({"decisionModel": "jev-latest"})

    def test_legacy_provider_only_section_still_valid(self):
        config = DecisionServiceRouteConfig.model_validate(
            {"providerId": 7, "decisionModel": "jev-latest"}
        )
        assert config.providerId == 7
        assert config.modelId is None
        assert config.to_gateway_rule()["activeProviderId"] == "provider-7"

    def test_model_referencing_route_ids_selects_policies_pointing_at_model(self):
        from gpustack.routes.plugins.decision_service.providers import (
            decision_model_referencing_route_ids,
        )

        policies = [
            CapabilityPolicy(
                capability="decision-service",
                route_id=1,
                config={"modelId": 12, "enabled": True},
            ),
            CapabilityPolicy(
                capability="decision-service",
                route_id=2,
                config={"providerId": 8, "enabled": True},
            ),
            CapabilityPolicy(
                capability="decision-service",
                route_id=3,
                config={"modelId": 13, "enabled": True},
            ),
        ]
        assert decision_model_referencing_route_ids(policies, 12) == {1}
        assert decision_model_referencing_route_ids(policies, 13) == {3}
        assert decision_model_referencing_route_ids(policies, 8) == set()
        assert decision_model_referencing_route_ids(None, 12) == set()


class TestLocalDecisionModelEntries:
    """The locally deployed half of the CR catalogue: entries are the
    deployment's running instance, absent without one."""

    def _model(self, model_id=12, categories=("decision",)):
        return Model(
            id=model_id,
            name="d3-flash",
            categories=list(categories),
        )

    def _instance(
        self,
        instance_id=35,
        model_id=12,
        state=ModelInstanceStateEnum.RUNNING,
        address="192.168.1.20",
        port=40201,
    ):
        return ModelInstance(
            id=instance_id,
            model_id=model_id,
            name=f"model-instance-{instance_id}",
            worker_ip=address,
            port=port,
            state=state,
        )

    def test_entry_points_at_running_instance_registry(self):
        # The catalogue entry is the deployment's instances: every servable
        # RUNNING one contributes a cluster to the list the plugin spreads
        # callouts across; the endpoint names the first instance for the
        # callout's :authority.
        entries = decision_model_entries([self._model()], [self._instance()])
        assert entries == [
            {
                "id": "gpustack-model-12",
                "type": "systemone",
                "endpoint": "http://192.168.1.20:40201",
                "model": "d3-flash",
                "clusters": ["outbound|80||model-12-35.static"],
                "clusterNameHeader": "X-GPUStack-Model-Instance",
            }
        ]

    def test_cluster_name_header_selects_the_worker_proxy_member(self):
        # The worker proxy picks the member by the cluster-name header;
        # naming the header (not a per-cluster value mapping) is all the
        # entry needs — the plugin derives the value from the cluster it
        # picks per callout, so it follows the pick by construction.
        from gpustack.schemas.config import ModelInstanceProxyModeEnum
        from gpustack.schemas.workers import Worker

        worker = Worker(
            id=77,
            name="worker-77",
            ip="10.0.0.77",
            port=10176,
            advertise_address="worker-77.internal",
            proxy_mode=ModelInstanceProxyModeEnum.WORKER,
        )
        first = self._instance(instance_id=35, port=40201)
        first.worker_id = 77
        second = self._instance(instance_id=41, port=40202)
        second.worker_id = 77

        entry = decision_model_entry(self._model(), [first, second], {77: worker})
        assert entry["clusterNameHeader"] == "X-GPUStack-Model-Instance"
        assert entry["clusters"] == [
            "outbound|10176||model-12-35.dns",
            "outbound|10176||model-12-41.dns",
        ]

    def test_model_without_running_instance_is_absent(self):
        # No servable RUNNING member means no catalogue entry, so routes
        # selecting the model fail open (rule stripped) instead of calling
        # out to a dead cluster.
        assert decision_model_entries([self._model()], []) == []
        assert (
            decision_model_entries(
                [self._model()],
                [self._instance(state=ModelInstanceStateEnum.ERROR)],
            )
            == []
        )

    def test_every_instance_is_a_cluster_in_id_order(self):
        # All replicas ride one entry; the plugin picks one cluster
        # uniformly at random per callout, so the decision traffic
        # spreads across the deployment.
        entries = decision_model_entries(
            [self._model()],
            [
                self._instance(instance_id=41, port=40202),
                self._instance(instance_id=35, port=40201),
            ],
        )
        assert entries[0]["clusters"] == [
            "outbound|80||model-12-35.static",
            "outbound|80||model-12-41.static",
        ]
        # The endpoint names the first (lowest-id) instance for the
        # callout's :authority; the plugin spreads callouts over the
        # clusters.
        assert entries[0]["endpoint"] == "http://192.168.1.20:40201"

    def test_entry_carries_cluster_registration_token(self):
        # The entry's clusters may name worker-proxy entrypoints that
        # authenticate with the cluster's registration token — the same
        # credential ai-proxy attaches for the deployment's inference
        # traffic. The plugin ships it as the callout's Authorization and
        # retries with the next entry on 401/403/429.
        entries = decision_model_entries(
            [self._model()],
            [self._instance()],
            api_tokens_by_cluster_id={None: ["cluster-reg-token"]},
        )
        assert entries[0]["apiTokens"] == ["cluster-reg-token"]

    def test_entry_without_cluster_token_is_anonymous(self):
        # A cluster without a registration token renders no apiTokens: the
        # anonymous callout a DIRECT instance (nothing between the gateway
        # and the inference server) already accepts.
        entries = decision_model_entries(
            [self._model()],
            [self._instance()],
            api_tokens_by_cluster_id={},
        )
        assert "apiTokens" not in entries[0]

    def test_non_decision_model_excluded(self):
        assert (
            decision_model_entries(
                [self._model(categories=("llm",))], [self._instance()]
            )
            == []
        )

    def test_model_and_provider_entries_coexist(self):
        # Both halves of the catalogue in one render: providers keep their
        # provider-{id} ids, models the gpustack-model-{id} ones, so a
        # route's activeProviderId can never cross the two namespaces.
        entries = decision_provider_entries(
            [_provider(provider_id=7, endpoint="http://jev.internal:8010")]
        ) + decision_model_entries([self._model()], [self._instance()])
        assert [entry["id"] for entry in entries] == [
            "provider-7",
            "gpustack-model-12",
        ]

    def test_worker_aware_cluster_matches_gateway_registration(self):
        # The gateway registers each instance's registry with its worker:
        # under the WORKER proxy mode the registered address is the
        # worker's, so the catalogue must resolve the same worker-aware
        # registry. Worker-blind, the cluster encodes the instance's
        # address form and matches no registration in the bridge.
        from gpustack.schemas.config import ModelInstanceProxyModeEnum
        from gpustack.schemas.workers import Worker

        worker = Worker(
            id=77,
            name="worker-77",
            ip="10.0.0.77",
            port=10176,
            advertise_address="worker-77.internal",
            proxy_mode=ModelInstanceProxyModeEnum.WORKER,
        )
        unassigned = self._instance()
        assigned = self._instance()
        assigned.worker_id = 77

        blind = decision_model_entry(self._model(), [unassigned])
        aware = decision_model_entry(self._model(), [assigned], {77: worker})
        assert blind["clusters"] == ["outbound|80||model-12-35.static"]
        assert aware["clusters"] == ["outbound|10176||model-12-35.dns"]

    def test_tunnel_worker_without_proxy_address_yields_no_entry(self):
        # The tunnel has not connected (no proxy address): the gateway
        # registers nothing for the instance, so the catalogue must render
        # no cluster for it either — a worker-blind registry would point
        # the callout straight at an unreachable instance address.
        from gpustack.schemas.config import ModelInstanceProxyModeEnum
        from gpustack.schemas.workers import Worker

        worker = Worker(
            id=78,
            name="worker-78",
            ip="10.0.0.78",
            port=10177,
            proxy_mode=ModelInstanceProxyModeEnum.TUNNEL,
        )
        instance = self._instance()
        instance.worker_id = 78

        assert decision_model_entry(self._model(), [instance], {78: worker}) is None

    def test_instance_with_gone_worker_row_is_skipped(self):
        # A stale instance (worker_id set, worker row gone) renders no
        # cluster: the worker-blind fallback would name the dead instance
        # address, and under the WORKER/TUNNEL proxy modes a registry the
        # bridge never registered. Surviving members still render.
        from gpustack.schemas.config import ModelInstanceProxyModeEnum
        from gpustack.schemas.workers import Worker

        live_worker = Worker(
            id=77,
            name="worker-77",
            ip="10.0.0.77",
            port=10176,
            advertise_address="worker-77.internal",
            proxy_mode=ModelInstanceProxyModeEnum.WORKER,
        )
        live = self._instance(instance_id=35)
        live.worker_id = 77
        stale = self._instance(instance_id=41, address="10.0.0.9")
        stale.worker_id = 99

        entry = decision_model_entry(self._model(), [live, stale], {77: live_worker})
        assert entry["clusters"] == ["outbound|10176||model-12-35.dns"]
        only_stale = self._instance(instance_id=41)
        only_stale.worker_id = 99
        assert decision_model_entry(self._model(), [only_stale], {}) is None


class TestDecisionClusterContext:
    """One Cluster read carrying both halves of the cluster context: the
    registration token (worker-proxy entrypoints and a remote gateway
    entry authenticate with it) and the remote clusters' gateway
    entrypoints — the local (default-principal) cluster renders none."""

    @pytest.mark.asyncio
    async def test_tokens_and_gateways_in_one_read(self, monkeypatch):
        from gpustack.routes.plugins.decision_service.providers import (
            decision_cluster_context,
        )
        from gpustack.schemas.clusters import Cluster
        from gpustack.schemas.users import Principal, PrincipalType

        remote = Cluster(
            id=1,
            name="remote",
            system_principal_id=9,
            gateway_endpoint="10.9.8.7:9090",
            registration_token="token-1",
        )
        tokenless = Cluster(id=2, name="c2", registration_token=None)
        principal = Principal(id=9, name="remote-cluster", kind=PrincipalType.SYSTEM)

        async def clusters_all(cls, session, fields=None, extra_conditions=None, **kw):
            return [remote, tokenless]

        async def principals_all(
            cls, session, fields=None, extra_conditions=None, **kw
        ):
            return [principal]

        monkeypatch.setattr(Cluster, "all_by_fields", classmethod(clusters_all))
        monkeypatch.setattr(Principal, "all_by_fields", classmethod(principals_all))
        tokens, registries = await decision_cluster_context(None, {1, 2, None})
        # Tokenless clusters map to no token (their entries render
        # anonymous); only the remote cluster yields a gateway entrypoint.
        assert tokens == {1: ["token-1"]}
        assert set(registries) == {1}
        assert registries[1].get_service_name() == "cluster-gateway.static"

    @pytest.mark.asyncio
    async def test_no_cluster_ids_queries_nothing(self, monkeypatch):
        from gpustack.routes.plugins.decision_service.providers import (
            decision_cluster_context,
        )
        from gpustack.schemas.clusters import Cluster

        async def all_by_fields(cls, session, fields=None, extra_conditions=None, **kw):
            raise AssertionError("no cluster to read")

        monkeypatch.setattr(Cluster, "all_by_fields", classmethod(all_by_fields))
        assert await decision_cluster_context(None, set()) == ({}, {})
        assert await decision_cluster_context(None, {None}) == ({}, {})


class TestInstanceWorkers:
    @pytest.mark.asyncio
    async def test_soft_deleted_workers_do_not_resolve(self, monkeypatch):
        # A soft-deleted worker is a gone worker: resolving it would render
        # a cluster against a bridge registration that no longer exists,
        # defeating the stale-instance skip in decision_model_entry.
        from gpustack.routes.plugins.decision_service.providers import (
            instance_workers_by_id,
        )
        from gpustack.schemas.workers import Worker

        captured = {}

        async def all_by_fields(cls, session, fields=None, extra_conditions=None, **kw):
            captured["conditions"] = extra_conditions
            return [Worker(id=7, name="w7", ip="10.0.0.7", port=10176)]

        monkeypatch.setattr(Worker, "all_by_fields", classmethod(all_by_fields))
        instance = ModelInstance(id=1, model_id=2, name="mi-1", worker_id=7)
        by_id = await instance_workers_by_id(None, [instance])
        assert set(by_id) == {7}
        assert any(
            condition.compare(Worker.deleted_at.is_(None))
            for condition in captured["conditions"]
        )


class TestDecisionTargetResolvesParity:
    """``_decision_target_resolves`` must answer exactly what the catalogue
    renders: a RUNNING member that cannot form an entry (no address/port,
    or a registry the gateway never registered) still leaves the route's
    activeProviderId pointing at an entry that does not exist."""

    def _config(self):
        return DecisionServiceRouteConfig.model_validate(
            {
                "modelId": 12,
                "decisionModel": "d3-flash",
                "modelSelection": {"criteria": {"m1": "fast"}},
            }
        )

    def _install(self, monkeypatch, model, instances):
        async def one_by_id(cls, session, model_id, for_update=False):
            return model

        async def all_by_fields(cls, session, fields=None, extra_conditions=None, **kw):
            return list(instances)

        monkeypatch.setattr(Model, "one_by_id", classmethod(one_by_id))
        monkeypatch.setattr(ModelInstance, "all_by_fields", classmethod(all_by_fields))

    @pytest.mark.asyncio
    async def test_running_member_without_port_is_not_resolving(self, monkeypatch):
        from gpustack.routes.plugins.decision_service.plugin import (
            _decision_target_resolves,
        )

        model = Model(id=12, name="d3-flash", categories=["decision"])
        bare = ModelInstance(
            id=35,
            model_id=12,
            name="model-instance-35",
            worker_ip=None,
            port=None,
            state=ModelInstanceStateEnum.RUNNING,
        )
        self._install(monkeypatch, model, [bare])
        assert not await _decision_target_resolves(object(), self._config())

    @pytest.mark.asyncio
    async def test_renderable_member_resolves(self, monkeypatch):
        from gpustack.routes.plugins.decision_service.plugin import (
            _decision_target_resolves,
        )

        model = Model(id=12, name="d3-flash", categories=["decision"])
        served = ModelInstance(
            id=35,
            model_id=12,
            name="model-instance-35",
            worker_ip="192.168.1.20",
            port=40201,
            state=ModelInstanceStateEnum.RUNNING,
        )
        self._install(monkeypatch, model, [served])
        assert await _decision_target_resolves(object(), self._config())

    @pytest.mark.asyncio
    async def test_reassigned_owner_is_not_resolving(self, monkeypatch):
        # The write-time check pins tenant alignment, but ownership can
        # move after the fact: the render gate re-checks it so a route
        # never keeps pointing its callout at a deployment another Org
        # now owns.
        from gpustack.routes.plugins.decision_service.plugin import (
            _decision_target_resolves,
        )

        model = Model(id=12, name="d3-flash", categories=["decision"])
        model.owner_principal_id = 42
        served = ModelInstance(
            id=35,
            model_id=12,
            name="model-instance-35",
            worker_ip="192.168.1.20",
            port=40201,
            state=ModelInstanceStateEnum.RUNNING,
        )
        self._install(monkeypatch, model, [served])
        assert not await _decision_target_resolves(
            object(), self._config(), route_owner_principal_id=7
        )
        # an ownerless (platform) route and the same Org both stay
        # resolving
        assert await _decision_target_resolves(
            object(), self._config(), route_owner_principal_id=None
        )
        assert await _decision_target_resolves(
            object(), self._config(), route_owner_principal_id=42
        )

    @pytest.mark.asyncio
    async def test_reassigned_provider_owner_is_not_resolving(self, monkeypatch):
        # The external target has the same exposure as the local one: the
        # write-time alignment check can be invalidated by a later owner
        # move, and a provider owned by another Org carries its endpoint
        # and token on the shared CR.
        from gpustack.routes.plugins.decision_service.plugin import (
            _decision_target_resolves,
        )

        provider = _provider(provider_id=7, endpoint="http://jev.internal:8010")
        provider.owner_principal_id = 42

        async def one_by_id(cls, session, provider_id, for_update=False):
            return provider

        monkeypatch.setattr(ModelProvider, "one_by_id", classmethod(one_by_id))
        config = DecisionServiceRouteConfig.model_validate(
            {
                "providerId": 7,
                "decisionModel": "jev-latest",
                "modelSelection": {"criteria": {"m1": "fast"}},
            }
        )
        assert not await _decision_target_resolves(
            object(), config, route_owner_principal_id=7
        )
        assert await _decision_target_resolves(
            object(), config, route_owner_principal_id=42
        )
        assert await _decision_target_resolves(
            object(), config, route_owner_principal_id=None
        )


class TestDecisionCatalogueAffects:
    """The catalogue gate reads the old categories side of both change
    producers: find_history wraps the whole old value in a 1-tuple,
    detect_changes stores the per-element removals. A decode that
    understands only one shape misses the other's decision removals and
    leaves a stale catalogue entry behind."""

    def _affects(self, categories, change):
        from gpustack.server.controllers import ModelController
        from gpustack.server.coordinator.base import Event, EventType

        controller = ModelController.__new__(ModelController)
        model = Model(id=12, name="d3", categories=list(categories))
        event = Event(
            type=EventType.UPDATED,
            data=None,
            id=12,
            changed_fields={"categories": change},
        )
        return controller._affects_decision_catalogue(model, event)

    @pytest.mark.parametrize(
        "change,expected",
        [
            # find_history: the whole old value wrapped in a 1-tuple
            ((["decision", "llm"], ["llm"]), True),
            ((["llm"], ["reranker"]), False),
            # detect_changes with keyed elements: per-element removals
            ((["decision"], []), True),
            ((["llm"], ["decision"]), False),
            # a flat scalar pair, as the producers emit for scalar columns
            (("decision", "llm"), True),
            # the empty old side is ambiguous: detect_changes emits it for
            # keyless lists (categories) whose content changed but cannot
            # be attributed, find_history for a None-side transition —
            # rebuild rather than guess, the addition case is covered by
            # the is_decision_model check above.
            (((), (["decision"],)), True),
            (([], []), True),
        ],
        ids=[
            "history-removed",
            "history-unrelated",
            "element-diff-removed",
            "element-diff-unrelated",
            "scalar-pair",
            "empty-side-history",
            "empty-side-keyless-diff",
        ],
    )
    def test_both_change_shapes_read_the_old_side(self, change, expected):
        assert self._affects(("llm",), change) is expected


class TestIdOnlyDeleteEntrySync:
    """A cross-instance DELETE whose row is gone still reconciles the
    model's own catalogue entry: removal by id needs no decision-model
    knowledge (an id the catalogue never held merges to a no-op), and the
    routes pinning the id are re-reconciled — BEFORE the CR write, gated
    on the rendered entry differing from the last render. A restart
    forgets the render cache, so the first event per model re-notifies
    once; a notification failure aborts before the write and the workqueue
    retry re-runs both halves."""

    def _controller(self):
        from gpustack.server.controllers import ModelController

        controller = ModelController.__new__(ModelController)
        controller._disable_gateway = False
        controller._config = None
        controller._higress_extension_api = None
        controller._decision_entry_cache = {}
        return controller

    def _install(self, monkeypatch, effects, entry=None, notify_error=None):
        from contextlib import asynccontextmanager

        import gpustack.server.controllers as controllers_mod

        async def _render(session, model_id):
            effects.append(("render", model_id))
            return entry

        async def _ensure(cfg, extensions_api, model_id, entry):
            effects.append(("ensure", model_id, entry))

        class _Policies:
            @staticmethod
            async def all_by_fields(session, fields=None, **kw):
                return [object()]

        class _Bus:
            @staticmethod
            async def publish(topic, event):
                if notify_error is not None:
                    raise notify_error
                effects.append((topic, event.type))

        @asynccontextmanager
        async def _session():
            yield None

        async def one_by_id(cls, session, route_id, **kw):
            if route_id != 3:
                return None
            return ModelRoute(id=3, name="d3-route", targets=0, ready_targets=0)

        monkeypatch.setattr(controllers_mod, "async_session", _session)
        monkeypatch.setattr(controllers_mod, "render_decision_model_entry", _render)
        monkeypatch.setattr(controllers_mod, "ensure_decision_model_entry", _ensure)
        monkeypatch.setattr(controllers_mod, "CapabilityPolicy", _Policies)
        monkeypatch.setattr(
            controllers_mod,
            "decision_model_referencing_route_ids",
            lambda policies, model_id: {3} if model_id == 4245 else set(),
        )
        monkeypatch.setattr(controllers_mod, "event_bus", _Bus)
        monkeypatch.setattr(ModelRoute, "one_by_id", classmethod(one_by_id))

    def _event(self, event_type):
        from gpustack.server.coordinator.base import Event

        event = Event(type=event_type, data=None)
        event.id = 4245
        return event

    @pytest.mark.asyncio
    async def test_entry_removal_notifies_then_ensures(self, monkeypatch):
        # The deleted model's entry was rendered before (cache holds it),
        # so the routes pinning it are re-reconciled BEFORE the entry is
        # removed from the CR.
        from gpustack.server.coordinator.base import EventType

        effects = []
        self._install(monkeypatch, effects, entry=None)
        controller = self._controller()
        controller._decision_entry_cache = {4245: {"id": "gpustack-model-4245"}}
        await controller._reconcile(self._event(EventType.DELETED))
        assert effects == [
            ("render", 4245),
            ("modelroute", EventType.UPDATED),
            ("ensure", 4245, None),
        ]

    @pytest.mark.asyncio
    async def test_id_only_update_skips_the_sync(self, monkeypatch):
        from gpustack.server.coordinator.base import EventType

        effects = []
        self._install(monkeypatch, effects)
        await self._controller()._reconcile(self._event(EventType.UPDATED))
        assert effects == []

    @pytest.mark.asyncio
    async def test_disabled_gateway_skips_the_sync(self, monkeypatch):
        from gpustack.server.coordinator.base import EventType

        effects = []
        self._install(monkeypatch, effects)
        controller = self._controller()
        controller._disable_gateway = True
        await controller._reconcile(self._event(EventType.DELETED))
        assert effects == []

    @pytest.mark.asyncio
    async def test_unchanged_render_skips_the_notify(self, monkeypatch):
        # The per-tick shape: the entry re-rendered byte-identical, so the
        # routes are not re-reconciled — the churn the render cache exists
        # to stop. The CR merge still runs (its own diff decides the write).
        from gpustack.server.coordinator.base import EventType

        effects = []
        entry = {"id": "gpustack-model-4245", "clusters": ["c"]}
        self._install(monkeypatch, effects, entry=entry)
        controller = self._controller()
        controller._decision_entry_cache = {4245: entry}
        await controller._reconcile(self._event(EventType.DELETED))
        assert effects == [
            ("render", 4245),
            ("ensure", 4245, entry),
        ]

    @pytest.mark.asyncio
    async def test_cold_cache_absent_render_still_notifies(self, monkeypatch):
        # After a restart the cache holds nothing: an absent render (the
        # target deleted or unavailable) must still notify, or the entry
        # removal would leave referencing rules pointing at a catalogue id
        # nothing renders. Key membership, not a .get() None, is the
        # "never seen" signal.
        from gpustack.server.coordinator.base import EventType

        effects = []
        self._install(monkeypatch, effects, entry=None)
        await self._controller()._reconcile(self._event(EventType.DELETED))
        assert effects == [
            ("render", 4245),
            ("modelroute", EventType.UPDATED),
            ("ensure", 4245, None),
        ]

    @pytest.mark.asyncio
    async def test_notify_failure_aborts_before_the_write(self, monkeypatch):
        # Notify-first ordering: a failed notification leaves the CR write
        # undone, so the workqueue retry re-runs both halves instead of
        # stranding a written entry with stale rules.
        from gpustack.server.coordinator.base import EventType

        effects = []
        self._install(
            monkeypatch,
            effects,
            entry={"id": "gpustack-model-4245"},
            notify_error=RuntimeError("bus down"),
        )
        controller = self._controller()
        with pytest.raises(RuntimeError):
            await controller._reconcile(self._event(EventType.DELETED))
        assert effects == [("render", 4245)]
        assert 4245 not in controller._decision_entry_cache
