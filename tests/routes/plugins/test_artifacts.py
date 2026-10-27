"""The declarative rule collector: ownership is declared through the
owner stamp, and one flush folds every writer's declarations into a
single read-modify-write per CR — the two writers on the shared
mapper CR can no longer delete each other's rules, whatever shape
their configs take."""

import pytest

from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
    WasmPluginSpec,
)
from gpustack.routes.plugins.artifacts import (
    RULE_OWNER_KEY,
    RouteArtifactCollector,
    RouteRuleUpdate,
    apply_rule_updates,
)


def _spec(*rules):
    return WasmPluginSpec(matchRules=list(rules))


def _rule(config, ingress, **kw):
    return WasmPluginMatchRule(config=config, ingress=list(ingress), **kw)


def _lb_update(ingresses, config=None, owner="lb"):
    rules = [_rule(config, [ingresses[0]], configDisable=False)] if config else []
    return RouteRuleUpdate(owner=owner, ingresses=list(ingresses), rules=rules)


class _Cfg:
    gateway_namespace = "higress-system"


class TestApplyRuleUpdates:
    def test_other_ingress_rules_survive(self):
        spec = _spec(
            _rule({"modelMapping": {"a": "b"}}, ["ns/other"]),
            _rule({"modelMapping": {"a": "b"}}, ["ns/x"]),
        )
        merged = apply_rule_updates(spec, [_lb_update(["ns/x"], {"candidates": []})])
        ingresses = [r.ingress[0] for r in merged.matchRules]
        assert "ns/other" in ingresses and "ns/x" in ingresses

    def test_replaces_previous_owner_rule_for_same_ingress(self):
        spec = _spec(_rule({RULE_OWNER_KEY: "lb", "candidates": ["stale"]}, ["ns/x"]))
        merged = apply_rule_updates(
            spec, [_lb_update(["ns/x"], {"candidates": ["fresh"]})]
        )
        assert len(merged.matchRules) == 1
        assert merged.matchRules[0].config["candidates"] == ["fresh"]
        assert merged.matchRules[0].config[RULE_OWNER_KEY] == "lb"

    def test_mapper_rules_on_same_ingress_survive(self):
        # The mapper sync's fallback rules dual-attach the main ingress,
        # and the LB rule attaches to it too: same ingresses, different
        # owners. Only the declared owner's rules are recycled — an
        # ingress-based predicate would make the two writers delete
        # each other's rules.
        spec = _spec(
            _rule({"candidates": ["stale"]}, ["ns/x"]),  # legacy, un-stamped
            _rule({"modelMapping": {"a": "b"}}, ["ns/x"]),
            _rule(
                {"modelMapping": {"a": "c"}}, ["ns/x", "ns/x.fallback"], service=["svc"]
            ),
        )
        merged = apply_rule_updates(
            spec, [_lb_update(["ns/x"], {"candidates": ["fresh"]})]
        )
        configs = [r.config for r in merged.matchRules]
        assert {"modelMapping": {"a": "b"}} in configs
        assert {"modelMapping": {"a": "c"}} in configs
        assert configs[-1].get("candidates") == ["fresh"]
        assert {"candidates": ["stale"]} not in configs

    def test_strip_recycles_only_own_rules(self):
        spec = _spec(
            _rule({"candidates": ["x"]}, ["ns/x"]),
            _rule({"modelMapping": {"a": "b"}}, ["ns/x"]),
        )
        merged = apply_rule_updates(spec, [_lb_update(["ns/x"])])
        assert [r.config for r in merged.matchRules] == [{"modelMapping": {"a": "b"}}]

    def test_legacy_unstamped_rules_recycled_by_shape(self):
        # Pre-upgrade rules carry no owner stamp; the two legacy
        # writers' shapes are inferred so an upgrade converges.
        spec = _spec(
            _rule({"candidates": ["old"]}, ["ns/x"]),
            _rule({"modelMapping": {"a": "b"}}, ["ns/x"]),
        )
        merged = apply_rule_updates(
            spec, [_lb_update(["ns/x"], {"candidates": ["new"]})]
        )
        configs = [r.config for r in merged.matchRules]
        assert {"candidates": ["old"]} not in configs
        assert {"modelMapping": {"a": "b"}} in configs

    def test_foreign_owner_never_recycled(self):
        spec = _spec(
            _rule({RULE_OWNER_KEY: "someone-else", "candidates": ["x"]}, ["ns/x"])
        )
        merged = apply_rule_updates(spec, [_lb_update(["ns/x"], {"candidates": ["y"]})])
        configs = [r.config for r in merged.matchRules]
        assert {RULE_OWNER_KEY: "someone-else", "candidates": ["x"]} in configs


class TestFlush:
    def _patch_ensure(self, monkeypatch):
        """Capture (name, spec_diff) per flush call instead of hitting
        a cluster; the diff functions are exercised against synthetic
        live specs."""
        import gpustack.gateway.utils as gateway_utils

        calls = []

        async def fake_ensure(api, name, namespace, spec_diff, extra_labels=None):
            calls.append((name, namespace, spec_diff))

        monkeypatch.setattr(gateway_utils, "ensure_wasm_plugin", fake_ensure)
        return calls

    @pytest.mark.asyncio
    async def test_one_call_per_cr_across_writers(self, monkeypatch):
        calls = self._patch_ensure(monkeypatch)
        collector = RouteArtifactCollector()
        collector.set_rules(
            "cr-a", "mapper", ["ns/x"], [_rule({"modelMapping": {}}, ["ns/x"])]
        )
        collector.set_rules(
            "cr-a",
            "lb",
            ["ns/x"],
            [_rule({"candidates": []}, ["ns/x"], configDisable=False)],
        )
        collector.set_rules("cr-b", "session-affinity", ["ns/x"], [])
        await collector.flush(cfg=_Cfg(), extensions_api=object())
        assert [name for name, _, _ in calls] == ["cr-a", "cr-b"]

    @pytest.mark.asyncio
    async def test_flush_diff_merges_all_writers(self, monkeypatch):
        calls = self._patch_ensure(monkeypatch)
        collector = RouteArtifactCollector()
        collector.set_rules(
            "cr-a",
            "lb",
            ["ns/x"],
            [_rule({"candidates": ["c"]}, ["ns/x"], configDisable=False)],
        )
        await collector.flush(cfg=_Cfg(), extensions_api=object())
        _, _, spec_diff = calls[0]
        live = _spec(
            _rule({"modelMapping": {"a": "b"}}, ["ns/x"]),
            _rule({"candidates": ["stale"]}, ["ns/x"]),
        )
        merged = spec_diff(live)
        configs = [r.config for r in merged.matchRules]
        assert {"modelMapping": {"a": "b"}} in configs
        assert any(c.get("candidates") == ["c"] for c in configs)

    @pytest.mark.asyncio
    async def test_missing_cr_without_create_base_skipped(self, monkeypatch):
        calls = self._patch_ensure(monkeypatch)
        collector = RouteArtifactCollector()
        collector.set_rules("cr-a", "lb", ["ns/x"], [])
        await collector.flush(cfg=_Cfg(), extensions_api=object())
        _, _, spec_diff = calls[0]
        # a manually deleted CR is not recreated from a route event
        assert spec_diff(None) is None

    @pytest.mark.asyncio
    async def test_missing_cr_with_create_base_created(self, monkeypatch):
        calls = self._patch_ensure(monkeypatch)
        base = WasmPluginSpec(matchRules=[])
        collector = RouteArtifactCollector()
        collector.set_rules(
            "cr-a",
            "session-affinity",
            ["ns/x"],
            [_rule({"enabled": True}, ["ns/x"])],
            create_base=base,
        )
        await collector.flush(cfg=_Cfg(), extensions_api=object())
        _, _, spec_diff = calls[0]
        created = spec_diff(None)
        assert base.matchRules == []  # the base is used as a copy, not mutated
        assert len(created.matchRules) == 1
        assert created.matchRules[0].config[RULE_OWNER_KEY] == "session-affinity"
