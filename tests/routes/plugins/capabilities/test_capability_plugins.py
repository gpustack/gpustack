import asyncio

import pytest
from pydantic import ValidationError

from gpustack.routes.plugins.least_load.config import LeastLoadConfig
from gpustack.routes.plugins.least_load.plugin import least_load_plugin
from gpustack.routes.plugins.session_affinity.config import (
    SessionAffinityConfig,
    SessionKey,
)
from gpustack.routes.plugins.session_affinity.plugin import session_affinity_plugin
from gpustack.schemas.model_routes import ModelRoute
from gpustack.routes.plugins.capability_policy import CapabilityPolicy


def _route(meta=None):
    return ModelRoute(id=1, name="r", targets=0, ready_targets=0, meta=meta)


class TestSessionAffinityConfig:
    def test_session_keys_required(self):
        with pytest.raises(ValidationError):
            SessionAffinityConfig.model_validate({})

    @pytest.mark.parametrize(
        "key",
        [
            {"header": "session_id"},  # header source
            {"bodyKey": "prompt_cache_key"},  # body source
        ],
    )
    def test_valid_sources(self, key):
        config = SessionAffinityConfig.model_validate({"sessionKeys": [key]})
        assert config.to_gateway_rule()["sessionKeys"] == [key]

    @pytest.mark.parametrize(
        "key",
        [
            {"header": "a", "bodyKey": "b"},  # both set
            {},  # neither set
        ],
    )
    def test_exactly_one_source(self, key):
        with pytest.raises(ValidationError, match="exactly one"):
            SessionKey.model_validate(key)

    def test_gateway_rule_shape(self):
        config = SessionAffinityConfig.model_validate(
            {
                "sessionKeys": [{"header": "session_id"}],
                "enableOnPathSuffix": ["/responses"],
                "weight": 2,
            }
        )
        rule = config.to_gateway_rule()
        assert rule == {
            "sessionKeys": [{"header": "session_id"}],
            "enableOnPathSuffix": ["/responses"],
            "weight": 2,
        }

    def test_weight_omitted_when_unset(self):
        config = SessionAffinityConfig.model_validate(
            {"sessionKeys": [{"header": "session_id"}]}
        )
        assert "weight" not in config.to_gateway_rule()

    def test_weight_accepts_fractions(self):
        # The design range is (0, N]: a plugin can be dialled to a small
        # share of the finisher's weighted sum.
        rule = SessionAffinityConfig.model_validate(
            {"sessionKeys": [{"header": "session_id"}], "weight": 0.25}
        ).to_gateway_rule()
        assert rule["weight"] == 0.25

    def test_zero_weight_rejected(self):
        with pytest.raises(ValidationError):
            SessionAffinityConfig.model_validate(
                {"sessionKeys": [{"header": "session_id"}], "weight": 0}
            )


class TestLeastLoadConfig:
    def test_gateway_rule_shape(self):
        rule = LeastLoadConfig.model_validate({"weight": 3}).to_gateway_rule()
        assert rule == {"enabled": True, "weight": 3}
        rule = LeastLoadConfig.model_validate({}).to_gateway_rule()
        assert rule == {"enabled": True}  # weight omitted -> plugin default

    def test_weight_accepts_fractions(self):
        rule = LeastLoadConfig.model_validate({"weight": 0.5}).to_gateway_rule()
        assert rule["weight"] == 0.5

    def test_zero_weight_rejected(self):
        with pytest.raises(ValidationError):
            LeastLoadConfig.model_validate({"weight": 0})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            LeastLoadConfig.model_validate({"weight": -1})


class _MemoryStore:
    """Stands in for the shared capability policy table: an in-memory
    dict keyed by (capability, route_id) behind the ActiveRecord
    classmethods the hooks use."""

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

    def row(self, capability, route_id=1):
        return self.rows.get((capability, route_id))


class TestHooks:
    def test_section_lands_in_shared_table(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        route = _route()
        asyncio.run(
            least_load_plugin.on_route_write(
                "update", route, {"weight": 5}, session=None
            )
        )
        row = store.row("least-load")
        # the weight is a first-class column; the JSON carries the rest
        assert row.weight == 5
        assert row.config == {"enabled": True}
        # the route row itself is untouched — storage is the plugin's own
        assert route.meta is None

    def test_capabilities_share_one_table(self, monkeypatch):
        # both plugins' policies coexist as rows keyed by capability
        store = _MemoryStore()
        store.install(monkeypatch)
        asyncio.run(
            least_load_plugin.on_route_write(
                "update", _route(), {"weight": 2}, session=None
            )
        )
        asyncio.run(
            session_affinity_plugin.on_route_write(
                "update",
                _route(),
                {"sessionKeys": [{"header": "x-session-id"}]},
                session=None,
            )
        )
        assert store.row("least-load").capability == "least-load"
        assert store.row("session-affinity").capability == "session-affinity"
        # one plugin's removal never touches the other's row
        asyncio.run(
            least_load_plugin.on_route_write(
                "update", _route(), None, session=None, removed=True
            )
        )
        assert store.row("least-load") is None
        assert store.row("session-affinity") is not None

    def test_untouched_section_keeps_stored_policy(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("least-load", 1)] = CapabilityPolicy(
            capability="least-load", route_id=1, config={"enabled": True}
        )
        asyncio.run(
            least_load_plugin.on_route_write("update", _route(), None, session=None)
        )
        assert store.row("least-load").config == {"enabled": True}

    def test_removal_deletes_the_row(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("session-affinity", 1)] = CapabilityPolicy(
            capability="session-affinity", route_id=1, config={"enabled": True}
        )
        asyncio.run(
            session_affinity_plugin.on_route_write(
                "update", _route(), None, session=None, removed=True
            )
        )
        assert store.row("session-affinity") is None

    def test_removal_hard_deletes_inside_the_caller_transaction(self, monkeypatch):
        # The table's (capability, route_id) unique constraint plus a
        # soft delete would block adding the policy back after removing
        # it, and an independent commit mid-transaction could leave the
        # route and its policy out of sync — so the delete is hard and
        # uncommitted.
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("session-affinity", 1)] = CapabilityPolicy(
            capability="session-affinity", route_id=1, config={"enabled": True}
        )
        asyncio.run(
            session_affinity_plugin.on_route_write(
                "update", _route(), None, session=None, removed=True
            )
        )
        assert store.delete_calls == [{"soft": False, "auto_commit": False}]
        # the row is gone, so the policy can be stored again
        asyncio.run(
            session_affinity_plugin.on_route_write(
                "update",
                _route(),
                {"sessionKeys": [{"header": "x-session-id"}]},
                session=None,
            )
        )
        assert store.row("session-affinity") is not None

    def test_empty_session_keys_chain_rejected(self):
        from gpustack.routes.plugins.session_affinity.config import (
            SessionAffinityConfig,
        )

        # An empty chain fails gateway rule parsing exactly like an
        # omitted one, so it never reaches storage.
        with pytest.raises(ValidationError):
            SessionAffinityConfig.model_validate({"sessionKeys": []})

    def test_enrich_reads_shared_table(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("least-load", 1)] = CapabilityPolicy(
            capability="least-load", route_id=1, config={"enabled": True}, weight=1.5
        )
        # other capabilities' rows never leak into this plugin's sections
        store.rows[("session-affinity", 1)] = CapabilityPolicy(
            capability="session-affinity",
            route_id=1,
            config={"enabled": True, "sessionKeys": [{"header": "h"}]},
        )
        sections = asyncio.run(
            least_load_plugin.enrich_routes([_route()], session=None)
        )
        # the weight column folds back into the section
        assert sections == {1: {"enabled": True, "weight": 1.5}}


class TestIsEffectiveOn:
    """The LB mode derivation asks this instead of reading enrichment
    output: the answer must track the enabled flag in stored config, and
    absent storage means not effective."""

    def test_absent_policy_reports_false(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        assert not asyncio.run(least_load_plugin.is_effective_on(_route(), None))

    def test_enabled_policy_reports_true(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("least-load", 1)] = CapabilityPolicy(
            capability="least-load", route_id=1, config={"enabled": True}
        )
        assert asyncio.run(least_load_plugin.is_effective_on(_route(), None))

    def test_disabled_policy_reports_false(self, monkeypatch):
        store = _MemoryStore()
        store.install(monkeypatch)
        store.rows[("session-affinity", 1)] = CapabilityPolicy(
            capability="session-affinity",
            route_id=1,
            config={"enabled": False, "sessionKeys": [{"header": "session_id"}]},
        )
        assert not asyncio.run(session_affinity_plugin.is_effective_on(_route(), None))


class TestGatewayEntries:
    def test_registered_capability_publishes_chain_stable_cr(self):
        # A capability plugin that loses its gateway_entries silently
        # disappears from the gateway (nothing publishes its CR). The
        # CR must be chain-stable: instantiated everywhere, inert where
        # no matchRule enables it.
        import gpustack.routes.plugins as plugins_pkg

        for plugin in plugins_pkg.route_plugins():
            if plugin.name not in ("session-affinity", "least-load"):
                continue
            entries = plugin.gateway_entries(None)
            assert len(entries) == 1, plugin.name
            spec = entries[0].spec
            assert spec.defaultConfigDisable is False, plugin.name
            assert spec.defaultConfig, plugin.name
            assert spec.failStrategy == "FAIL_OPEN", plugin.name
