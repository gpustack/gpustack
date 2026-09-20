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
    """Stands in for the policy table: an in-memory dict behind the
    ActiveRecord classmethods the hooks use."""

    def __init__(self, cls):
        self.cls = cls
        self.rows = {}

    def install(self, monkeypatch):
        store = self

        async def one_by_fields(cls, session, fields, **kw):
            return store.rows.get(fields.get("route_id"))

        async def all_by_fields(cls, session, fields=None, extra_conditions=None, **kw):
            return list(store.rows.values())

        async def create(cls, session, source, **kw):
            row = cls(**source)
            store.rows[source["route_id"]] = row
            return row

        async def _delete(row, session=None, **kw):
            store.rows.pop(row.route_id, None)

        async def _update(row, session=None, source=None, **kw):
            for k, v in (source or {}).items():
                setattr(row, k, v)
            return row

        monkeypatch.setattr(self.cls, "one_by_fields", classmethod(one_by_fields))
        monkeypatch.setattr(self.cls, "all_by_fields", classmethod(all_by_fields))
        monkeypatch.setattr(self.cls, "create", classmethod(create))
        monkeypatch.setattr(self.cls, "delete", _delete)
        monkeypatch.setattr(self.cls, "update", _update)


class TestHooks:
    def test_section_lands_in_own_table(self, monkeypatch):
        from gpustack.routes.plugins.least_load.schemas import LeastLoadPolicy

        store = _MemoryStore(LeastLoadPolicy)
        store.install(monkeypatch)
        route = _route()
        asyncio.run(
            least_load_plugin.on_route_write(
                "update", route, {"weight": 5}, session=None
            )
        )
        assert store.rows[1].config == {"enabled": True, "weight": 5}
        # the route row itself is untouched — storage is the plugin's own
        assert route.meta is None

    def test_untouched_section_keeps_stored_policy(self, monkeypatch):
        from gpustack.routes.plugins.least_load.schemas import LeastLoadPolicy

        store = _MemoryStore(LeastLoadPolicy)
        store.install(monkeypatch)
        store.rows[1] = LeastLoadPolicy(route_id=1, config={"enabled": True})
        asyncio.run(
            least_load_plugin.on_route_write("update", _route(), None, session=None)
        )
        assert store.rows[1].config == {"enabled": True}

    def test_removal_deletes_the_row(self, monkeypatch):
        from gpustack.routes.plugins.session_affinity.schemas import (
            SessionAffinityPolicy,
        )

        store = _MemoryStore(SessionAffinityPolicy)
        store.install(monkeypatch)
        store.rows[1] = SessionAffinityPolicy(route_id=1, config={"enabled": True})
        asyncio.run(
            session_affinity_plugin.on_route_write(
                "update", _route(), None, session=None, removed=True
            )
        )
        assert 1 not in store.rows

    def test_enrich_reads_own_table(self, monkeypatch):
        from gpustack.routes.plugins.least_load.schemas import LeastLoadPolicy

        store = _MemoryStore(LeastLoadPolicy)
        store.install(monkeypatch)
        store.rows[1] = LeastLoadPolicy(route_id=1, config={"enabled": True})
        sections = asyncio.run(
            least_load_plugin.enrich_routes([_route()], session=None)
        )
        assert sections == {1: {"enabled": True}}


class TestIsEffectiveOn:
    """The LB mode derivation asks this instead of reading enrichment
    output: the answer must track the enabled flag in stored config, and
    absent storage means not effective."""

    def test_absent_policy_reports_false(self, monkeypatch):
        from gpustack.routes.plugins.least_load.schemas import LeastLoadPolicy

        store = _MemoryStore(LeastLoadPolicy)
        store.install(monkeypatch)
        assert not asyncio.run(least_load_plugin.is_effective_on(_route(), None))

    def test_enabled_policy_reports_true(self, monkeypatch):
        from gpustack.routes.plugins.least_load.schemas import LeastLoadPolicy

        store = _MemoryStore(LeastLoadPolicy)
        store.install(monkeypatch)
        store.rows[1] = LeastLoadPolicy(route_id=1, config={"enabled": True})
        assert asyncio.run(least_load_plugin.is_effective_on(_route(), None))

    def test_disabled_policy_reports_false(self, monkeypatch):
        from gpustack.routes.plugins.session_affinity.schemas import (
            SessionAffinityPolicy,
        )

        store = _MemoryStore(SessionAffinityPolicy)
        store.install(monkeypatch)
        store.rows[1] = SessionAffinityPolicy(
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
