import pytest
from typing import Any, Dict, List

from gpustack.routes.plugins import (
    RoutePlugin,
    dispatch_route_hooks,
    dispatch_target_hooks,
    dispatch_route_reconcile,
    enrich_routes,
    enrich_targets,
    register_route_plugin,
    route_plugins,
)
from gpustack.routes.plugins import registry as plugin_registry
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget


class RecordingPlugin(RoutePlugin):
    """Records every hook call; stands in for a real plugin without
    touching a database or the gateway."""

    def __init__(self, name: str, extends: bool = False):
        self.name = name
        if extends:
            self.RouteExtension = dict
            self.TargetExtension = dict
        self.route_calls: List[tuple] = []
        self.target_calls: List[tuple] = []

    async def on_route_write(self, action, route, section, session, removed=False):
        self.route_calls.append((action, route.id, section))

    async def on_target_write(self, action, target, section, session, removed=False):
        self.target_calls.append((action, target.id, section))

    async def enrich_routes(self, routes, session) -> Dict[int, Dict[str, Any]]:
        return {route.id: {"route_id": route.id} for route in routes}

    async def enrich_targets(self, targets, session) -> Dict[int, Dict[str, Any]]:
        return {target.id: {"target_id": target.id} for target in targets}


@pytest.fixture
def clean_registry():
    saved = dict(plugin_registry._REGISTRY)
    plugin_registry._REGISTRY.clear()
    yield plugin_registry._REGISTRY
    plugin_registry._REGISTRY.clear()
    plugin_registry._REGISTRY.update(saved)


def _route(route_id: int = 1) -> ModelRoute:
    return ModelRoute(id=route_id, name="r", targets=0, ready_targets=0)


def _target(target_id: int = 2) -> ModelRouteTarget:
    return ModelRouteTarget(id=target_id, name="t", route_name="r", route_id=1)


def test_register_is_write_once(clean_registry):
    plugin = RecordingPlugin("p1")
    register_route_plugin(plugin)
    with pytest.raises(RuntimeError, match="already registered"):
        register_route_plugin(RecordingPlugin("p1"))


def test_register_requires_name(clean_registry):
    class Nameless(RoutePlugin):
        pass

    with pytest.raises(ValueError, match="no 'name'"):
        register_route_plugin(Nameless())


def test_plugins_in_registration_order(clean_registry):
    register_route_plugin(RecordingPlugin("first"))
    register_route_plugin(RecordingPlugin("second"))
    assert [p.name for p in route_plugins()] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sections,expected",
    [
        (None, False),  # request mentioned nothing
        ({}, False),  # empty plugins mapping
        ({"p1": {"a": 1}}, True),  # this plugin's section
        ({"p2": {"a": 1}}, False),  # another REGISTERED plugin's section
    ],
)
async def test_route_dispatch_fires_only_for_touched_plugin(
    clean_registry, sections, expected
):
    plugin = RecordingPlugin("p1")
    register_route_plugin(plugin)
    register_route_plugin(RecordingPlugin("p2"))
    await dispatch_route_hooks("update", _route(), sections, session=None)
    assert bool(plugin.route_calls) is expected
    if expected:
        assert plugin.route_calls[0][2] == {"a": 1}


@pytest.mark.asyncio
async def test_delete_dispatches_even_without_section(clean_registry):
    plugin = RecordingPlugin("p1")
    register_route_plugin(plugin)
    await dispatch_route_hooks("delete", _route(), None, session=None)
    assert plugin.route_calls == [("delete", 1, None)]


@pytest.mark.asyncio
async def test_dispatch_rejects_non_object_section(clean_registry):
    register_route_plugin(RecordingPlugin("p1"))
    with pytest.raises(ValueError, match="must be an object"):
        await dispatch_route_hooks("update", _route(), {"p1": "oops"}, session=None)


@pytest.mark.asyncio
async def test_target_dispatch_mirrors_route(clean_registry):
    plugin = RecordingPlugin("p1")
    register_route_plugin(plugin)
    await dispatch_target_hooks("create", _target(), {"p1": {"x": 2}}, session=None)
    assert plugin.target_calls == [("create", 2, {"x": 2})]


@pytest.mark.asyncio
async def test_enrich_route_without_extensions_is_none(clean_registry):
    register_route_plugin(RecordingPlugin("p1", extends=False))
    route = _route()
    target = _target()
    assert await enrich_routes([route], session=None) == {}
    assert await enrich_targets([target], session=None) == {}


@pytest.mark.asyncio
async def test_enrich_collects_sections_per_plugin(clean_registry):
    register_route_plugin(RecordingPlugin("p1", extends=True))
    register_route_plugin(RecordingPlugin("p2", extends=True))
    route_sections = await enrich_routes([_route(7)], session=None)
    assert route_sections == {7: {"p1": {"route_id": 7}, "p2": {"route_id": 7}}}
    target_sections = await enrich_targets([_target(9)], session=None)
    assert target_sections == {9: {"p1": {"target_id": 9}, "p2": {"target_id": 9}}}


@pytest.mark.asyncio
async def test_enrich_survives_plugin_failure(clean_registry):
    class Broken(RecordingPlugin):
        async def enrich_routes(self, routes, session):
            raise RuntimeError("boom")

    register_route_plugin(Broken("broken", extends=True))
    register_route_plugin(RecordingPlugin("ok", extends=True))
    assert await enrich_routes([_route()], session=None) == {1: {"ok": {"route_id": 1}}}


@pytest.mark.asyncio
async def test_no_registered_plugins_is_noop(clean_registry):
    # The default server state: hooks and enrichment are cheap no-ops.
    await dispatch_route_hooks("update", _route(), None, session=None)
    assert await enrich_routes([_route()], session=None) == {}


@pytest.mark.asyncio
async def test_reconcile_dispatch_follows_registry(clean_registry):
    # Reconcile wiring follows the registry, not the import graph: an
    # unregistered plugin's logic must not run.
    calls = []

    class Reconciling(RecordingPlugin):
        async def reconcile_route(self, ctx):
            calls.append(self.name)

    register_route_plugin(Reconciling("late"))
    register_route_plugin(Reconciling("early"))
    # no plugin registered under 'absent' — nothing to call
    await dispatch_route_reconcile(ctx=object())
    assert calls == ["late", "early"]


@pytest.mark.asyncio
async def test_explicit_null_dispatches_removal(clean_registry):
    calls = []
    removed_flags = []

    class Removable(RecordingPlugin):
        async def on_route_write(self, action, route, section, session, removed=False):
            calls.append(section)
            removed_flags.append(removed)

    register_route_plugin(Removable("p1"))
    register_route_plugin(RecordingPlugin("p2"))
    await dispatch_route_hooks("update", _route(), {"p1": None, "p2": {}}, session=None)
    assert calls == [None]  # section is None for the removal call
    assert removed_flags == [True]


@pytest.mark.asyncio
async def test_null_section_of_other_plugin_not_visible(clean_registry):
    plugin = RecordingPlugin("p1")
    register_route_plugin(plugin)
    register_route_plugin(RecordingPlugin("p2"))
    # {"p2": null} mentions another registered plugin only — p1 is untouched
    await dispatch_route_hooks("update", _route(), {"p2": None}, session=None)
    assert plugin.route_calls == []


@pytest.mark.asyncio
async def test_unknown_plugin_section_is_rejected(clean_registry):
    # A section for a plugin this server does not register cannot be
    # stored here — rejecting (ValueError, mapped to a 400 by the
    # dispatch wrappers) beats dropping the client's input silently,
    # e.g. on a typo like "sesion_affinity".
    register_route_plugin(RecordingPlugin("p1"))
    with pytest.raises(ValueError, match="unknown plugin section"):
        await dispatch_route_hooks(
            "update", _route(), {"sesion_affinity": {}}, session=None
        )
    with pytest.raises(ValueError, match="plugins must be an object"):
        await dispatch_route_hooks("update", _route(), "foo", session=None)
