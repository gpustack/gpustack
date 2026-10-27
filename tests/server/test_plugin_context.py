"""Contract tests for the static plugin shape and PluginContext.

Covers the invariants of ``gpustack/extension.py``, the loader in
``gpustack/server/app.py`` and the server-side dispatch in
``gpustack/server/server.py``:

* a plugin subclass that does not override ``__init__`` is never
  instantiated — its classmethod hooks are collected with its own
  context instead;
* a subclass that overrides ``__init__`` keeps the legacy instance
  path unchanged, with the instance recorded on its context;
* contexts are structurally isolated: one plugin can never read
  another plugin's store;
* a failing hook refuses startup — plugin loading, coordinator
  construction and event-topic registration alike;
* coordinator selection follows entry-point order across both plugin
  shapes, so a later static plugin cannot displace an earlier
  legacy one;
* a topic already owned by core (or another extension) is not
  re-registered and not tracked as external;
* leader-only tasks are a static-only capability — the legacy shape
  is frozen, and a legacy instance's same-named method is never
  dispatched to.
"""

from types import SimpleNamespace
from typing import List

import pytest
from fastapi import FastAPI

from gpustack.extension import (
    Plugin,
    PluginContext,
    plugin_needs_instantiation,
)
from gpustack.server.app import _load_extension_plugins
from gpustack.server.coordinator.models import (
    _ModelRegistry,
    external_topics,
    get_model_for_topic,
    register_topic,
)
from gpustack.server.server import Server


class StaticPlugin(Plugin):
    """Preferred shape: no ``__init__`` override, classmethod hooks."""

    calls: List[str] = []

    @classmethod
    def migrate(cls, ctx):
        cls.calls.append("migrate")

    @classmethod
    def setup(cls, ctx):
        cls.calls.append("setup")

    @classmethod
    def build_coordinator(cls, ctx):
        cls.calls.append("build_coordinator")
        return None

    @classmethod
    def async_tasks(cls, ctx):
        cls.calls.append(f"async_tasks:{ctx.name}")
        return []

    @classmethod
    def leader_async_tasks(cls, ctx):
        cls.calls.append(f"leader_async_tasks:{ctx.name}")
        return []


class LegacyPlugin(Plugin):
    """Legacy shape: everything happens in ``__init__``."""

    constructed = False

    def __init__(self, app, cfg):
        self.app = app
        self.coordinator = None
        type(self).constructed = True

    def async_tasks(self):
        return []


def _reset_marks():
    StaticPlugin.calls = []
    LegacyPlugin.constructed = False


@pytest.fixture
def clean_topic_registry(monkeypatch):
    """Isolate the process-wide topic registry per test."""
    monkeypatch.setattr(_ModelRegistry, "_REGISTRY", dict(_ModelRegistry._REGISTRY))
    monkeypatch.setattr(
        _ModelRegistry, "_EXTERNAL_TOPICS", list(_ModelRegistry._EXTERNAL_TOPICS)
    )


def _load_plugins(monkeypatch, plugins) -> FastAPI:
    """Run the loader over the given ``(name, plugin_class)`` pairs."""
    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter(plugins),
    )
    app = FastAPI()
    _load_extension_plugins(app, SimpleNamespace())
    return app


class _FakeCoordinator:
    def __init__(self, label):
        self.label = label
        self.started = False

    async def start(self):
        self.started = True


def _stub_coordinator_plumbing(monkeypatch):
    """Neutralize the post-coordinator wiring ``_init_coordinator`` does.

    The bus, the cache coordinator and the JWT-secret check all reach
    real subsystems; the paths under test stop at the plugin contract.
    """

    async def _noop(*args, **kwargs):
        pass

    monkeypatch.setattr("gpustack.server.bus.set_coordinator", lambda coordinator: None)
    monkeypatch.setattr("gpustack.server.bus.event_bus", SimpleNamespace(start=_noop))
    monkeypatch.setattr(
        "gpustack.server.cache.set_coordinator", lambda coordinator: None
    )
    monkeypatch.setattr(Server, "_prepare_jwt_secret_key", _noop)
    local = _FakeCoordinator("local")
    monkeypatch.setattr("gpustack.server.server.LocalCoordinator", lambda config: local)
    return local


def test_instantiation_flag_follows_init_override():
    assert plugin_needs_instantiation(StaticPlugin) is False
    assert plugin_needs_instantiation(LegacyPlugin) is True


def test_context_store_is_isolated_per_plugin():
    app = FastAPI()
    cfg = SimpleNamespace()
    a = PluginContext(name="a", app=app, cfg=cfg)
    b = PluginContext(name="b", app=app, cfg=cfg)

    a.set("secret", "from-a")

    assert a.get("secret") == "from-a"
    assert b.get("secret") is None
    # There is no accessor for another plugin's store: the only state a
    # context exposes is its own ``_store``.
    assert a._store is not b._store


def test_coordinator_injection_is_visible_through_context():
    ctx = PluginContext(name="a", app=FastAPI(), cfg=SimpleNamespace())
    coordinator = SimpleNamespace(is_leader=lambda: True)
    ctx.coordinator = coordinator
    assert ctx.coordinator.is_leader() is True


def test_loader_collects_static_hooks_without_instantiating(monkeypatch):
    _reset_marks()
    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter([("static", StaticPlugin)]),
    )
    app = FastAPI()
    _load_extension_plugins(app, SimpleNamespace())

    assert app.state.extension_plugins == []
    ctx = app.state.plugin_contexts["static"]
    assert isinstance(ctx, PluginContext)
    assert ctx.plugin_cls is StaticPlugin
    # A static plugin is never instantiated.
    assert ctx.instance is None
    # Assembly order: migrations before wiring.
    assert StaticPlugin.calls == ["migrate", "setup"]


def test_loader_keeps_legacy_instance_path(monkeypatch):
    _reset_marks()
    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter([("legacy", LegacyPlugin)]),
    )
    app = FastAPI()
    _load_extension_plugins(app, SimpleNamespace())

    assert LegacyPlugin.constructed is True
    assert len(app.state.extension_plugins) == 1
    # No classmethod hooks are collected for an instance-based plugin.
    assert StaticPlugin.calls == []
    ctx = app.state.plugin_contexts["legacy"]
    assert "legacy" in app.state.plugin_contexts
    # The legacy instance is recorded on its context so server-side
    # dispatch stays in entry-point order.
    assert ctx.instance is app.state.extension_plugins[0]


def test_task_hooks_receive_the_plugin_context(monkeypatch):
    _reset_marks()
    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter([("static", StaticPlugin)]),
    )
    app = FastAPI()
    _load_extension_plugins(app, SimpleNamespace())
    ctx = app.state.plugin_contexts["static"]
    _reset_marks()

    # The server dispatches tasks exactly this way (server.py
    # ``_start_extension_plugins`` / ``_start_extension_plugin_leader_tasks``).
    assert StaticPlugin.async_tasks(ctx) == []
    assert StaticPlugin.leader_async_tasks(ctx) == []
    assert StaticPlugin.calls == [
        "async_tasks:static",
        "leader_async_tasks:static",
    ]


def test_base_defaults_are_safe_without_override():
    # A static plugin that overrides nothing still loads: the base
    # ``async_tasks`` ignores its bound argument and returns no tasks.
    class Bare(Plugin):
        pass

    ctx = PluginContext(name="bare", app=FastAPI(), cfg=SimpleNamespace())
    assert plugin_needs_instantiation(Bare) is False
    assert Bare.async_tasks(ctx) == []
    assert Bare.leader_async_tasks(ctx) == []
    assert Bare.build_coordinator(ctx) is None
    assert Bare.event_topics() == {}


def test_failing_static_hook_refuses_startup(monkeypatch):
    class Broken(Plugin):
        @classmethod
        def migrate(cls, ctx):
            raise RuntimeError("schema diverged")

    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter([("broken", Broken)]),
    )
    with pytest.raises(RuntimeError, match="broken"):
        _load_extension_plugins(FastAPI(), SimpleNamespace())


# ---- coordinator selection ----


@pytest.mark.asyncio
async def test_legacy_coordinator_wins_when_registered_first(monkeypatch):
    _stub_coordinator_plumbing(monkeypatch)
    legacy_coordinator = _FakeCoordinator("legacy")
    static_coordinator = _FakeCoordinator("static")

    class LegacyCoordinatorPlugin(Plugin):
        def __init__(self, app, cfg):
            self.coordinator = legacy_coordinator

        def async_tasks(self):
            return []

    class StaticCoordinatorPlugin(Plugin):
        @classmethod
        def build_coordinator(cls, ctx):
            return static_coordinator

    # Entry-point order decides, not shape: the legacy plugin registered
    # first keeps precedence over the later static plugin.
    app = _load_plugins(
        monkeypatch,
        [
            ("legacy-first", LegacyCoordinatorPlugin),
            ("static-second", StaticCoordinatorPlugin),
        ],
    )
    server = Server(SimpleNamespace(), None)
    await server._init_coordinator(app)

    assert server._coordinator is legacy_coordinator
    assert legacy_coordinator.started is True
    assert static_coordinator.started is False


@pytest.mark.asyncio
async def test_static_coordinator_wins_when_registered_first(monkeypatch):
    _stub_coordinator_plumbing(monkeypatch)
    legacy_coordinator = _FakeCoordinator("legacy")
    static_coordinator = _FakeCoordinator("static")

    class LegacyCoordinatorPlugin(Plugin):
        def __init__(self, app, cfg):
            self.coordinator = legacy_coordinator

        def async_tasks(self):
            return []

    class StaticCoordinatorPlugin(Plugin):
        @classmethod
        def build_coordinator(cls, ctx):
            return static_coordinator

    app = _load_plugins(
        monkeypatch,
        [
            ("static-first", StaticCoordinatorPlugin),
            ("legacy-second", LegacyCoordinatorPlugin),
        ],
    )
    server = Server(SimpleNamespace(), None)
    await server._init_coordinator(app)

    assert server._coordinator is static_coordinator
    assert static_coordinator.started is True
    assert legacy_coordinator.started is False


@pytest.mark.asyncio
async def test_failing_build_coordinator_refuses_startup(monkeypatch):
    _stub_coordinator_plumbing(monkeypatch)

    class Broken(Plugin):
        @classmethod
        def build_coordinator(cls, ctx):
            raise RuntimeError("coordinator exploded")

    app = _load_plugins(monkeypatch, [("broken", Broken)])
    server = Server(SimpleNamespace(), None)
    # Skipping a failed coordinator would silently fall through to
    # LocalCoordinator in a multi-instance deployment.
    with pytest.raises(RuntimeError, match="Failed to build coordinator.*'broken'"):
        await server._init_coordinator(app)


# ---- event-topic registration ----


def test_register_topic_tracks_extension_topics(clean_topic_registry):
    class Widget:
        pass

    register_topic("widget", Widget)

    assert external_topics() == ["widget"]
    assert get_model_for_topic("widget") is Widget


def test_register_topic_ignores_core_and_duplicate_topics(clean_topic_registry):
    from gpustack.schemas.workers import Worker

    class Impostor:
        pass

    class First:
        pass

    class Second:
        pass

    # Core's own entry is kept; the impostor is neither registered nor
    # tracked as external.
    register_topic("worker", Impostor)
    assert "worker" not in external_topics()
    assert get_model_for_topic("worker") is Worker

    # The first extension registration wins over a later one.
    register_topic("dup", First)
    register_topic("dup", Second)
    assert external_topics() == ["dup"]
    assert get_model_for_topic("dup") is First


@pytest.mark.asyncio
async def test_event_topics_registration_via_server_startup(
    monkeypatch, clean_topic_registry
):
    _stub_coordinator_plumbing(monkeypatch)
    from gpustack.schemas.workers import Worker

    class Widget:
        pass

    class TopicsPlugin(Plugin):
        @classmethod
        def event_topics(cls):
            # "worker" is core-owned: the registration must be ignored
            # and must not be tracked as external (which would preload
            # the worker cache twice).
            return {"widget": Widget, "worker": Widget}

    app = _load_plugins(monkeypatch, [("topics", TopicsPlugin)])
    server = Server(SimpleNamespace(), None)
    await server._init_coordinator(app)

    assert external_topics() == ["widget"]
    assert get_model_for_topic("widget") is Widget
    assert get_model_for_topic("worker") is Worker


@pytest.mark.asyncio
async def test_failing_event_topics_refuses_startup(monkeypatch):
    _stub_coordinator_plumbing(monkeypatch)

    class Broken(Plugin):
        @classmethod
        def event_topics(cls):
            raise RuntimeError("topics exploded")

    app = _load_plugins(monkeypatch, [("broken", Broken)])
    server = Server(SimpleNamespace(), None)
    # Continuing would leave the plugin running without cross-instance
    # enrichment: writes served by another instance would be dropped.
    with pytest.raises(RuntimeError, match="Failed to register event topics.*'broken'"):
        await server._init_coordinator(app)


# ---- leader-only task collection ----


@pytest.mark.asyncio
async def test_leader_tasks_are_collected_from_static_plugins_only(monkeypatch):
    import asyncio

    started = []

    class StaticLeaderPlugin(Plugin):
        @classmethod
        def leader_async_tasks(cls, ctx):
            async def _leader_task():
                started.append("static")

            return [_leader_task()]

    class LegacyWithLeaderTasks(Plugin):
        """Legacy shape, frozen: its same-named method is not dispatched."""

        def __init__(self, app, cfg):
            pass

        def async_tasks(self):
            return []

        def leader_async_tasks(self):
            async def _leader_task():
                started.append("legacy")

            return [_leader_task()]

    app = _load_plugins(
        monkeypatch,
        [
            ("static", StaticLeaderPlugin),
            ("legacy", LegacyWithLeaderTasks),
        ],
    )
    server = Server(SimpleNamespace(), None)
    server._app = app
    server._start_extension_plugin_leader_tasks()
    await asyncio.sleep(0)

    # Only the static hook ran; the legacy instance's same-named
    # method was never dispatched to.
    assert started == ["static"]
    assert len(server._async_tasks) == 1


@pytest.mark.asyncio
async def test_failing_leader_task_collection_exits_for_restart(monkeypatch):
    exited = []

    class Broken(Plugin):
        @classmethod
        def leader_async_tasks(cls, ctx):
            raise RuntimeError("collection exploded")

    monkeypatch.setattr(
        "gpustack.server.server.os._exit", lambda code: exited.append(code)
    )
    app = _load_plugins(monkeypatch, [("broken", Broken)])
    server = Server(SimpleNamespace(), None)
    server._app = app
    server._start_extension_plugin_leader_tasks()

    # A leader missing a plugin's single-writer reconciler must not keep
    # serving: hard-exit so the container restarts and retries.
    assert exited == [1]


def test_failing_static_async_tasks_collection_refuses_startup(monkeypatch):
    class Broken(Plugin):
        @classmethod
        def async_tasks(cls, ctx):
            raise RuntimeError("task collection exploded")

    app = _load_plugins(monkeypatch, [("broken", Broken)])
    server = Server(SimpleNamespace(), None)

    # Static-hook collection failures are uniform: no less fatal than
    # a running task dying in the main gather.
    with pytest.raises(RuntimeError, match="Failed to start async tasks.*'broken'"):
        server._start_extension_plugins(app)


@pytest.mark.asyncio
async def test_dead_leader_task_exits_for_restart(monkeypatch):
    import asyncio

    exited = []

    class Dying(Plugin):
        @classmethod
        def leader_async_tasks(cls, ctx):
            async def _dying_task():
                raise RuntimeError("reconciler died mid-flight")

            return [_dying_task()]

    monkeypatch.setattr(
        "gpustack.server.server.os._exit", lambda code: exited.append(code)
    )
    app = _load_plugins(monkeypatch, [("dying", Dying)])
    server = Server(SimpleNamespace(), None)
    server._app = app
    server._start_extension_plugin_leader_tasks()
    await asyncio.sleep(0.05)

    # In distributed mode these tasks start after the main gather has
    # snapshotted its task list, so nothing else would observe the
    # death — the done-callback must fence it.
    assert exited == [1]


def test_duplicate_entry_point_names_refuse_startup(monkeypatch):
    _reset_marks()
    monkeypatch.setattr(
        "gpustack.server.app.iter_plugin_classes",
        lambda: iter([("dup", StaticPlugin), ("dup", LegacyPlugin)]),
    )
    # A second distribution registering the same entry-point name must
    # not silently replace the first plugin's context (dropping its
    # hooks, or even its coordinator).
    with pytest.raises(RuntimeError, match="Duplicate plugin entry-point name 'dup'"):
        _load_extension_plugins(FastAPI(), SimpleNamespace())
