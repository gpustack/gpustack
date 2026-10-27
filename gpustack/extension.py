"""
Extension plugin interface for GPUStack.

Third-party plugins can implement this interface
and register via the ``gpustack.plugins`` entry-point group.

A plugin is *static*: the entry point resolves to a ``Plugin``
subclass that is never instantiated. Every hook is a classmethod and
receives the plugin's own :class:`PluginContext`, which carries the
app, the config, core-owned services, and a private per-plugin store
(see :class:`PluginContext` for the visibility contract). ``migrate``
and ``setup`` run during ``create_app``; ``build_coordinator`` /
``async_tasks`` / ``leader_async_tasks`` are collected at server
startup. :class:`Plugin` and :class:`PluginContext` below document the
full contract.

The older instance-based shape — a subclass overriding
``__init__(app, cfg)`` that does all of its wiring from the
constructor, exposes tasks via the ``async_tasks`` instance method and
a coordinator via the ``coordinator`` attribute — keeps working
unchanged through :func:`plugin_needs_instantiation`; it is the
migration path, not the target, and it is frozen: new capabilities
land on the static hooks only.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Coroutine, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

from fastapi import FastAPI, Request

from gpustack.config.config import Config

if TYPE_CHECKING:
    from gpustack.api.exceptions import HTTPException
    from gpustack.server.coordinator import Coordinator
    from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def resolve_version_info() -> Tuple[str, str]:
    """Let plugins override the reported version. First non-None wins; on
    any failure, fall back to core's baked-in values.

    Use this anywhere core reports a version to an operator, rather than
    ``from gpustack import __version__`` — a module-level copy is bound at
    import time and cannot reflect a plugin, so a downstream repackage would
    surface its own build string in some places and the wheel's baked-in one
    in others. Current callers: ``gpustack version``, the ``start`` /
    ``reload-config`` banners, the ``/version`` endpoint, OpenAPI
    ``info.version``, the worker version registered on join and checked
    against the server's, the deployment-document export header, and the
    worker image tag from the cluster registration-token endpoint.

    Deliberately not used for two things. The outbound ``User-Agent`` sent to
    model sources identifies the core client to a third party and has no
    reason to carry a repackager's build string. The update check compares
    against a release feed indexed by core's own versions, where a repackaged
    build has no entry at all -- a plugin that reports its own version is
    expected to serve ``/update`` itself.

    Memoized: the entry-point scan reads package metadata off disk, and the
    installed plugins cannot change within a process.
    """
    from gpustack import __version__, __git_commit__

    for name, plugin_class in iter_plugin_classes():
        if not (isinstance(plugin_class, type) and issubclass(plugin_class, Plugin)):
            continue
        try:
            info = plugin_class.get_version_info()
        except Exception:
            logger.debug(
                "Failed to read version info from plugin '%s'", name, exc_info=True
            )
            continue
        if info is not None:
            return info
    return __version__, __git_commit__


def iter_plugin_classes() -> Generator[Tuple[str, type], None, None]:
    """Iterate over registered plugin classes via the ``gpustack.plugins``
    entry-point group.

    Used at CLI-parse time (before any FastAPI app exists) so plugins can
    contribute ``start`` command arguments via ``Plugin.setup_start_cmd``.
    At runtime the server instantiates plugins inside ``create_app`` and
    stores instances on ``app.state.extension_plugins``.
    """
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="gpustack.plugins"):
            try:
                yield ep.name, ep.load()
            except Exception:
                logger.warning(f"Failed to load plugin class: {ep.name}", exc_info=True)
    except ImportError:
        pass


@dataclass
class PluginContext:
    """Per-plugin, core-owned context handed to every classmethod hook.

    Visibility contract, three tiers:

    * **Platform facts and core-owned services** — ``app``, ``cfg`` and
      ``coordinator`` — are the same objects for every plugin and are
      meant to be shared.
    * **Private state** — the ``set``/``get`` store — is scoped
      structurally: a plugin can only ever touch its own context, and
      there is deliberately no accessor for another plugin's store.
      Two plugins therefore cannot read each other's wiring, neither
      by accident nor by ordering. This is a coupling boundary, not a
      security boundary: plugins are trusted in-process code, and the
      context registry (``app.state.plugin_contexts``) is core
      plumbing — a plugin reaching into another context through it is
      outside the contract.
    * **Published capabilities** — a plugin exposing something for
      other plugins to consume — are deliberately absent. Cross-plugin
      needs go through core-owned services today; an explicit
      publish/consume mechanism can be added later without disturbing
      this contract.

    Store wiring results once (in ``setup``) and reuse them from later
    hooks: constructing the same component lazily in two hooks yields
    two copies of it — two caches, two clients — which is exactly the
    failure the single-store rule exists to prevent.
    """

    name: str
    """Entry-point name of the owning plugin."""

    app: FastAPI
    cfg: Config

    plugin_cls: Optional[type] = None
    """The plugin class this context was created for. Core plumbing —
    hooks are invoked through it — not for plugin code."""

    instance: Optional["Plugin"] = None
    """The constructed instance, legacy shape only. Core plumbing — the
    server dispatches legacy hooks through it and keeps its dispatch
    loops in entry-point order. A static plugin is never
    instantiated, so this stays ``None``. Not for plugin code."""

    coordinator: Optional["Coordinator"] = None
    """The process-wide coordinator, injected by the server after
    ``build_coordinator`` discovery, before task collection. Read-only
    from a plugin's point of view."""

    _store: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)


def plugin_needs_instantiation(plugin_cls: type) -> bool:
    """Whether ``plugin_cls`` follows the legacy instance-based shape.

    Only a subclass that overrides ``__init__(app, cfg)`` is
    instantiated; core then talks to it exactly as before (instance
    task methods, ``coordinator`` attribute). A subclass that leaves
    ``__init__`` alone is static: core calls its classmethod hooks
    with a :class:`PluginContext` and never constructs it.
    """
    return plugin_cls.__init__ is not Plugin.__init__


class Plugin:
    """Base class that all extension plugins must implement.

    Preferred shape: do not override ``__init__``. Implement the
    classmethod hooks instead — ``migrate`` / ``setup`` run during app
    assembly, ``build_coordinator`` / ``async_tasks`` /
    ``leader_async_tasks`` are collected at server startup — each
    receiving the plugin's own :class:`PluginContext`.

    Legacy shape: override ``__init__(app, cfg)`` and do the full
    registration from the constructor; expose tasks via the
    ``async_tasks`` instance method below and a distributed-mode
    coordinator via the ``coordinator`` attribute. Still fully
    supported; see :func:`plugin_needs_instantiation`. The legacy shape
    is frozen — new capabilities (leader-only tasks, event topics) land
    on the static hooks only.

    Pick one shape per plugin. A legacy instance's classmethod-style
    overrides (or the reverse) are not dispatched to.
    """

    # Optional distributed-mode coordinator; the server starts/stops it.
    # Legacy shape only — the preferred shape overrides build_coordinator.
    coordinator: Optional["Coordinator"] = None

    def __init__(self, app: FastAPI, cfg: Config) -> None:
        pass

    def async_tasks(self) -> List[Coroutine[Any, Any, Any]]:
        """Long-running background coroutines to be scheduled alongside
        the API server. Each coroutine is awaited in the server's main
        gather(), so an uncaught exception aborts the server — plugins
        are expected to handle their own restart semantics. Default: no
        tasks.

        Static shape: override as a classmethod taking the plugin's
        PluginContext. The base definition serves both dispatches — a
        legacy instance is called with no arguments, the class is called
        with the context, which the base ignores.
        """
        return []

    # ---- preferred (static) shape ----
    # The hooks below are collected only for subclasses that do NOT
    # override __init__; they receive the plugin's PluginContext.

    @classmethod
    def migrate(cls, ctx: "PluginContext") -> None:
        """Run the plugin's database migrations.

        Called once during ``create_app``, before ``setup`` and before
        the app serves. An exception refuses startup — an un-migrated
        plugin serving traffic is worse than a server that does not
        come up. Default: nothing to migrate.
        """
        pass

    @classmethod
    def setup(cls, ctx: "PluginContext") -> None:
        """One-shot wiring during ``create_app``, after core routes are
        registered: mount the plugin's own routers, mutate ``app``, and
        construct the plugin's shared components into ``ctx.set`` so
        later hooks reuse the same objects.

        Runs before a coordinator exists — ``ctx.coordinator`` is still
        ``None`` here; anything leadership-aware belongs in
        ``async_tasks`` / ``leader_async_tasks`` instead. An exception
        refuses startup. Default: nothing to wire.
        """
        pass

    @classmethod
    def build_coordinator(cls, ctx: "PluginContext") -> Optional["Coordinator"]:
        """Offer a distributed-mode ``Coordinator`` for this process.

        Collected once at server startup, before the event bus goes
        online: the first non-``None`` result across plugins wins and
        the server owns its lifecycle (start/stop, leader election).
        The chosen coordinator is injected into every plugin's context
        (``ctx.coordinator``) before tasks are collected. Return
        ``None`` — the default — to run under core's LocalCoordinator.
        """
        return None

    @classmethod
    def leader_async_tasks(cls, ctx: "PluginContext") -> List[Coroutine[Any, Any, Any]]:
        """Leader-only background coroutines, for reconcilers whose
        writes must have a single owner.

        Started with the server's own leader-only tasks: on a single-node
        deployment (``LocalCoordinator``) they start immediately, and in
        distributed mode they start when this instance acquires
        leadership. A deposed leader exits the process (see
        ``Server._leader_election_loop``), so tasks never outlive their
        leadership — plugins do not need cancellation handling.

        This is the place for controllers that reconcile external state
        (gateway CRs, and anything else where two writers racing can
        revert each other's writes), rather than gating every write on
        ``coordinator.is_leader()`` from ``async_tasks``. Static
        shape only — a legacy instance's same-named method is not
        dispatched to. Default: no tasks.
        """
        return []

    @classmethod
    def setup_start_cmd(cls, parser) -> None:
        """Contribute arguments to the ``gpustack start`` argparse parser.

        Called at CLI-parse time, before ``Config`` is built and before
        any FastAPI app exists, so this must be a classmethod and must
        not depend on instance state.
        """
        pass

    @classmethod
    def contribute_config(cls, args, config_data: dict) -> None:
        """Forward plugin-contributed CLI args into the ``Config`` kwargs dict.

        Called after argparse parsing and core's ``set_*_options`` have
        populated ``config_data``, but before ``Config(**config_data)`` is
        constructed. Args registered via ``setup_start_cmd`` end up on the
        ``args`` namespace but are not automatically forwarded — override
        this method to copy the relevant fields. Because ``Config`` uses
        ``extra="allow"``, copied keys appear as attributes on ``cfg``.

        Plugins should not overwrite keys that core already set unless the
        intent is to override; this hook runs last and last-write-wins.
        """
        pass

    @classmethod
    def extra_image_list(cls) -> List[str]:
        """Contribute image names to the global GPUStack image list.

        Called at CLI-parse time, before any FastAPI app exists, so this
        must be a classmethod and must not depend on instance state.
        """
        return []

    @classmethod
    def event_topics(cls) -> Dict[str, type]:
        """Register the plugin's own tables for cross-instance events.

        Maps a bus topic (the model's topic name) to its model class.
        Without a registration, an event published on that topic is
        delivered to the publishing instance's subscribers only — a
        controller running on another instance (typically the leader)
        would never see writes this instance served.

        Collected once at server startup, before the event bus delivers
        cross-instance events. Registering a topic core already owns is
        logged and ignored. Still a classmethod, and the model classes
        must be importable at that point.
        """
        return {}

    @classmethod
    async def on_authenticated(
        cls,
        ctx: "PluginContext",
        request: "Request",
        session: "AsyncSession",
        error: "Optional[HTTPException]",
    ) -> None:
        """Apply a policy at the moment a request authenticates.

        Called once when authentication settles — at the end of
        ``authenticate_request``, and on the inference gateway path of
        ``authenticate_gateway_asserted_identity``, for identities the
        gateway verified itself — on success and on failure alike. On
        success the resolved identity is already attached —
        ``request.state.user``, plus ``request.state.api_key`` when the
        request authenticated with an API key — and ``error`` is ``None``;
        raise an ``HTTPException`` subclass (e.g. ``ForbiddenException``)
        to reject the request. A rejection reaches the client as raised:
        it is never re-interpreted as anonymous access, whatever its
        status code. Any other exception is logged and the request fails
        closed (500). On failure nothing is attached and ``error`` is the
        exception the request will fail with; exceptions from the hook are
        only logged, never masking the real auth error.

        The hook may only reject: it has no return value, because letting
        it rewrite ``user`` / ``api_key`` would be an
        identity-escalation surface. It sees the auth lookup's
        short-lived ``session`` (query your own tables through it, but
        do not keep it), runs on every request that presents credentials
        or carries a gateway-asserted identity, and must stay cheap —
        cache parsed rules in the context store instead of querying per
        call. Static shape only: a legacy instance's same-named method is
        not dispatched to. WebSocket-scope authenticators hold no
        ``Request`` and stay hook-free by design. Default: no policy.
        """
        return None

    @classmethod
    def cache_provider_assets(cls) -> List[Tuple[str, str]]:
        """Contribute cache-provider catalog assets as (package, resource)
        pairs, read after the bundled catalog.

        A declaration replaces the bundled entry of the same name, which is
        how a plugin carrying what a provider needs turns the catalog's
        placeholder for it into the real thing.

        Read where the packaged baseline is seeded into its source row — the
        server, and only the instance holding leadership. Everything else
        reads that row or the catalog it materializes: another server
        instance serves the download from it, and a worker fetches the
        merged catalog over the API. A plugin therefore belongs on the
        server; installing it on a worker contributes nothing. Still a
        classmethod, and must not depend on instance state.
        """
        return []

    @classmethod
    def get_version_info(cls) -> Optional[Tuple[str, str]]:
        """Override the version reported by ``gpustack version`` and the
        ``/version`` endpoint.

        Return ``(version, git_commit)`` to override the core values, or
        ``None`` to defer to core. Called at CLI-parse time, so this must
        be a classmethod and must not depend on instance state. If multiple
        plugins return a value, the first non-None wins.
        """
        return None
