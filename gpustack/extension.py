"""
Extension plugin interface for GPUStack.

Third-party or enterprise plugins can implement this interface
and register via the ``gpustack.plugins`` entry-point group.

A plugin is fully wired at construction time: the subclass's
``__init__(app, cfg)`` is expected to mount routers, install
middleware, run migrations, and otherwise mutate ``app`` as needed.
After construction the instance can optionally expose long-running
background coroutines via ``async_tasks()`` and a distributed-mode
coordinator via the ``coordinator`` attribute.
"""

import logging
from typing import Any, Coroutine, Generator, List, Optional, TYPE_CHECKING, Tuple

from fastapi import FastAPI

from gpustack.config.config import Config

if TYPE_CHECKING:
    from gpustack.server.coordinator import Coordinator

logger = logging.getLogger(__name__)


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


class Plugin:
    """Base class that all extension plugins must implement.

    Subclasses override ``__init__(app, cfg)`` to perform the full
    registration — there is no separate ``register`` phase. To opt into
    distributed-mode coordination, an ``__init__`` may assign a
    ``Coordinator`` to ``self.coordinator``; the server picks it up from
    ``app.state.extension_plugins`` and owns its lifecycle.
    """

    # Optional distributed-mode coordinator; the server starts/stops it.
    coordinator: Optional["Coordinator"] = None

    def __init__(self, app: FastAPI, cfg: Config) -> None:
        pass

    def async_tasks(self) -> List[Coroutine[Any, Any, Any]]:
        """Long-running background coroutines to be scheduled alongside
        the API server. Each coroutine is awaited in the server's main
        gather(), so an uncaught exception aborts the server — plugins
        are expected to handle their own restart semantics. Default: no
        tasks."""
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
