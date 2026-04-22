"""
Extension plugin interface for GPUStack.

Third-party or enterprise plugins can implement this interface
and register via the ``gpustack.plugins`` entry-point group.

A plugin is fully wired at construction time: the subclass's
``__init__(app, cfg)`` is expected to mount routers, install
middleware, run migrations, and otherwise mutate ``app`` as needed.
After construction the instance can optionally expose long-running
background coroutines via ``async_tasks()``.
"""

from typing import Any, Coroutine, List

from fastapi import FastAPI

from gpustack.config.config import Config


class Plugin:
    """Base class that all extension plugins must implement.

    Subclasses override ``__init__(app, cfg)`` to perform the full
    registration — there is no separate ``register`` phase.
    """

    def __init__(self, app: FastAPI, cfg: Config) -> None:
        pass

    def async_tasks(self) -> List[Coroutine[Any, Any, Any]]:
        """Long-running background coroutines to be scheduled alongside
        the API server. Each coroutine is awaited in the server's main
        gather(), so an uncaught exception aborts the server — plugins
        are expected to handle their own restart semantics. Default: no
        tasks."""
        return []
