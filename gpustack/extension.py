"""
Extension plugin interface for GPUStack.

Third-party or enterprise plugins can implement this interface
and register via the ``gpustack.plugins`` entry-point group.
"""

from abc import ABC, abstractmethod

from fastapi import FastAPI

from gpustack.config.config import Config


class Plugin(ABC):
    """Base class that all extension plugins must implement."""

    @abstractmethod
    def register(self, app: FastAPI, cfg: Config) -> None:
        """Register the plugin with the application."""
