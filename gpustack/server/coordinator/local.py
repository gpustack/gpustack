"""
Local Coordinator - Single Node Implementation.

This is the default implementation that operates in single-node mode.
All coordination methods are no-ops or return default values
that work for a single instance.
"""

import logging
from typing import Any

from gpustack.server.coordinator.base import Coordinator, Event

logger = logging.getLogger(__name__)


class LocalCoordinator(Coordinator):
    """
    Local (single-node) implementation of Coordinator.

    In this mode:
    - This instance is always the leader
    - Pub/Sub is local only (no distribution)
    - Shared state is in-memory
    """

    def __init__(self, config: Any = None, **kwargs):
        super().__init__(config, **kwargs)
        self._started = False

    async def start(self):
        """Start the local coordinator (no-op)."""
        self._started = True
        self._is_leader = True
        logger.debug("Local coordinator started (single-node mode)")

    async def stop(self):
        """Stop the local coordinator (no-op)."""
        self._started = False
        logger.debug("Local coordinator stopped")

    # Leader Election - Always leader in local mode
    async def acquire_leadership(self, ttl: int) -> bool:
        """Always returns True in local mode."""
        self._is_leader = True
        return True

    async def renew_leadership(self, ttl: int) -> bool:
        """Always returns True in local mode."""
        return True

    async def release_leadership(self):
        """No-op in local mode."""
        self._is_leader = False

    # Pub/Sub - Local only
    async def publish(self, channel: str, event: Event):
        """
        Publish event to local subscribers only.

        In local mode, events don't leave this process.
        """
        self._notify_local_subscribers(channel, event)
