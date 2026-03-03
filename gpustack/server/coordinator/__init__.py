"""
Coordination module for GPUStack.

This module provides coordination capabilities for server instances.
The open-source edition provides a local implementation for single-node
deployments, while the enterprise edition provides distributed
implementations for multi-instance deployments.

Usage:
    from gpustack.server.coordinator import Coordinator, LocalCoordinator, Event

    # Single node mode (default)
    coordinator = LocalCoordinator(config)
    await coordinator.start()

    # Enterprise edition can provide distributed implementations
    # via the plugin system.
"""

from gpustack.server.coordinator.base import Coordinator, Event, EventType
from gpustack.server.coordinator.local import LocalCoordinator

__all__ = [
    'Coordinator',
    'Event',
    'EventType',
    'LocalCoordinator',
]
