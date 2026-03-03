"""
Coordinator Abstract Base Class.

This module defines the interface for coordinating multiple server instances.
The open-source edition provides a local (single-node) implementation, while
the enterprise edition provides distributed implementations using Redis or
PostgreSQL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    CREATED = 1
    UPDATED = 2
    DELETED = 3
    UNKNOWN = 4
    HEARTBEAT = 5

    def __str__(self):
        return self.name


@dataclass
class Event:
    type: EventType
    data: Any
    changed_fields: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    id: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)

        if self.id is None:
            self.id = self._derive_id_from_data()

    def _derive_id_from_data(self) -> Optional[Any]:
        if self.data is None:
            return None

        # For SQLModel objects
        if hasattr(self.data, "id"):
            return getattr(self.data, "id")

        # For plain dict
        if isinstance(self.data, dict):
            return self.data.get("id")

        return None

    def to_dict(self) -> Dict:
        """Serialize event to dict for transmission.

        For cross-instance communication, only the ID is transmitted.
        Subscribers should fetch full data from database and maintain local cache
        to detect changes if needed.
        """
        # Only pass ID to avoid serialization issues and NOTIFY payload limits
        data = None
        if self.data is not None:
            if hasattr(self.data, "id"):
                # SQLModel object - only get ID
                data = {"id": getattr(self.data, 'id')}
            elif isinstance(self.data, dict):
                data = {"id": self.data.get("id")} if "id" in self.data else self.data
            else:
                data = {"id": self.id} if self.id is not None else None

        return {
            "type": self.type.name,
            "data": data,
            # changed_fields is not transmitted across instances
            # Subscribers should detect changes using local cache
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Event":
        """Deserialize event from dict."""
        return cls(
            type=EventType[data.get("type", "UNKNOWN")],
            data=data.get("data"),
            # changed_fields is not transmitted, subscribers detect changes locally
            id=data.get("id"),
        )


class Coordinator(ABC):
    """
    Abstract base class for coordinating server instances.

    Implementations must provide:
    - Leader election for active-passive mode
    - Pub/Sub for event distribution across instances
    """

    def __init__(
        self,
        config: Any,
        leader_election_ttl: int = 30,
        leader_election_renew_interval: int = 10,
    ):
        self._config = config
        self._leader_election_ttl = leader_election_ttl
        self._leader_election_renew_interval = leader_election_renew_interval
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = {}
        self._is_leader = False

    @property
    def leader_election_ttl(self) -> int:
        """Get the leader election TTL in seconds."""
        return self._leader_election_ttl

    @property
    def leader_election_renew_interval(self) -> int:
        """Get the leader election renew interval in seconds."""
        return self._leader_election_renew_interval

    @abstractmethod
    async def start(self):
        """Start the coordinator and establish connections."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the coordinator and release resources."""
        pass

    # Leader Election
    @abstractmethod
    async def acquire_leadership(self, ttl: int) -> bool:
        """
        Try to acquire leadership lock.

        Args:
            ttl: Time to live in seconds for the leadership lock

        Returns:
            True if leadership was acquired, False otherwise
        """
        pass

    @abstractmethod
    async def renew_leadership(self, ttl: int) -> bool:
        """
        Renew the current leadership lock.

        Args:
            ttl: Time to live in seconds

        Returns:
            True if renewal was successful, False if leadership was lost
        """
        pass

    @abstractmethod
    async def release_leadership(self):
        """Release the current leadership lock."""
        pass

    def is_leader(self) -> bool:
        """Check if this instance is the current leader."""
        return self._is_leader

    def _set_leader(self, is_leader: bool):
        """Internal method to set leadership status."""
        was_leader = self._is_leader
        self._is_leader = is_leader
        if was_leader != is_leader:
            logger.info(f"Leadership changed: {was_leader} -> {is_leader}")

    # Pub/Sub
    @abstractmethod
    async def publish(self, channel: str, event: Event):
        """
        Publish an event to a channel.

        Args:
            channel: Channel name (e.g., 'model', 'worker')
            event: Event to publish
        """
        pass

    def subscribe(self, channel: str, callback: Callable[[Event], Any]):
        """
        Subscribe to a channel.

        Implementations MUST invoke ``callback`` on the main asyncio event
        loop. Coordinators whose underlying driver delivers events from a
        background thread must bridge to the main loop themselves (e.g. via
        ``loop.call_soon_threadsafe``) before calling the callback.

        Args:
            channel: Channel name
            callback: Function to call when event is received
        """
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)
        logger.debug(f"Subscribed to channel: {channel}")

    def unsubscribe(self, channel: str, callback: Callable[[Event], Any]):
        """Unsubscribe from a channel."""
        if channel in self._subscribers:
            self._subscribers[channel].remove(callback)
            if not self._subscribers[channel]:
                del self._subscribers[channel]

    def _notify_local_subscribers(self, channel: str, event: Event):
        """Notify local subscribers of an event."""
        if channel in self._subscribers:
            for callback in self._subscribers[channel]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
