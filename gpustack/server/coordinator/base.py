"""
Coordinator Abstract Base Class.

This module defines the interface for coordinating multiple server instances.
Ships with a local (single-node) implementation; plugins can contribute
distributed implementations (e.g. Redis- or PostgreSQL-backed coordinators).
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
    """A change on a topic, as seen by a subscriber.

    ``data`` has **two** shapes and every consumer has to handle both:

    * the hydrated model object, on the local path and whenever the bus
      could re-read the row from the database, and
    * the id-only ``{"id": N}`` dict, when it could not.

    The second shape is not a degradation to be fixed -- it is what a
    cross-instance DELETE genuinely carries. :meth:`to_dict` reduces the
    payload to an id on the wire (a full row does not fit a NOTIFY payload),
    and by the time the event lands on another instance the row is gone, so
    there is nothing left to re-read. ``EventBus._process_coordinator_event``
    falls back to its ChangeDetector cache, which the bus warms from the
    startup preload, from cross-instance events, and from writes this
    instance serves -- but a restart, a dropped event or an LRU eviction all
    still leave it cold, and then the id is all a subscriber gets.

    **Only DELETE.** CREATED and UPDATED are re-read from the database on
    arrival and delivered hydrated, or dropped if the row is already gone --
    they never reach a subscriber id-only. So a handler that needs fields the
    payload cannot supply is choosing what to do about a *deletion*, and
    guarding on the shape does not cost it any update. See
    ``tests/server/test_bus.py`` for the pinned matrix.

    So attribute access on ``data`` is only safe once you have established
    the shape. Use :func:`event_field` to read a field and
    :func:`resolve_event_id` to get the id.

    ``id`` is the reliable identifier for a row-backed event: it is derived
    from ``data`` when the event is constructed locally and carried
    explicitly over the wire, so it survives both paths. It is ``None`` for
    an event that stands for no row -- HEARTBEAT, and anything built with
    ``data=None`` -- so callers still check before using it.
    """

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


def event_field(data: Any, name: str, default: Any = None) -> Any:
    """Read one field off an event payload, whatever shape it arrived in.

    See :class:`Event` for why there is more than one shape.

    ``default`` covers three cases that callers here treat alike: no payload,
    a field the payload could not carry, and a field carried as ``None``.
    They all read as "nothing to act on", which is the branch every caller
    wants, so collapsing them keeps that a single check. It does mean a NULL
    column is indistinguishable from an absent one -- read ``data`` directly
    if you ever need to tell those apart.

    Takes the **payload**, not the :class:`Event`. Passing an event would
    otherwise find no matching attribute and quietly return ``default``, so
    that mistake raises instead.
    """
    if isinstance(data, Event):
        raise TypeError(
            "event_field takes Event.data, not the Event; "
            "use event_field(event.data, ...) or resolve_event_id(event)"
        )
    if data is None:
        return default
    if isinstance(data, dict):
        value = data.get(name)
    else:
        value = getattr(data, name, None)
    return default if value is None else value


def resolve_event_id(event: "Event") -> Optional[Any]:
    """Return the primary key an event refers to, for either payload shape.

    Prefer this over ``event.data.id``, which raises ``AttributeError`` on an
    id-only payload -- the single most common way to get :class:`Event`
    wrong, and one that only shows up on a replica that did not serve the
    write.

    ``Event.id`` is populated on both paths: ``__post_init__`` derives it
    from ``data`` locally, and it is carried explicitly over the wire.
    ``None`` means the event stands for no row (HEARTBEAT, ``data=None``).
    """
    return event.id


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
    def is_distributed(self) -> bool:
        """Whether events reach this process from other instances.

        Defaults to True: a coordinator exists to distribute, and the
        single-node implementation is the exception. Anything keyed on the
        id-only payload shape should gate on this -- that shape only arises
        on the cross-instance path, so work done to anticipate it is pure
        overhead where there is no such path.
        """
        return True

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
