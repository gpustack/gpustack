"""
Local cache for detecting changes in cross-instance events.

When using PostgreSQL-based pub/sub, only the ID is transmitted across instances.
Subscribers use this cache to store the previous state and detect what fields changed.
"""

import logging
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar
from cachetools import LRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ChangeDetector(Generic[T]):
    """
    Detects changes between old and new object states.

    Usage:
        detector = ChangeDetector[Worker]("worker")

        # On first event
        worker = await Worker.one_by_id(session, event_id)
        detector.put(worker.id, worker)

        # On subsequent events
        old_worker = detector.get(event_id)  # Get cached old state
        worker = await Worker.one_by_id(session, event_id)  # Query new state
        changed_fields = detector.detect_changes(old_worker, worker)
        detector.put(worker.id, worker)  # Update cache
    """

    def __init__(self, entity_name: str, maxsize: int = 10000):
        self._entity_name = entity_name
        self._cache: LRUCache[Any, T] = LRUCache(maxsize=maxsize)

    def get(self, id: Any) -> Optional[T]:
        """Get cached object by ID."""
        return self._cache.get(id)

    def put(self, id: Any, obj: T) -> None:
        """Cache an object."""
        self._cache[id] = obj

    def remove(self, id: Any) -> None:
        """Remove an object from cache."""
        self._cache.pop(id, None)

    def detect_changes(
        self, old_obj: Optional[T], new_obj: T
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Detect field changes between old and new object.

        For list (relationship) fields, emit a ``(removed, added)`` delta
        matching the shape produced by the local ``find_history`` hook in
        ``active_record.py``, so callbacks work identically on local and
        cross-instance paths. Fields that fail to load (e.g. lazy relationship
        on a detached instance) are silently skipped.

        Returns:
            Dict of field_name -> (old_value, new_value) for scalar fields, or
            (removed_list, added_list) for relationship fields.
        """
        if old_obj is None:
            return {}

        changed_fields = {}

        # Get fields to compare (exclude internal SQLModel fields)
        fields_to_compare = getattr(new_obj, "model_fields", None)
        if fields_to_compare is None:
            # Fallback: compare all attributes
            fields_to_compare = [
                attr
                for attr in dir(new_obj)
                if not attr.startswith("_")
                and not callable(getattr(new_obj, attr, None))
            ]

        for field_name in fields_to_compare:
            if field_name.startswith("_"):
                continue

            try:
                old_val = getattr(old_obj, field_name, None)
                new_val = getattr(new_obj, field_name, None)

                if isinstance(old_val, list) or isinstance(new_val, list):
                    old_list = old_val if isinstance(old_val, list) else []
                    new_list = new_val if isinstance(new_val, list) else []
                    diff = self._list_diff(old_list, new_list)
                    if diff is not None:
                        changed_fields[field_name] = diff
                    continue

                if old_val != new_val:
                    changed_fields[field_name] = (old_val, new_val)
            except Exception as e:
                logger.debug(
                    f"Error comparing field {field_name} for {self._entity_name}: {e}"
                )
                continue

        return changed_fields

    @staticmethod
    def _list_diff(old_list: list, new_list: list) -> Optional[Tuple[list, list]]:
        """Return a ``(removed, added)`` delta between two relationship lists.

        Elements are keyed by ``.id`` (attribute) or ``["id"]`` (dict). If any
        element is keyless, fall back to whole-list equality and emit an empty
        delta to signal a change without trying to attribute add/remove.

        Returns None when the lists are equivalent (no change).
        """

        def key_of(item: Any) -> Any:
            if item is None:
                return None
            if hasattr(item, "id"):
                return getattr(item, "id")
            if isinstance(item, dict):
                return item.get("id")
            return None

        old_keys = [key_of(o) for o in old_list]
        new_keys = [key_of(n) for n in new_list]

        if any(k is None for k in old_keys) or any(k is None for k in new_keys):
            # Keyless elements — can't reliably attribute add/remove.
            return None if old_list == new_list else ([], [])

        old_set = set(old_keys)
        new_set = set(new_keys)
        if old_set == new_set:
            return None

        removed = [o for o, k in zip(old_list, old_keys) if k not in new_set]
        added = [n for n, k in zip(new_list, new_keys) if k not in old_set]
        return (removed, added)


class EventCacheManager:
    """
    Manages change detectors for different entity types.

    This is a singleton-style manager that provides cached change detection
    for cross-instance events where only IDs are transmitted.
    """

    def __init__(self):
        self._detectors: Dict[str, ChangeDetector] = {}
        self._preloaded: set = set()

    def get_detector(self, entity_name: str) -> ChangeDetector:
        """Get or create a change detector for an entity type."""
        if entity_name not in self._detectors:
            self._detectors[entity_name] = ChangeDetector(entity_name)
        return self._detectors[entity_name]

    def is_preloaded(self, entity_name: str) -> bool:
        """Check if an entity type has been preloaded."""
        return entity_name in self._preloaded

    def mark_preloaded(self, entity_name: str) -> None:
        """Mark an entity type as preloaded."""
        self._preloaded.add(entity_name)

    def clear(self) -> None:
        """Clear all caches."""
        self._detectors.clear()
        self._preloaded.clear()


# Global cache manager instance
_cache_manager = EventCacheManager()


def get_change_detector(entity_name: str) -> ChangeDetector:
    """Get a change detector for the specified entity type."""
    return _cache_manager.get_detector(entity_name)


def clear_all_caches() -> None:
    """Clear all event caches. Useful for testing."""
    _cache_manager.clear()


async def preload_cache(entity_name: str, model_class, session) -> int:
    """
    Preload cache for an entity type by querying all records.

    This ensures that the first cross-instance event can detect changes correctly.
    Without preloading, the first event will have empty changed_fields.

    Args:
        entity_name: The entity type name (e.g., 'worker', 'model')
        model_class: The SQLModel class to query
        session: Database session

    Returns:
        Number of records cached

    Example:
        async with async_session() as session:
            count = await preload_cache('worker', Worker, session)
            logger.info(f"Preloaded {count} workers into cache")
    """
    manager = _cache_manager
    if manager.is_preloaded(entity_name):
        return 0

    detector = manager.get_detector(entity_name)
    records = await model_class.all(session)

    for record in records:
        if hasattr(record, 'id'):
            detector.put(record.id, record)

    manager.mark_preloaded(entity_name)
    logger.info(f"Preloaded {len(records)} {entity_name} records into event cache")
    return len(records)
