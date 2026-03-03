import inspect
import asyncio
import logging
import functools
from typing import Any, Callable, Optional, TYPE_CHECKING
from cachetools import LRUCache
from aiocache import Cache, BaseCache

from gpustack import envs
from gpustack.server.coordinator.base import Event, EventType

if TYPE_CHECKING:
    from gpustack.server.coordinator.base import Coordinator

logger = logging.getLogger(__name__)

cache = Cache(Cache.MEMORY)
# Cache locks for locked_cached decorator
# Locks are created per cache key and should be cleaned up when cache expires
# Using LRUCache from cachetools for automatic LRU eviction
_cache_locks: LRUCache[str, asyncio.Lock] = LRUCache(
    maxsize=envs.SERVER_CACHE_LOCKS_MAX_SIZE
)

# Global coordinator reference for distributed cache synchronization
_coordinator: Optional["Coordinator"] = None


def set_coordinator(coordinator: Optional["Coordinator"]) -> None:
    """Set the coordinator for distributed cache synchronization.

    This is called during server startup to enable cache invalidation
    broadcasting across instances.
    """
    global _coordinator
    _coordinator = coordinator
    if coordinator:
        # Subscribe to cache invalidation events
        coordinator.subscribe("cache", _handle_cache_invalidate)
        logger.debug("Distributed cache synchronization enabled")


def _handle_cache_invalidate(event: "Event") -> None:
    """Handle cache invalidation events from other instances."""
    if event.type == EventType.DELETED and event.data:
        key = event.data.get("key")
        if key:
            # Use asyncio.create_task since this is called from sync context
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_local_delete_cache(key))
            except RuntimeError:
                # No event loop running, ignore
                pass


async def _local_delete_cache(key: str) -> None:
    """Delete cache locally without broadcasting (for remote events)."""
    logger.trace(f"Deleting cache for key: {key} (from remote)")
    await cache.delete(key)
    _cache_locks.pop(key, None)


async def _broadcast_invalidation(key: str) -> None:
    """Broadcast cache invalidation to other instances."""
    if _coordinator is None:
        return

    try:
        await _coordinator.publish(
            "cache", Event(type=EventType.DELETED, data={"key": key})
        )
        logger.trace(f"Broadcasted cache invalidation for key: {key}")
    except Exception as e:
        logger.warning(f"Failed to broadcast cache invalidation: {e}")


def build_cache_key(func: Callable, *args, **kwargs):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    # locked_cached.decorator strips 'self' before calling here, but unbound
    # functions still have 'self' in their signature. Strip it so keys match
    # when delete_cache_by_key is called with a bound method (no self in sig).
    if params and params[0].name in ("self", "cls") and not hasattr(func, "__self__"):
        sig = sig.replace(parameters=params[1:])
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        # bound.arguments follows declaration order, so kwargs ordering is stable.
        return func.__qualname__ + str(tuple(bound.arguments.values()))
    except TypeError:
        # Fallback for callers that pass args not matching the function signature
        # (e.g. build_cache_key used as a manual key-construction helper).
        # Sort kwargs for a stable key regardless of call-site ordering.
        return func.__qualname__ + str(args) + str(sorted(kwargs.items()))


async def delete_cache_by_key(
    func=None, *args, sync_coordinator: bool = True, **kwargs
):
    """Delete cache by key or function.

    Args:
        func: The cached function (optional)
        *args: Arguments to build the cache key
        sync_coordinator: Whether to broadcast invalidation to other instances via coordinator.
                         Default is True for security-sensitive data.
                         Set to False for high-frequency, non-critical caches (e.g., Worker status).
        **kwargs: Additional arguments including `_key` for explicit key
    """
    key = kwargs.pop("_key", None)
    if key is None:
        if func is None:
            raise ValueError("Either func or key must be provided")
        key = build_cache_key(func, *args, **kwargs)
    logger.trace(f"Deleting cache for key: {key}")
    await cache.delete(key)
    _cache_locks.pop(key, None)

    # Broadcast to other instances via coordinator
    if sync_coordinator:
        await _broadcast_invalidation(key)


async def set_cache_by_key(key: str, value: Any):
    logger.trace(f"Set cache for key: {key}")
    await cache.set(key, value)


def class_key(suffix: str):
    """Generate a cache key builder for class methods.

    Usage:
        @locked_cached(key=class_key("all_cached"))
        async def cached_all(cls, session, ...):
            ...

    The generated key will be "{ClassName}.{suffix}", e.g., "Worker.all_cached"
    """

    # FIXME: The kwargs should be taken into account for more fine-grained cache keys,
    # but for now we just use the class name and suffix for simplicity.
    # Using kwargs as key causes https://github.com/gpustack/gpustack/issues/4813.
    def builder(f, *args, **kwargs):
        cls = args[0]  # First arg is cls for classmethod
        return f"{cls.__name__}.{suffix}"

    return builder


class locked_cached:
    def __init__(
        self,
        ttl: int = envs.SERVER_CACHE_TTL_SECONDS,
        cache: BaseCache = cache,
        key: str = None,
    ):
        self.cache = cache
        self.ttl = ttl
        self.key = key

    def __call__(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            return await self.decorator(f, *args, **kwargs)

        wrapper.cache = self.cache
        wrapper.cache_key = self.key
        return wrapper

    async def get_from_cache(self, key: str):
        return await self.cache.get(key)

    async def set_in_cache(self, key: str, value: Any):
        await self.cache.set(key, value, ttl=self.ttl)

    async def decorator(self, f, *args, **kwargs):
        if self.key is not None:
            key = self.key(f, *args, **kwargs) if callable(self.key) else self.key
        else:
            # no self arg
            key = build_cache_key(f, *args[1:], **kwargs)
        value = await self.get_from_cache(key)
        if value is not None:
            return value

        lock = _cache_locks.setdefault(key, asyncio.Lock())

        async with lock:
            value = await self.get_from_cache(key)
            if value is not None:
                return value

            logger.trace(f"cache miss for key: {key}")
            result = await f(*args, **kwargs)
            if result is not None:
                await self.set_in_cache(key, result)

        return result
