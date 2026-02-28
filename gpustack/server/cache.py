import asyncio
import logging
import functools
from typing import Any, Callable
from cachetools import LRUCache
from aiocache import Cache, BaseCache

from gpustack import envs

logger = logging.getLogger(__name__)

cache = Cache(Cache.MEMORY)
# Cache locks for locked_cached decorator
# Locks are created per cache key and should be cleaned up when cache expires
# Using LRUCache from cachetools for automatic LRU eviction
_cache_locks: LRUCache[str, asyncio.Lock] = LRUCache(
    maxsize=envs.SERVER_CACHE_LOCKS_MAX_SIZE
)


def build_cache_key(func: Callable, *args, **kwargs):
    if kwargs is None:
        kwargs = {}
    ordered_kwargs = sorted(kwargs.items())
    return func.__qualname__ + str(args) + str(ordered_kwargs)


async def delete_cache_by_key(func=None, *args, **kwargs):
    key = kwargs.pop("_key", None)
    if key is None:
        if func is None:
            raise ValueError("Either func or key must be provided")
        key = build_cache_key(func, *args, **kwargs)
    logger.trace(f"Deleting cache for key: {key}")
    await cache.delete(key)
    _cache_locks.pop(key, None)


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
