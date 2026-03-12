import asyncio
import json
import logging
import threading
from typing import Any, Callable, Dict, Optional, Union, Awaitable

import httpx
from gpustack.api.exceptions import (
    raise_if_response_error,
    async_raise_if_response_error,
)
from gpustack.server.bus import Event, EventType
from gpustack.schemas import *
from gpustack.schemas.common import Pagination

from .generated_http_client import HTTPClient

logger = logging.getLogger(__name__)


class BenchmarkClient:
    def __init__(self, client: HTTPClient, enable_cache: bool = True):
        self._client = client
        self._url = "/benchmarks"
        self._enable_cache = enable_cache
        self._cache: Dict[int, BenchmarkPublic] = {}
        self._cache_lock = None
        self._watch_started = False
        self._initial_sync_logged = False

    def _get_cache_lock(self):
        """Lazy initialization of cache lock."""
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        return self._cache_lock

    def list(
        self, params: Dict[str, Any] = None, use_cache: bool = True
    ) -> BenchmarksPublic:
        """
        List resources.

        Args:
            params: Query parameters for filtering
            use_cache: Whether to use cache. Defaults to True (use cache if available).
                      Automatically falls back to API if cache watch is not running.
                      Note: If 'page' or 'perPage' params are provided, always calls API.

        Returns:
            List of resources
        """
        # Determine if we should use cache
        # Don't use cache if pagination params are provided
        pagination_params = {"page", "perPage"} if params else set()
        has_pagination = any(k in pagination_params for k in (params or {}))

        should_use_cache = (
            use_cache
            and self._enable_cache
            and self._watch_started
            and not has_pagination  # Don't use cache if pagination params exist
        )

        # If cache should be used, try to read from cache
        if should_use_cache:
            return self._list_from_cache(params)

        # Otherwise, make API call
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)

        return BenchmarksPublic.model_validate(response.json())

    def _list_from_cache(self, params: Dict[str, Any] = None) -> BenchmarksPublic:
        """
        List resources from cache instead of making an API call.

        Note: Cache is automatically populated when awatch() is called.
        The first call to awatch() will set _watch_started=True and enable caching.
        """
        # Get all cached items
        with self._get_cache_lock():
            all_items = list(self._cache.values())

        # Apply filters if params provided
        if params:
            filtered_items = []
            for item in all_items:
                match = True
                for key, value in params.items():
                    # Skip non-filter params like 'watch'
                    if key == 'watch':
                        continue
                    # Convert attribute to string for comparison
                    attr_value = getattr(item, key, None)
                    if attr_value is not None and str(attr_value) != str(value):
                        match = False
                        break
                if match:
                    filtered_items.append(item)
            all_items = filtered_items

        # Return in the same format as the original list()
        total = len(all_items)

        # Create pagination info for PaginatedList types
        pagination = Pagination(
            page=1,
            perPage=total if total > 0 else 100,
            total=total,
            totalPage=1 if total > 0 else 0,
        )

        return BenchmarksPublic(items=all_items, total=total, pagination=pagination)

    def _update_cache_from_event(self, event: Event):
        """Update cache based on received event."""
        if not self._enable_cache:
            return

        try:
            item = BenchmarkPublic.model_validate(event.data)
            if not hasattr(item, 'id'):
                return

            with self._get_cache_lock():
                if event.type == EventType.DELETED:
                    self._cache.pop(item.id, None)
                    logger.debug(f"Cache: removed benchmark {item.id}")
                else:  # CREATED or UPDATED
                    self._cache[item.id] = item
                    logger.trace(f"Cache: updated benchmark {item.id}")
        except Exception as e:
            logger.error(f"Failed to update benchmarks cache from event: {e}")

    def watch(
        self,
        callback: Optional[Callable[[Event], None]] = None,
        stop_condition: Optional[Callable[[Event], bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = {}
        params["watch"] = "true"

        if stop_condition is None:
            stop_condition = lambda event: False

        with self._client.get_httpx_client().stream(
            "GET", self._url, params=params, timeout=None
        ) as response:
            raise_if_response_error(response)
            for line in response.iter_lines():
                if line:
                    event_data = json.loads(line)
                    event = Event(**event_data)
                    if callback:
                        callback(event)
                    if stop_condition(event):
                        break

    async def awatch(
        self,
        callback: Optional[
            Union[Callable[[Event], None], Callable[[Event], Awaitable[Any]]]
        ] = None,
        stop_condition: Optional[Callable[[Event], bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = {}
        params["watch"] = "true"

        if stop_condition is None:
            stop_condition = lambda event: False

        # Mark watch as started when awatch is called
        # This enables list()/get() to use cache automatically
        if self._enable_cache and not self._watch_started:
            self._watch_started = True
            logger.debug(f"benchmarks cache watch started")

        async with self._client.get_async_httpx_client().stream(
            "GET",
            self._url,
            params=params,
            timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10),
        ) as response:
            await async_raise_if_response_error(response)
            lines = response.aiter_lines()
            while True:
                try:
                    line = await asyncio.wait_for(lines.__anext__(), timeout=45)
                    if line:
                        event_data = json.loads(line)
                        event = Event(**event_data)

                        # Update cache if enabled
                        if self._enable_cache:
                            self._update_cache_from_event(event)

                            # Log cache size after initial events (approximately)
                            if (
                                not self._initial_sync_logged
                                and event.type == EventType.CREATED
                            ):
                                # Check if we have accumulated enough items (heuristic)
                                with self._get_cache_lock():
                                    cache_size = len(self._cache)
                                if cache_size > 0:
                                    # Set a flag to avoid repeated logging
                                    self._initial_sync_logged = True
                                    logger.debug(
                                        f"benchmarks cache populated with {cache_size} items"
                                    )

                        if callback:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)
                        if stop_condition(event):
                            break
                except asyncio.TimeoutError:
                    raise Exception("watch timeout")

    def get(self, id: int, use_cache: bool = True) -> BenchmarkPublic:
        """
        Get a resource by ID.

        Args:
            id: Resource ID
            use_cache: Whether to use cache. Defaults to True (use cache if available).
                      Automatically falls back to API if cache watch is not running.

        Returns:
            Resource object
        """
        # Use cache if enabled, watch is running, and use_cache is True
        should_use_cache = use_cache and self._enable_cache and self._watch_started

        # Try to get from cache first if it should be used
        if should_use_cache:
            with self._get_cache_lock():
                if id in self._cache:
                    logger.trace(f"Cache hit for benchmark {id}")
                    return self._cache[id]

        # Fall back to API call
        response = self._client.get_httpx_client().get(f"{self._url}/{id}")
        raise_if_response_error(response)
        result = BenchmarkPublic.model_validate(response.json())

        # Update cache if enabled
        if self._enable_cache:
            with self._get_cache_lock():
                self._cache[id] = result

        return result

    def create(self, model_create: BenchmarkCreate):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return BenchmarkPublic.model_validate(response.json())

    def update(self, id: int, model_update: BenchmarkUpdate):
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}",
            content=model_update.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return BenchmarkPublic.model_validate(response.json())

    def delete(self, id: int):
        response = self._client.get_httpx_client().delete(f"{self._url}/{id}")
        raise_if_response_error(response)
