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


class UserClient:
    def __init__(self, client: HTTPClient, enable_cache: bool = True):
        self._client = client
        self._url = "/users"
        self._enable_cache = enable_cache
        self._cache: Dict[int, UserPublic] = {}
        self._cache_lock = threading.Lock()
        self._watch_started = False
        self._initial_sync_logged = False

    def list(
        self, params: Dict[str, Any] = None, use_cache: bool = True
    ) -> UsersPublic:
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

        if use_cache and self._enable_cache and not has_pagination:
            cached = self._list_from_cache(params)
            if cached is not None:
                return cached

        # Fall back to API call
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)

        return UsersPublic.model_validate(response.json())

    def _list_from_cache(self, params: Dict[str, Any] = None) -> Optional[UsersPublic]:
        """
        Snapshot the cache atomically with the _watch_started check.

        Returns None if the watch cache is not currently authoritative —
        caller should fall back to a direct API call.
        """
        # Atomically check _watch_started and snapshot the items so that
        # awatch's teardown (which flips _watch_started=False and clears
        # the cache under the same lock) can never leave us reading an
        # empty cache while still believing it's authoritative.
        with self._cache_lock:
            if not self._watch_started:
                return None
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

        return UsersPublic(items=all_items, total=total, pagination=pagination)

    async def _update_cache_from_event(self, event: Event):
        """Update cache based on received event.

        Runs on the awatch event loop. Network I/O uses the async httpx
        client and happens outside the cache lock, so concurrent readers
        (list/get) are never blocked waiting on HTTP.
        """
        if not self._enable_cache:
            return

        try:
            # Server only emits ID-only events for DELETED (when its own
            # enrichment cache misses on a row that's already gone from DB).
            # CREATED/UPDATED are always enriched server-side or dropped, so
            # we only handle the DELETED case here.
            is_id_only_delete = (
                event.type == EventType.DELETED
                and isinstance(event.data, dict)
                and event.id is not None
                and set(event.data.keys()) == {"id"}
            )
            if is_id_only_delete:
                with self._cache_lock:
                    item = self._cache.pop(event.id, None)
                if item is not None:
                    # Enrich so downstream callbacks (e.g. ServeManager) see
                    # a validated object instead of {"id": ...}.
                    event.data = item
                logger.debug(f"Cache: removed user {event.id}")
                return

            item = UserPublic.model_validate(event.data)
            if not hasattr(item, 'id'):
                return

            with self._cache_lock:
                if event.type == EventType.DELETED:
                    self._cache.pop(item.id, None)
                    logger.debug(f"Cache: removed user {item.id}")
                else:  # CREATED or UPDATED
                    self._cache[item.id] = item
                    logger.trace(f"Cache: updated user {item.id}")
        except Exception as e:
            logger.error(f"Failed to update users cache from event: {e}")

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

        try:
            async with self._client.get_async_httpx_client().stream(
                "GET",
                self._url,
                params=params,
                timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10),
            ) as response:
                await async_raise_if_response_error(response)

                # Connection is up. The server's replay_existing=True snapshot
                # that follows is authoritative, so drop any stale entries
                # from a prior session — items deleted while we were
                # disconnected won't get a DELETED event (already gone from
                # DB) and would otherwise persist in the cache forever.
                # Flip _watch_started here (not before connect) so that a
                # failed connect doesn't leave list()/get() reading an empty
                # cache while believing it's authoritative.
                first_start = False
                if self._enable_cache:
                    with self._cache_lock:
                        self._cache.clear()
                        self._initial_sync_logged = False
                        first_start = not self._watch_started
                        self._watch_started = True
                    if first_start:
                        logger.debug(f"users cache watch started")

                lines = response.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(lines.__anext__(), timeout=45)
                        if line:
                            event_data = json.loads(line)
                            event = Event(**event_data)

                            # Update cache if enabled
                            if self._enable_cache:
                                await self._update_cache_from_event(event)

                                # Log cache size after initial events (approximately)
                                if (
                                    not self._initial_sync_logged
                                    and event.type == EventType.CREATED
                                ):
                                    # Check if we have accumulated enough items (heuristic)
                                    with self._cache_lock:
                                        cache_size = len(self._cache)
                                    if cache_size > 0:
                                        # Set a flag to avoid repeated logging
                                        self._initial_sync_logged = True
                                        logger.debug(
                                            f"users cache populated with {cache_size} items"
                                        )

                            # Skip the callback if the event is still ID-only after
                            # cache update (e.g. DELETED for an item this client
                            # never saw). Subscribers like ServeManager call
                            # model_validate(event.data) and would otherwise fail;
                            # also they can't act without the full object.
                            if (
                                isinstance(event.data, dict)
                                and event.id is not None
                                and set(event.data.keys()) == {"id"}
                            ):
                                logger.debug(
                                    f"Skipping callback for ID-only {event.type} event on users {event.id}"
                                )
                            elif callback:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(event)
                                else:
                                    callback(event)
                            if stop_condition(event):
                                break
                    except StopAsyncIteration:
                        # Surface as an exception (not a clean return) so the
                        # caller's reconnect loop hits its `except` branch and
                        # respects its backoff — otherwise a server that keeps
                        # closing the stream would trigger a tight reconnect loop.
                        raise ConnectionError("Watch stream closed by server")
                    except asyncio.TimeoutError:
                        # Re-raise as asyncio.TimeoutError (not the built-in
                        # TimeoutError) — on Python 3.10 they are distinct
                        # classes and callers may catch the asyncio variant.
                        raise asyncio.TimeoutError("watch timeout")
        finally:
            # When awatch is no longer running the cache is not authoritative
            # — flip _watch_started=False so list()/get() fall back to direct
            # API calls until the next successful (re)connect. Done under
            # the lock together with the clear so concurrent readers can't
            # observe (_watch_started=True, cache=empty).
            if self._enable_cache:
                with self._cache_lock:
                    if self._watch_started:
                        self._watch_started = False
                        self._cache.clear()

    def get(self, id: int, use_cache: bool = True) -> UserPublic:
        """
        Get a resource by ID.

        Args:
            id: Resource ID
            use_cache: Whether to use cache. Defaults to True (use cache if available).
                      Automatically falls back to API if cache watch is not running.

        Returns:
            Resource object
        """
        # Atomically check _watch_started and read from the cache so awatch's
        # teardown can't slip between the two and leave us returning a miss.
        if use_cache and self._enable_cache:
            with self._cache_lock:
                if self._watch_started and id in self._cache:
                    logger.trace(f"Cache hit for user {id}")
                    return self._cache[id]

        # Fall back to API call. Do NOT write the result into the cache:
        # the awatch stream is the single source of truth, and a concurrent
        # DELETED/UPDATED event arriving during this API call could be
        # overwritten by a stale result here.
        response = self._client.get_httpx_client().get(f"{self._url}/{id}")
        raise_if_response_error(response)
        return UserPublic.model_validate(response.json())

    def create(self, model_create: UserCreate):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return UserPublic.model_validate(response.json())

    def update(self, id: int, model_update: UserUpdate):
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}",
            content=model_update.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return UserPublic.model_validate(response.json())

    def delete(self, id: int):
        response = self._client.get_httpx_client().delete(f"{self._url}/{id}")
        raise_if_response_error(response)
