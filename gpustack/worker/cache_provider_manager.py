"""The cache-provider catalog, as a worker sees it.

A worker launches cache servers from the catalog: the image a version runs, the
command, the ports, the health probe. That catalog is no longer a file the
worker can read for itself — an admin may replace it with a document of their
own, which lives on the server — so the worker holds a copy fetched from the API
and refreshed in the background.

Nothing is cached to disk on purpose. A worker that cannot reach the server
cannot start an instance anyway (it has no instance to start, and nowhere to
report one), and a stale catalog on disk would make an instance's declaration
depend on when that worker last had a connection.
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from gpustack.client import ClientSet
from gpustack.schemas.cache_providers import CacheProvider

logger = logging.getLogger(__name__)

# Minimum spacing between miss-driven refreshes, so lookups for a provider the
# catalog genuinely does not carry cannot hammer the API.
_REFRESH_MIN_INTERVAL_SECONDS = 30

_cache_lock = threading.RLock()


class CacheProviderManager:
    """Thread-safe access to the catalog the server serves."""

    def __init__(self, clientset_getter: Callable[[], ClientSet]):
        # A getter, not the client: the worker builds its managers before it
        # has registered and has a client to give them.
        self._clientset_getter = clientset_getter
        # Lower-cased provider name -> declaration. Empty and ``_loaded`` false
        # until the first successful fetch, which is what tells a caller apart
        # from a catalog that genuinely carries no such provider.
        self._providers: Dict[str, CacheProvider] = {}
        self._loaded = False
        self._last_refresh: float = 0.0

    @property
    def loaded(self) -> bool:
        """Whether this worker has ever read the catalog."""
        with _cache_lock:
            return self._loaded

    def get(self, name: Optional[str]) -> Optional[CacheProvider]:
        """The declaration of a provider, refreshing once if it is not cached.

        A provider added to the catalog after the last refresh is the normal
        reason for a miss — an instance of it can be scheduled here before the
        next periodic pass — so a miss is worth one fetch.
        """
        key = (name or "").lower()
        with _cache_lock:
            provider = self._providers.get(key)
        if provider is not None:
            return provider
        if self.refresh():
            with _cache_lock:
                return self._providers.get(key)
        return None

    def refresh(self, force: bool = False) -> bool:
        """Re-fetch the catalog. Answers whether a fetch actually ran and
        succeeded; throttled unless ``force``."""
        with _cache_lock:
            # Claim the throttle slot before fetching, so concurrent callers
            # (instance starts, the health loop) cannot issue duplicate
            # fetches; a failed fetch keeps the slot claimed, which spaces out
            # retries against a server that is struggling.
            if (
                not force
                and time.monotonic() - self._last_refresh
                < _REFRESH_MIN_INTERVAL_SECONDS
            ):
                return False
            self._last_refresh = time.monotonic()
        try:
            providers = self._fetch()
        except Exception as e:
            logger.error(f"Failed to read the cache provider catalog: {e}")
            return False
        with _cache_lock:
            self._providers = {
                provider.name.lower(): provider for provider in providers
            }
            self._loaded = True
        logger.debug(
            f"Cache provider catalog refreshed: "
            f"{', '.join(sorted(self._providers)) or 'no providers'}"
        )
        return True

    def sync(self) -> None:
        """The periodic pass. Unthrottled: its own cadence is the throttle."""
        self.refresh(force=True)

    def _fetch(self) -> List[CacheProvider]:
        # page=0 asks the route for every provider in one response: a catalog
        # is a handful of declarations, and paging through it would only add
        # round trips.
        response = (
            self._clientset_getter()
            .http_client.get_httpx_client()
            .get("/cache-providers", params={"page": 0})
        )
        response.raise_for_status()
        payload = response.json() or {}
        return [
            CacheProvider.model_validate(item) for item in payload.get("items") or []
        ]
