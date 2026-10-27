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

import enum
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from gpustack.client import ClientSet
from gpustack.schemas.cache_providers import CacheProvider

logger = logging.getLogger(__name__)

# Minimum spacing between miss-driven refreshes, so lookups for a provider the
# catalog genuinely does not carry cannot hammer the API.
_REFRESH_MIN_INTERVAL_SECONDS = 30

_cache_lock = threading.RLock()


class RefreshOutcome(enum.Enum):
    """What a refresh did, which is what a miss after it means.

    Only ``FETCHED`` makes a miss authoritative: the catalog was read just now
    and does not carry the provider. The other two say the copy on hand is
    whatever it was, so a miss against it says nothing about the catalog.
    """

    FETCHED = "fetched"
    THROTTLED = "throttled"
    FAILED = "failed"


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
        # When the copy on hand was fetched, which is not when the last fetch
        # was attempted: the throttle slot is claimed before fetching and kept
        # on failure, so a failed attempt would otherwise pass for a recent
        # read of the catalog.
        self._loaded_at: Optional[float] = None
        self._last_refresh: float = 0.0

    @property
    def loaded(self) -> bool:
        """Whether this worker has ever read the catalog."""
        with _cache_lock:
            return self._loaded

    def lookup(self, name: Optional[str]) -> Tuple[Optional[CacheProvider], bool]:
        """The declaration of a provider, and whether the catalog it was looked
        up in is one this worker could actually read.

        The two say different things about a miss. A catalog read successfully
        that does not carry the provider is a configuration fact — the
        declaration is gone, and the instance cannot start. A read that failed,
        or a copy too old to trust, is a connectivity fact: the provider may
        well exist, and reporting it as unknown would blame the wrong thing.
        """
        key = (name or "").lower()
        with _cache_lock:
            provider = self._providers.get(key)
        if provider is not None:
            return provider, True
        # A provider added to the catalog after the last pass is the normal
        # reason for a miss — an instance of it can be scheduled here before
        # the next one — so a miss is worth a fetch. A throttled or failed one
        # leaves the copy on hand as it was, and a miss against that says
        # nothing about what the catalog carries.
        outcome = self.refresh()
        with _cache_lock:
            # A throttled refresh protects a copy fetched within the window,
            # which is recent enough to answer a miss: reporting it as a
            # catalog this worker could not read would blame connectivity for
            # a provider that genuinely is not declared. Measured from the last
            # successful fetch, not the last attempt — a failed one claims the
            # throttle slot too, and answering from the copy it left standing
            # is the conflation this is here to avoid.
            authoritative = outcome is RefreshOutcome.FETCHED or (
                outcome is RefreshOutcome.THROTTLED
                and self._loaded_at is not None
                and time.monotonic() - self._loaded_at < _REFRESH_MIN_INTERVAL_SECONDS
            )
            return self._providers.get(key), authoritative

    def get(self, name: Optional[str]) -> Optional[CacheProvider]:
        """The declaration alone, for callers with nothing to say about why it
        is missing."""
        return self.lookup(name)[0]

    def reread(self, name: Optional[str]) -> Optional[CacheProvider]:
        """The declaration as the server has it now, fetched whatever the
        throttle says.

        A name that is present does not mean the declaration behind it is
        current: this worker's copy can predate the document a service was
        created against, and a provider the packaged catalog holds a
        placeholder for is present under both. What says the copy is stale is
        the service asking it for something it does not carry.
        """
        self.refresh(force=True)
        with _cache_lock:
            return self._providers.get((name or "").lower())

    def refresh(self, force: bool = False) -> RefreshOutcome:
        """Re-fetch the catalog, answering what happened: throttled unless
        ``force``, and a fetch that ran either landed or failed."""
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
                return RefreshOutcome.THROTTLED
            self._last_refresh = time.monotonic()
        try:
            providers = self._fetch()
        except Exception as e:
            logger.error(f"Failed to read the cache provider catalog: {e}")
            return RefreshOutcome.FAILED
        with _cache_lock:
            self._providers = {
                provider.name.lower(): provider for provider in providers
            }
            self._loaded = True
            self._loaded_at = time.monotonic()
        logger.debug(
            f"Cache provider catalog refreshed: "
            f"{', '.join(sorted(self._providers)) or 'no providers'}"
        )
        return RefreshOutcome.FETCHED

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
