import asyncio
import logging
import threading
import time
from typing import Dict, Optional, Set, Tuple


from gpustack.client import ClientSet
from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict
from gpustack.server.bus import EventType, Event


logger = logging.getLogger(__name__)

# Global lock for cache operations to avoid pickle serialization issues
_cache_lock = threading.RLock()

# Minimum spacing between cache-miss refreshes so repeated lookups for a
# scope the server genuinely doesn't expose can't hammer the API.
_REFRESH_MIN_INTERVAL_SECONDS = 30

# Fields that never fall back from the owner row to the Platform row when
# merging the two scopes: identity/scope keys, the version map (merged
# separately), and row bookkeeping.
_NON_FALLBACK_FIELDS = {
    "backend_name",
    "owner_principal_id",
    "version_configs",
    "id",
    "created_at",
    "updated_at",
    "deleted_at",
}


class InferenceBackendManager:
    """
    Unified singleton manager for InferenceBackend data.

    This class provides thread-safe access to InferenceBackend data
    across the worker layer, combining database operations and real-time listening.
    """

    def __init__(self, clientset: ClientSet):
        # backend_name -> owner_principal_id (None = Platform row) -> row.
        # Rows are cached per Hybrid scope so one Org's update can never
        # clobber another Org's (or the Platform's) custom versions;
        # get_backend_by_name merges the relevant scopes at read time.
        self.backends_cache: Dict[str, Dict[Optional[int], InferenceBackend]] = {}

        # (backend_name, owner_principal_id) scopes a refresh confirmed
        # the server doesn't expose. Negative-cached so the steady-state
        # common case (a model whose Org has no override row — including
        # every Default-Org model) doesn't re-poll /inference-backends/all
        # forever; invalidated by watch events and watch (re)connects.
        self._missing_scopes: Set[Tuple[str, int]] = set()

        # Listener related attributes
        self._clientset: Optional[ClientSet] = clientset
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._last_refresh: float = 0.0

        self._initialize_cache()

    async def start_listener(self) -> None:
        """Start the listener service."""
        if not self._clientset:
            logger.warning("ClientSet not set, cannot start listener")
            return

        if self._running:
            logger.warning("InferenceBackendManager listener is already running")
            return

        self._running = True
        logger.info("Starting InferenceBackend listener service")

        # Start watching for changes
        self._watch_task = asyncio.create_task(self._watch_changes())

    def get_backend_by_name(
        self,
        backend_name: str,
        owner_principal_id: Optional[int] = None,
    ) -> Optional[InferenceBackend]:
        """
        Resolve a backend for the given owner under the Hybrid scope:
        the Platform row merged with the owner's row, the owner's
        version_configs keys overriding the Platform ones. Rows owned by
        other principals are ignored.
        """
        backend = self._resolve_backend(backend_name, owner_principal_id)
        if owner_principal_id is not None and not self._has_scope(
            backend_name, owner_principal_id
        ):
            scope_key = (backend_name, owner_principal_id)
            with _cache_lock:
                known_missing = scope_key in self._missing_scopes
            # The owner's row may have become visible to this worker
            # after the cache was built (e.g. created between the cache
            # fetch and the watch starting) — refresh once before
            # serving the Platform-only fallback. A confirmed miss is
            # negative-cached: most owners legitimately have no override
            # row, and the watch stream delivers any row created later.
            if not known_missing and self._refresh_cache():
                if self._has_scope(backend_name, owner_principal_id):
                    backend = self._resolve_backend(backend_name, owner_principal_id)
                else:
                    with _cache_lock:
                        self._missing_scopes.add(scope_key)
                    logger.warning(
                        f"No backend row for owner {owner_principal_id} of "
                        f"backend {backend_name} after cache refresh; using "
                        f"the Platform configuration only. If this owner is "
                        f"expected to have overrides, the server-side scope "
                        f"doesn't expose them to this worker."
                    )
        return backend

    def _has_scope(self, backend_name: str, owner_principal_id: int) -> bool:
        with _cache_lock:
            return owner_principal_id in self.backends_cache.get(backend_name, {})

    def _resolve_backend(
        self,
        backend_name: str,
        owner_principal_id: Optional[int],
    ) -> Optional[InferenceBackend]:
        with _cache_lock:
            scopes = self.backends_cache.get(backend_name)
            if not scopes:
                return None
            platform_row = scopes.get(None)
            owner_row = (
                scopes.get(owner_principal_id)
                if owner_principal_id is not None
                else None
            )
            if owner_row is None:
                return platform_row
            if platform_row is None:
                return owner_row
            merged = owner_row.model_copy()
            # Field-level fallback for scalars: an Org row may be sparse
            # (e.g. created directly with just one custom version and no
            # health_check_path); unset fields inherit the Platform value
            # instead of shadowing it with None.
            for field_name in type(owner_row).model_fields:
                if field_name in _NON_FALLBACK_FIELDS:
                    continue
                if getattr(merged, field_name) is None:
                    platform_value = getattr(platform_row, field_name)
                    if platform_value is not None:
                        setattr(merged, field_name, platform_value)
            # Booleans carry concrete defaults, so the None-fallback above
            # never reaches them. Align with the server's collapse
            # semantics (_collapse_by_backend_name): ``enabled`` is OR'd
            # so an Org row can't shadow a Platform-enabled backend. The
            # same for ``is_built_in`` — a sparse Org-created override of
            # a built-in defaults to is_built_in=False, which would
            # otherwise flip version resolution onto the non-built-in
            # path (resolve_target_version auto-picks the latest).
            merged.enabled = bool(owner_row.enabled) or bool(platform_row.enabled)
            merged.is_built_in = owner_row.is_built_in or platform_row.is_built_in
            merged.version_configs = VersionConfigDict(
                root={
                    **(
                        platform_row.version_configs.root
                        if platform_row.version_configs
                        else {}
                    ),
                    **(
                        owner_row.version_configs.root
                        if owner_row.version_configs
                        else {}
                    ),
                }
            )
            return merged

    def _fetch_backends(self) -> Dict[str, Dict[Optional[int], InferenceBackend]]:
        """Load all visible backends from the server into a fresh
        per-scope cache dict."""
        resp = self._clientset.http_client.get_httpx_client().get(
            "/inference-backends/all"
        )
        resp.raise_for_status()
        cache: Dict[str, Dict[Optional[int], InferenceBackend]] = {}
        for backend in resp.json() or []:
            backend = InferenceBackend.model_validate(backend)
            if backend:
                cache.setdefault(backend.backend_name, {})[
                    backend.owner_principal_id
                ] = backend
        return cache

    def _initialize_cache(self) -> None:
        """Initialize the cache with existing InferenceBackend data."""
        try:
            logger.info("Initializing InferenceBackend cache")
            cache = self._fetch_backends()
            with _cache_lock:
                self.backends_cache = cache
            self._last_refresh = time.monotonic()
            if cache:
                logger.info(f"Initialized cache with {cache.keys()} InferenceBackends")
            else:
                logger.info("No existing InferenceBackends found")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceBackend cache: {e}")
            raise

    def _refresh_cache(self) -> bool:
        """Re-fetch the cache on a scope miss; throttled, best-effort.

        Returns True when a refresh actually ran.
        """
        with _cache_lock:
            # Claim the throttle slot before fetching so concurrent
            # callers (health-check loop, instance starts) can't issue
            # duplicate fetches; a failed fetch keeps the slot claimed,
            # which also spaces out retries against a struggling server.
            if time.monotonic() - self._last_refresh < _REFRESH_MIN_INTERVAL_SECONDS:
                return False
            self._last_refresh = time.monotonic()
        try:
            cache = self._fetch_backends()
        except Exception as e:
            logger.error(f"Failed to refresh InferenceBackend cache: {e}")
            return False
        with _cache_lock:
            # Known race: a watch event landing between the fetch above
            # and this swap is overwritten by the (older) snapshot until
            # that row's next event. The window is tiny and self-healing,
            # and merging instead of swapping would need delete-tracking
            # — not worth it.
            self.backends_cache = cache
            self._missing_scopes.clear()
        logger.info("Refreshed InferenceBackend cache after scope miss")
        return True

    async def _watch_changes(self) -> None:
        """Watch for InferenceBackend changes and update the cache."""
        while self._running:
            try:
                logger.info("Starting to watch InferenceBackend changes")
                # Events may have been missed while disconnected — drop
                # the confirmed-missing scopes so the next lookup may
                # refresh and pick up rows created during the gap.
                with _cache_lock:
                    self._missing_scopes.clear()
                await self._clientset.inference_backends.awatch(
                    callback=self._handle_event
                )

            except asyncio.CancelledError:
                logger.info("InferenceBackend watch cancelled")
                break
            except Exception as e:
                logger.error(f"Error watching InferenceBackend changes: {e}")
                if self._running:
                    # Wait before retrying
                    await asyncio.sleep(5)

    def _merge_version_configs(
        self,
        old: Optional[InferenceBackend],
        backend: InferenceBackend,
    ) -> None:
        """
        Merge incoming backend version configs into the cached one while
        preserving built-in entries.

        Args:
            old: Previously cached backend for the same name and scope.
            backend: Incoming backend to be merged into the cache.
        """
        if old and backend.is_built_in:
            # Snapshot previous and incoming version maps
            old_version = old.version_configs.root if old.version_configs else {}
            new_version = (
                backend.version_configs.root if backend.version_configs else {}
            )

            # Compute deletions: drop outdated non-built-in entries not present in new map
            delete_version = set()
            new_version_keys = set(new_version.keys())
            for k, v in old_version.items():
                if not v.built_in_frameworks and k not in new_version_keys:
                    delete_version.add(k)

            # Start from old (preserves built-ins), then apply incoming updates
            merged = old_version
            for k, v in new_version.items():
                merged[k] = v

            # Remove marked entries and finalize
            for k in delete_version:
                merged.pop(k, None)
            backend.version_configs.root = merged

    def _handle_event(self, event: Event):
        """Handle a single InferenceBackend event."""
        try:
            # Parse the backend data
            backend = InferenceBackend.model_validate(event.data)
            scope = backend.owner_principal_id

            if event.type == EventType.CREATED or event.type == EventType.UPDATED:
                with _cache_lock:
                    scopes = self.backends_cache.setdefault(backend.backend_name, {})
                    old = scopes.get(scope)
                    self._merge_version_configs(old, backend)
                    scopes[scope] = backend
                    if scope is not None:
                        self._missing_scopes.discard((backend.backend_name, scope))
                logger.debug(
                    f"Updated InferenceBackend in cache: {backend.id} ({event.type})"
                )
            elif event.type == EventType.DELETED:
                with _cache_lock:
                    # Remove only the changed scope; other Orgs' rows and
                    # the Platform row of the same backend stay cached.
                    scopes = self.backends_cache.get(backend.backend_name)
                    if scopes is not None:
                        scopes.pop(scope, None)
                        if not scopes:
                            self.backends_cache.pop(backend.backend_name, None)
                    logger.debug(
                        f"Removed InferenceBackend from cache: {backend.backend_name}"
                    )
            else:
                logger.warning(f"Unknown event type: {event.type}")

        except Exception as e:
            logger.error(f"Error handling InferenceBackend event: {e}")
