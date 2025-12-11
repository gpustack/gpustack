import asyncio
import logging
import threading
from typing import Dict, Optional


from gpustack.client import ClientSet
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.server.bus import EventType, Event


logger = logging.getLogger(__name__)

# Global lock for cache operations to avoid pickle serialization issues
_cache_lock = threading.RLock()


class InferenceBackendManager:
    """
    Unified singleton manager for InferenceBackend data.

    This class provides thread-safe access to InferenceBackend data
    across the worker layer, combining database operations and real-time listening.
    """

    def __init__(self, clientset: ClientSet):
        self.backends_cache: Dict[str, InferenceBackend] = {}

        # Listener related attributes
        self._clientset: Optional[ClientSet] = clientset
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None

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

    def get_backend_by_name(self, backend_name: str) -> Optional[InferenceBackend]:
        with _cache_lock:
            if backend_name in self.backends_cache:
                return self.backends_cache[backend_name]

        return None

    def _initialize_cache(self) -> None:
        """Initialize the cache with existing InferenceBackend data."""
        try:
            logger.info("Initializing InferenceBackend cache")
            resp = self._clientset.http_client.get_httpx_client().get(
                "/inference-backends/all"
            )
            backends = resp.json()
            if backends:
                with _cache_lock:
                    for backend in backends:
                        backend = InferenceBackend.model_validate(backend)
                        if backend:
                            self.backends_cache[backend.backend_name] = backend
                logger.info(
                    f"Initialized cache with {self.backends_cache.keys()} InferenceBackends"
                )
            else:
                logger.info("No existing InferenceBackends found")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceBackend cache: {e}")
            raise

    async def _watch_changes(self) -> None:
        """Watch for InferenceBackend changes and update the cache."""
        while self._running:
            try:
                logger.info("Starting to watch InferenceBackend changes")
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
            old: Previously cached backend for the same name.
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

            if event.type == EventType.CREATED or event.type == EventType.UPDATED:
                with _cache_lock:
                    old = self.backends_cache.get(backend.backend_name)
                    self._merge_version_configs(old, backend)
                    self.backends_cache[backend.backend_name] = backend
                logger.debug(
                    f"Updated InferenceBackend in cache: {backend.id} ({event.type})"
                )
            elif event.type == EventType.DELETED:
                with _cache_lock:
                    # Remove from both caches
                    self.backends_cache.pop(backend.backend_name, None)
                    logger.debug(
                        f"Removed InferenceBackend from cache: {backend.backend_name}"
                    )
            else:
                logger.warning(f"Unknown event type: {event.type}")

        except Exception as e:
            logger.error(f"Error handling InferenceBackend event: {e}")
