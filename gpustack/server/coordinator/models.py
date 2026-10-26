"""
Model registry for cross-instance event enrichment.

This module provides lazy loading of model classes to avoid circular imports.
It's separate from bus.py to prevent the import cycle:
bus.py -> models -> active_record.py -> bus.py
"""

import logging
from typing import Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class _ModelRegistry:
    """
    Registry for topic-to-model mappings with lazy loading.

    This avoids circular imports by only importing model classes when first accessed.
    """

    # Topics added through :func:`register_topic`, tracked separately from
    # the built-in entries so startup wiring can enumerate just the
    # extension ones.
    _EXTERNAL_TOPICS: list = []

    _REGISTRY: Dict[str, Callable[[], Optional[Type]]] = {
        'worker': lambda: _import_model('gpustack.schemas.workers', 'Worker'),
        'model': lambda: _import_model('gpustack.schemas.models', 'Model'),
        'modelinstance': lambda: _import_model(
            'gpustack.schemas.models', 'ModelInstance'
        ),
        'modelfile': lambda: _import_model('gpustack.schemas.model_files', 'ModelFile'),
        'modelroute': lambda: _import_model(
            'gpustack.schemas.model_routes', 'ModelRoute'
        ),
        'modelroutetarget': lambda: _import_model(
            'gpustack.schemas.model_routes', 'ModelRouteTarget'
        ),
        'cluster': lambda: _import_model('gpustack.schemas.clusters', 'Cluster'),
        'workerpool': lambda: _import_model('gpustack.schemas.clusters', 'WorkerPool'),
        'cloudcredential': lambda: _import_model(
            'gpustack.schemas.clusters', 'CloudCredential'
        ),
        'modelprovider': lambda: _import_model(
            'gpustack.schemas.model_provider', 'ModelProvider'
        ),
        # Identity consolidation made ``User`` an alias of ``Principal``, and
        # ``subscribe()`` keys the topic off ``cls.__name__`` -- so everything
        # publishes to 'principal' and the 'user' entry below has been dead
        # since. Without 'principal' registered, every cross-instance
        # principal event was dropped by _process_coordinator_event. Keep the
        # stale key as an alias rather than deleting it: it costs nothing and
        # covers anything still constructing the topic string by hand.
        'principal': lambda: _import_model('gpustack.schemas.principals', 'Principal'),
        'user': lambda: _import_model('gpustack.schemas.users', 'User'),
        'apikey': lambda: _import_model('gpustack.schemas.api_keys', 'ApiKey'),
        'benchmark': lambda: _import_model('gpustack.schemas.benchmark', 'Benchmark'),
        'inferencebackend': lambda: _import_model(
            'gpustack.schemas.inference_backend', 'InferenceBackend'
        ),
        # Every configurable-content kind: a controller on the leader
        # subscribes to its rows, while the write that changes one lands on
        # whichever server served the request. Without an entry here the
        # event never crosses, so a document configured through a standby
        # reaches the leader only on its next full pass -- or, for a kind
        # whose controller acts on events alone, not at all.
        'catalogsource': lambda: _import_model(
            'gpustack.schemas.catalog_source', 'CatalogSource'
        ),
        'inferencebackendsource': lambda: _import_model(
            'gpustack.schemas.inference_backend_source', 'InferenceBackendSource'
        ),
        'inferencerunnersource': lambda: _import_model(
            'gpustack.schemas.runner_source', 'InferenceRunnerSource'
        ),
        'cacheprovidersource': lambda: _import_model(
            'gpustack.schemas.cache_provider_source', 'CacheProviderSource'
        ),
    }

    @classmethod
    def get_model(cls, topic: str) -> Optional[Type]:
        """Get model class for a topic, or None if not registered."""
        loader = cls._REGISTRY.get(topic)
        return loader() if loader else None

    @classmethod
    def set_entry(cls, topic: str, model_class: Type) -> bool:
        """Add a topic→model entry from an extension package.

        Core's own entries win: an extension re-registering a built-in
        topic is a bug on its side, logged and skipped rather than
        allowed to silently repoint enrichment at another model.

        Returns True when the entry took effect, False when an existing
        registration was kept.
        """
        if topic in cls._REGISTRY:
            logger.warning(
                "Topic '%s' is already registered; ignoring extension "
                "registration for it",
                topic,
            )
            return False
        cls._REGISTRY[topic] = lambda: model_class
        return True


def external_topics() -> list:
    """Topics registered by extension packages via :func:`register_topic`."""
    return list(_ModelRegistry._EXTERNAL_TOPICS)


def _import_model(module_path: str, class_name: str) -> Optional[Type]:
    """Import a model class by module path and class name."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {module_path}.{class_name}: {e}")
        return None


def get_model_for_topic(topic: str) -> Optional[Type]:
    """Get the model class associated with a topic name."""
    return _ModelRegistry.get_model(topic)


def register_topic(topic: str, model_class: Type) -> None:
    """Register an extension package's model for cross-instance delivery.

    An ``Event`` published on an unregistered topic is delivered to the
    publishing instance's own subscribers only — cross-instance delivery
    needs a topic→model entry so the receiving side can enrich the
    id-only payload back into a full row (see ``bus``'s
    ``_process_coordinator_event``). Extension packages that own tables
    and want their writes visible to subscribers on other instances
    (typically a leader-only controller) register each table here,
    through the plugin contract's ``event_topics()``.

    Must run before the event bus starts delivering cross-instance
    events; the server collects plugin registrations during startup.

    Registering a topic that is already registered — core's own or
    another extension's — is logged and ignored, and the topic is not
    tracked as external.
    """
    if _ModelRegistry.set_entry(topic, model_class):
        _ModelRegistry._EXTERNAL_TOPICS.append(topic)
