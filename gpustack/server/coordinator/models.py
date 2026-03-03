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
        'user': lambda: _import_model('gpustack.schemas.users', 'User'),
        'apikey': lambda: _import_model('gpustack.schemas.api_keys', 'ApiKey'),
        'benchmark': lambda: _import_model('gpustack.schemas.benchmark', 'Benchmark'),
        'inferencebackend': lambda: _import_model(
            'gpustack.schemas.inference_backend', 'InferenceBackend'
        ),
    }

    @classmethod
    def get_model(cls, topic: str) -> Optional[Type]:
        """Get model class for a topic, or None if not registered."""
        loader = cls._REGISTRY.get(topic)
        return loader() if loader else None


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
