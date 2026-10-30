from typing import Optional, Union

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.schemas.models import (
    ModelInstancePublic,
    ModelInstanceRankUpdate,
    ModelInstanceStopUpdate,
    ModelInstanceUpdate,
)

from .generated_model_instance_client import (
    ModelInstanceClient as GeneratedModelInstanceClient,
)


class ModelInstanceClient(GeneratedModelInstanceClient):
    """Instance routing and Worker reports over the generated CRUD client."""

    def get_cached(self, id: int) -> Optional[ModelInstancePublic]:
        """Resolve routing from the watch snapshot without control-plane HTTP.

        A disconnected watch cannot establish current routing ownership. Fail
        closed until it reconnects instead of occupying inference threads with
        synchronous API calls or trusting a stale lifecycle snapshot.
        """
        with self._cache_lock:
            if not self._enable_cache or not self._watch_started:
                raise ServiceUnavailableException(
                    message="Model instance routing cache is not synchronized"
                )
            return self._cache.get(id)

    def update(
        self,
        id: int,
        model_update: Union[
            ModelInstanceUpdate, ModelInstanceRankUpdate, ModelInstanceStopUpdate
        ],
    ) -> ModelInstancePublic:
        """Send an instance update or a fenced Worker report unchanged."""
        return super().update(id, model_update)
