"""Reclassify existing Diffusers video deployments from artifact metadata."""

import asyncio
import logging

from cachetools import TTLCache

from gpustack.scheduler.calculator import get_diffusers_model_index_with_workers
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    Model,
    get_backend,
    is_gguf_model,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.services import ModelService
from gpustack.utils.diffusion import diffusion_model_category

logger = logging.getLogger(__name__)


class VideoCategoryReconciler:
    """Retry unavailable artifacts without blocking model reconciliation."""

    def __init__(self) -> None:
        self._checked = TTLCache(maxsize=1024, ttl=3600)

    async def start(self) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception:
                logger.exception("Failed to reconcile video model categories")
            await asyncio.sleep(180)

    async def reconcile(self) -> None:
        async with async_session() as session:
            models = await Model.all_by_fields(session, fields={"deleted_at": None})
            models = [
                model
                for model in models
                if model.categories == [CategoryEnum.IMAGE]
                and get_backend(model) in (BackendEnum.VLLM, BackendEnum.SGLANG)
                and not is_gguf_model(model)
            ]
            if not models:
                return
            workers = await Worker.all(session)

        for model in models:
            key = (model.id, model.model_source_index, model.cluster_id)
            if key in self._checked:
                continue
            try:
                index = await get_diffusers_model_index_with_workers(model, workers)
                if index is None:
                    continue
                if diffusion_model_category(index) == CategoryEnum.VIDEO:
                    async with async_session() as session:
                        current = await Model.one_by_id(session, model.id)
                        if (
                            current is not None
                            and current.deleted_at is None
                            and current.model_source_index == model.model_source_index
                            and current.cluster_id == model.cluster_id
                            and current.categories == [CategoryEnum.IMAGE]
                            and get_backend(current)
                            in (BackendEnum.VLLM, BackendEnum.SGLANG)
                        ):
                            # ModelService publishes the event that updates only
                            # the routes created by this deployment. Categories
                            # do not change its desired instance generation.
                            await ModelService(session).update(
                                current, source={"categories": [CategoryEnum.VIDEO]}
                            )
                self._checked[key] = True
            except Exception:
                logger.warning(
                    "Could not inspect video category for model %s",
                    model.id,
                    exc_info=True,
                )
