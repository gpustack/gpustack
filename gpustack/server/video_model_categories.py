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
from gpustack.utils.video import get_video_capabilities, update_video_metadata

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
                if model.categories in ([CategoryEnum.IMAGE], [CategoryEnum.VIDEO])
                and get_backend(model) in (BackendEnum.VLLM, BackendEnum.SGLANG)
                and not is_gguf_model(model)
            ]
            if not models:
                return
            workers = await Worker.all(session)

        for model in models:
            key = (
                model.id,
                model.model_source_index,
                model.cluster_id,
                model.backend,
                model.backend_version,
                tuple(model.backend_parameters or []),
            )
            if key in self._checked:
                continue
            try:
                index = None
                if get_video_capabilities(model) is None:
                    index = await get_diffusers_model_index_with_workers(model, workers)
                capabilities = get_video_capabilities(model, index)
                if (
                    capabilities is not None
                    or index is not None
                    and diffusion_model_category(index) == CategoryEnum.VIDEO
                ):
                    async with async_session() as session:
                        current = await Model.one_by_id(session, model.id)
                        if (
                            current is not None
                            and current.deleted_at is None
                            and current.model_source_index == model.model_source_index
                            and current.cluster_id == model.cluster_id
                            and current.categories
                            in ([CategoryEnum.IMAGE], [CategoryEnum.VIDEO])
                            and current.backend == model.backend
                            and current.backend_version == model.backend_version
                            and current.backend_parameters == model.backend_parameters
                            and get_backend(current)
                            in (BackendEnum.VLLM, BackendEnum.SGLANG)
                        ):
                            # ModelService publishes the event that updates only
                            # the routes created by this deployment. Categories
                            # do not change its desired instance generation.
                            source = {}
                            if current.categories != [CategoryEnum.VIDEO]:
                                source["categories"] = [CategoryEnum.VIDEO]
                            if update_video_metadata(current, index):
                                source["meta"] = current.meta
                            if source:
                                await ModelService(session).update(
                                    current, source=source
                                )
                self._checked[key] = True
            except Exception:
                logger.warning(
                    "Could not inspect video category for model %s",
                    model.id,
                    exc_info=True,
                )
