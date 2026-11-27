"""Refresh capabilities of video deployments without changing explicit categories."""

import asyncio
import logging
import time

from cachetools import LRUCache, TTLCache

from gpustack.routes.model_common import build_category_conditions
from gpustack.scheduler.calculator import get_diffusers_model_index_with_workers
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    Model,
    get_backend,
    is_gguf_model,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session
from gpustack.server.services import ModelService
from gpustack.utils.diffusion import diffusion_model_category
from gpustack.utils.video import get_video_capabilities, update_video_metadata

logger = logging.getLogger(__name__)


class VideoCategoryReconciler:
    """Retry unavailable artifacts without blocking model reconciliation."""

    def __init__(self) -> None:
        self._checked = TTLCache(maxsize=1024, ttl=3600)
        self._failures = LRUCache(maxsize=1024)

    async def start(self) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception:
                logger.exception("Failed to reconcile video model categories")
            await asyncio.sleep(180)

    async def reconcile(self) -> None:
        async with async_session() as session:
            models = await Model.all_by_fields(
                session,
                fields={"deleted_at": None},
                extra_conditions=build_category_conditions(
                    session, Model, [CategoryEnum.VIDEO.value]
                ),
            )
            models = [
                model
                for model in models
                if model.categories == [CategoryEnum.VIDEO]
                and get_backend(model) in (BackendEnum.VLLM, BackendEnum.SGLANG)
                and not is_gguf_model(model)
            ]
            if not models:
                return
            workers = await Worker.all_by_fields(
                session,
                fields={"deleted_at": None},
                extra_conditions=[Worker.state == WorkerStateEnum.READY],
            )

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
            attempts, retry_at = self._failures.get(key, (0, 0))
            if time.monotonic() < retry_at:
                continue
            try:
                await self._reconcile_model(model, workers)
                self._checked[key] = True
                self._failures.pop(key, None)
            except Exception:
                delay = min(180 * 2 ** min(attempts, 5), 3600)
                self._failures[key] = (attempts + 1, time.monotonic() + delay)
                logger.warning(
                    "Could not inspect video category for model %s; retry in %s seconds",
                    model.id,
                    delay,
                    exc_info=True,
                )

    async def _reconcile_model(self, model: Model, workers) -> None:
        index = None
        if get_video_capabilities(model) is None:
            index = await get_diffusers_model_index_with_workers(model, workers)
        capabilities = get_video_capabilities(model, index)
        if not (
            capabilities is not None
            or index is not None
            and diffusion_model_category(index) == CategoryEnum.VIDEO
        ):
            return
        async with async_session() as session:
            current = await Model.one_by_id(session, model.id)
            if (
                current is not None
                and current.deleted_at is None
                and current.model_source_index == model.model_source_index
                and current.cluster_id == model.cluster_id
                and current.categories == [CategoryEnum.VIDEO]
                and current.backend == model.backend
                and current.backend_version == model.backend_version
                and current.backend_parameters == model.backend_parameters
                and get_backend(current) in (BackendEnum.VLLM, BackendEnum.SGLANG)
            ):
                source = {}
                if update_video_metadata(current, index):
                    source["meta"] = current.meta
                if source:
                    await ModelService(session).update(current, source=source)
