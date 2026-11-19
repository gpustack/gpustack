"""Platform-only Benchmark archive capacity administration."""

import logging

from fastapi import APIRouter

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.artifacts.capacity import CapacityManager
from gpustack.schemas.artifact_storage import (
    ArtifactCapacityUpdate,
    ArtifactCapacityUsage,
)
from gpustack.server.deps import PlatformAdminDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/capacity', response_model=ArtifactCapacityUsage)
async def artifact_storage_capacity(ctx: PlatformAdminDep):
    try:
        return await CapacityManager().usage()
    except Exception as exc:
        # Database exception messages may contain credentials.
        logger.error('Artifact storage capacity read failed (%s)', type(exc).__name__)
        raise ServiceUnavailableException(
            message='Artifact storage capacity is unavailable',
            details={'code': 'capacity_unavailable'},
        ) from None


@router.put('/capacity', response_model=ArtifactCapacityUsage)
async def update_artifact_storage_capacity(
    body: ArtifactCapacityUpdate, ctx: PlatformAdminDep
):
    try:
        return await CapacityManager().set_limit(body.max_bytes)
    except Exception as exc:
        logger.error('Artifact storage capacity update failed (%s)', type(exc).__name__)
        raise ServiceUnavailableException(
            message='Artifact storage capacity is unavailable',
            details={'code': 'capacity_unavailable'},
        ) from None
