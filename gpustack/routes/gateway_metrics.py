import logging

from fastapi import APIRouter, Depends

from gpustack.api.auth import gateway_token_auth
from gpustack.server.metrics_collector import (
    ModelUsageMetrics,
    accumulate_gateway_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/gateway-metrics", include_in_schema=False)
async def report_gateway_metrics(
    item: ModelUsageMetrics,
    _: None = Depends(gateway_token_auth),
):
    await accumulate_gateway_metrics([item])
