from fastapi import APIRouter, HTTPException, Request, status

from gpustack.config.config import get_global_config
from gpustack.routes.proxy import proxy_to

router = APIRouter()

PROMETHEUS_BASE_URL = "http://127.0.0.1:9090"


@router.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
)
async def prometheus_proxy(
    path: str,
    request: Request,
):
    """
    Prometheus proxy endpoint.
    Only admin users can access.
    """
    cfg = get_global_config()
    if cfg.disable_builtin_observability:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    target_path = path.lstrip("/")
    if target_path:
        target_url = f"{PROMETHEUS_BASE_URL}/prometheus/{target_path}"
    else:
        target_url = f"{PROMETHEUS_BASE_URL}/prometheus"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    return await proxy_to(request, target_url)
