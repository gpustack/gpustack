from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, status

from gpustack.config.config import get_global_config
from gpustack.routes.proxy import proxy_to
from gpustack.utils.grafana import normalize_grafana_url

router = APIRouter()


def grafana_target_base(grafana_url: str) -> str:
    grafana_url = normalize_grafana_url(grafana_url)
    if not grafana_url:
        return ""
    parsed = urlparse(grafana_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


@router.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
)
async def grafana_proxy(
    path: str,
    request: Request,
):
    """
    Grafana proxy endpoint.
    Only admin users can access.
    """
    cfg = get_global_config()
    target_base = grafana_target_base(cfg.get_grafana_url() or "")
    if not target_base:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grafana settings are not configured",
        )

    target_path = path.lstrip("/")
    target_url = f"{target_base}/{target_path}" if target_path else target_base
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    return await proxy_to(request, target_url)
