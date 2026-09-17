import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from gpustack.config.config import get_global_config
from gpustack.utils.certificates import read_server_ca_bundle

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/cacerts", response_class=Response)
def get_ca_certificates() -> Response:
    """Return the public CA bundle used to bootstrap worker TLS trust."""
    config = get_global_config()
    if config is None:
        raise HTTPException(status_code=404)

    try:
        bundle = read_server_ca_bundle(config.ssl_ca_certfile, config.ssl_certfile)
    except (OSError, ValueError) as error:
        logger.error("Failed to read the server CA bundle: %s", error)
        raise HTTPException(status_code=500) from error

    if bundle is None:
        raise HTTPException(status_code=404)

    return Response(
        content=bundle,
        media_type="application/x-pem-file",
        headers={"Cache-Control": "no-store"},
    )
