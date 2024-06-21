from fastapi import APIRouter, Depends

from gpustack.routes import (
    api_keys,
    auth,
    model_instances,
    probes,
    users,
    models,
    nodes,
    openai,
)
from gpustack.api.exceptions import error_responses
from gpustack.server.auth import get_current_user

resource_router = APIRouter()
resource_router.include_router(users.router, prefix="/users", tags=["users"])
resource_router.include_router(models.router, prefix="/models", tags=["models"])
resource_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
resource_router.include_router(
    model_instances.router, prefix="/model_instances", tags=["model instances"]
)
resource_router.include_router(api_keys.router, prefix="/api_keys", tags=["api keys"])

authed_api_router = APIRouter(dependencies=[Depends(get_current_user)])
authed_api_router.include_router(resource_router, prefix="/v1")
authed_api_router.include_router(openai.router, tags=["openai"])

api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["probes"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(authed_api_router)
