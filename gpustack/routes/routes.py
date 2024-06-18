from fastapi import APIRouter

from gpustack.routes import (
    model_instances,
    probes,
    users,
    models,
    nodes,
    openai,
)
from gpustack.api.exceptions import error_responses

resource_router = APIRouter()
resource_router.include_router(users.router, prefix="/users", tags=["users"])
resource_router.include_router(models.router, prefix="/models", tags=["models"])
resource_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
resource_router.include_router(
    model_instances.router, prefix="/model_instances", tags=["model instances"]
)


api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["probes"])
api_router.include_router(openai.router, tags=["openai"])
api_router.include_router(resource_router, prefix="/v1")
