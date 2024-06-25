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
from gpustack.server.auth import get_admin_user, get_current_user


api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["probes"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# authed routes

base_router = APIRouter(dependencies=[Depends(get_current_user)])
base_router.include_router(users.me_router, prefix="/users", tags=["users"])
base_router.include_router(api_keys.router, prefix="/api-keys", tags=["api keys"])

admin_router = APIRouter()
admin_router.include_router(users.router, prefix="/users", tags=["users"])
admin_router.include_router(models.router, prefix="/models", tags=["models"])
admin_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
admin_router.include_router(
    model_instances.router, prefix="/model-instances", tags=["model instances"]
)

api_router.include_router(
    base_router, dependencies=[Depends(get_current_user)], prefix="/v1"
)
api_router.include_router(
    admin_router, dependencies=[Depends(get_admin_user)], prefix="/v1"
)
api_router.include_router(
    openai.router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1-openai",
    tags=["openai"],
)
