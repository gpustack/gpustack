from fastapi import APIRouter, Depends

from gpustack.routes import (
    api_keys,
    auth,
    dashboard,
    debug,
    gpu_devices,
    model_instances,
    probes,
    users,
    models,
    openai,
    workers,
)

from gpustack.api.exceptions import error_responses
from gpustack.server.auth import get_admin_user, get_current_user


api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["Probes"])
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])

# authed routes

v1_base_router = APIRouter(dependencies=[Depends(get_current_user)])
v1_base_router.include_router(users.me_router, prefix="/users", tags=["Users"])
v1_base_router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])

v1_admin_router = APIRouter()
v1_admin_router.include_router(
    dashboard.router, prefix="/dashboard", tags=["Dashboard"]
)
v1_admin_router.include_router(users.router, prefix="/users", tags=["Users"])
v1_admin_router.include_router(models.router, prefix="/models", tags=["Models"])
v1_admin_router.include_router(workers.router, prefix="/workers", tags=["Workers"])
v1_admin_router.include_router(
    model_instances.router, prefix="/model-instances", tags=["Model Instances"]
)
v1_admin_router.include_router(
    gpu_devices.router, prefix="/gpu-devices", tags=["GPU Devices"]
)

api_router.include_router(
    v1_base_router, dependencies=[Depends(get_current_user)], prefix="/v1"
)
api_router.include_router(
    v1_admin_router, dependencies=[Depends(get_admin_user)], prefix="/v1"
)
api_router.include_router(
    debug.router,
    dependencies=[Depends(get_admin_user)],
    prefix="/debug",
    include_in_schema=False,
)
api_router.include_router(
    openai.router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1-openai",
    tags=["OpenAI Compatible APIs"],
)
