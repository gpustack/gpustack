from fastapi import APIRouter, Depends

from gpustack.routes import (
    api_keys,
    auth,
    dashboard,
    debug,
    gpu_devices,
    model_instances,
    model_sets,
    probes,
    proxy,
    update,
    users,
    models,
    openai,
    voice,
    workers,
)

from gpustack.api.exceptions import error_responses, openai_api_error_responses
from gpustack.routes import rerank
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
    model_sets.router, prefix="/model-sets", tags=["Model Sets"]
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
    update.router,
    dependencies=[Depends(get_admin_user)],
    prefix="/update",
    include_in_schema=False,
)
api_router.include_router(
    openai.router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1-openai",
    responses=openai_api_error_responses,
    tags=["OpenAI-Compatible APIs"],
)
api_router.include_router(
    openai.aliasable_router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1",
    responses=openai_api_error_responses,
    tags=["OpenAI-Compatible APIs using the /v1 alias"],
)
api_router.include_router(
    rerank.router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1",
    tags=["Rerank"],
)
api_router.include_router(
    voice.router,
    dependencies=[Depends(get_current_user)],
    prefix="/v1",
    tags=["Voice"],
)
api_router.include_router(
    proxy.router,
    dependencies=[Depends(get_current_user)],
    prefix="/proxy",
    tags=["Server-Side Proxy"],
    include_in_schema=False,
)
