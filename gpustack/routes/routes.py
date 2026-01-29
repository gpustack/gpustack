from fastapi import APIRouter, Depends

from gpustack.routes import (
    api_keys,
    auth,
    config,
    dashboard,
    debug,
    draft_models,
    gpu_devices,
    inference_backend,
    metrics,
    model_evaluations,
    model_files,
    model_instances,
    model_sets,
    probes,
    proxy,
    update,
    users,
    models,
    openai,
    workers,
    cloud_credentials,
    worker_pools,
    clusters,
    token,
    benchmarks,
    benchmark_profiles,
    model_provider,
    rerank,
    model_routes,
)

from gpustack.api.exceptions import error_responses, openai_api_error_responses
from gpustack.api.auth import (
    get_admin_user,
    get_current_user,
    get_cluster_user,
    get_worker_user,
)

versioned_prefix = "/v2"

api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["Probes"])
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(
    router=token.router,
    prefix="/token-auth",
    include_in_schema=False,
)

# authed routes

v1_base_router = APIRouter(dependencies=[Depends(get_current_user)])
v1_base_router.include_router(users.me_router, prefix="/users", tags=["Users"])
v1_base_router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
v1_base_router.include_router(
    metrics.router, prefix="/metrics", include_in_schema=False
)
v1_base_router.include_router(
    model_routes.my_models_router,
    dependencies=[Depends(get_current_user)],
    prefix="/my-models",
    tags=["My Models"],
)

cluster_client_router = APIRouter()
cluster_client_router.add_api_route(
    path="/clusters/{id}/manifests",
    endpoint=clusters.get_cluster_manifests,
    methods=["GET"],
)
cluster_client_router.add_api_route(
    path="/workers",
    endpoint=workers.create_worker,
    methods=["POST"],
)

model_routers = [
    {"router": models.router, "prefix": "/models", "tags": ["Models"]},
    {
        "router": model_instances.router,
        "prefix": "/model-instances",
        "tags": ["Model Instances"],
    },
    {"router": model_files.router, "prefix": "/model-files", "tags": ["Model Files"]},
    {"router": benchmarks.router, "prefix": "/benchmarks", "tags": ["Benchmarks"]},
    {
        "router": benchmark_profiles.router,
        "prefix": "/benchmark-profiles",
        "tags": ["Benchmark Profiles"],
    },
    {
        "router": model_routes.target_router,
        "prefix": "/model-route-targets",
        "tags": ["Model Route Targets"],
    },
]
# worker client have full access to model and model instances
worker_client_router = APIRouter()
for model_router in model_routers:
    worker_client_router.include_router(**model_router)
# ready only access to workers
worker_client_router.add_api_route(
    path="/workers",
    endpoint=workers.get_workers,
    methods=["GET"],
    response_model=workers.WorkersPublic,
)
worker_client_router.add_api_route(
    path="/workers/{id}",
    endpoint=workers.get_worker,
    methods=["GET"],
    response_model=workers.WorkerPublic,
)
worker_client_router.add_api_route(
    path="/worker-status",
    endpoint=workers.create_worker_status,
    methods=["POST"],
    include_in_schema=False,
)
worker_client_router.add_api_route(
    path="/worker-heartbeat",
    endpoint=workers.heartbeat,
    methods=["POST"],
    include_in_schema=False,
)
worker_client_router.include_router(
    inference_backend.router, prefix="/inference-backends", tags=["Inference Backend"]
)

admin_routers = model_routers + [
    {"router": dashboard.router, "prefix": "/dashboard", "tags": ["Dashboard"]},
    {"router": workers.router, "prefix": "/workers", "tags": ["Workers"]},
    {"router": users.router, "prefix": "/users", "tags": ["Users"]},
    {"router": model_sets.router, "prefix": "/model-sets", "tags": ["Model Sets"]},
    {
        "router": draft_models.router,
        "prefix": "/draft-models",
        "tags": ["Draft Models"],
    },
    {
        "router": model_evaluations.router,
        "prefix": "/model-evaluations",
        "tags": ["Model Evaluations"],
    },
    {"router": gpu_devices.router, "prefix": "/gpu-devices", "tags": ["GPU Devices"]},
    # following routers are introduced by gpustack v2.0
    {
        "router": cloud_credentials.router,
        "prefix": "/cloud-credentials",
        "tags": ["Cloud Credentials"],
    },
    {
        "router": worker_pools.router,
        "prefix": "/worker-pools",
        "tags": ["Worker Pools"],
    },
    {"router": clusters.router, "prefix": "/clusters", "tags": ["Clusters"]},
    {
        "router": model_provider.router,
        "prefix": "/model-providers",
        "tags": ["Model Providers"],
    },
    {
        "router": model_routes.router,
        "prefix": "/model-routes",
        "tags": ["Model Routes"],
    },
]

v1_admin_router = APIRouter()
for admin_router in admin_routers:
    v1_admin_router.include_router(**admin_router)

api_router.include_router(
    worker_client_router,
    dependencies=[Depends(get_worker_user)],
    prefix=versioned_prefix,
)
api_router.include_router(
    cluster_client_router,
    dependencies=[Depends(get_cluster_user)],
    prefix=versioned_prefix,
)
api_router.include_router(
    v1_base_router, dependencies=[Depends(get_current_user)], prefix=versioned_prefix
)
api_router.include_router(
    v1_admin_router, dependencies=[Depends(get_admin_user)], prefix=versioned_prefix
)
api_router.include_router(
    config.router,
    dependencies=[Depends(get_admin_user)],
    prefix=versioned_prefix,
    include_in_schema=False,
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
    openai.router,
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
    proxy.router,
    dependencies=[Depends(get_current_user)],
    prefix="/proxy",
    tags=["Server-Side Proxy"],
    include_in_schema=False,
)
