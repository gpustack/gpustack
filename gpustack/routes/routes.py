import os

from fastapi import APIRouter, Depends

from gpustack.routes import (
    api_keys,
    auth,
    cluster_access,
    config,
    dashboard,
    debug,
    draft_models,
    gpu_devices,
    inference_backend,
    me_orgs,
    metrics,
    model_evaluations,
    model_files,
    model_instances,
    model_route_principals,
    model_sets,
    organization_members,
    organizations,
    probes,
    proxy,
    update,
    user_groups,
    users,
    models,
    openai,
    workers,
    usage,
    cloud_credentials,
    worker_pools,
    clusters,
    token,
    benchmarks,
    benchmark_profiles,
    model_provider,
    rerank,
    model_routes,
    grafana,
    prometheus,
)

from gpustack.api.exceptions import error_responses, openai_api_error_responses
from gpustack.api.auth import (
    get_admin_user,
    get_current_user,
    get_cluster_user,
    get_worker_user,
    management_scope,
    inference_scope,
)
from gpustack.websocket_proxy.message_server import router as message_server_router
from gpustack.routes.gateway_metrics import router as gateway_metrics_router

from gpustack_higress_plugins.server import router as higress_plugins_router

versioned_prefix = "/v2"


# Toggle for surfacing extended API endpoints in the OpenAPI schema
# and ``/docs``. Endpoints stay mounted regardless — only the public
# docs surface is gated. Off by default; set the env var to a truthy
# value to expose the full surface.
_EXTENDED_API_IN_SCHEMA = os.getenv("GPUSTACK_EXTENDED_API_DOCS", "").lower() in (
    "1",
    "true",
    "yes",
)

api_router = APIRouter(responses=error_responses)
management_router = APIRouter(dependencies=[Depends(management_scope)])
management_router.include_router(
    grafana.router,
    prefix="/grafana",
    dependencies=[Depends(get_admin_user)],
    include_in_schema=False,
)
management_router.include_router(
    prometheus.router,
    prefix="/prometheus",
    dependencies=[Depends(get_admin_user)],
    include_in_schema=False,
)


# authed routes

v1_base_router = APIRouter(dependencies=[Depends(get_current_user)])
v1_base_router.include_router(users.me_router, prefix="/users", tags=["Users"])
v1_base_router.include_router(users.directory_router, tags=["Users"])
v1_base_router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
v1_base_router.include_router(usage.router, prefix="/usage", tags=["Usage"])
v1_base_router.include_router(
    me_orgs.router,
    prefix="/users/me",
    tags=["My Organizations"],
    include_in_schema=_EXTENDED_API_IN_SCHEMA,
)
v1_base_router.include_router(
    organization_members.router,
    tags=["Organization Members"],
    include_in_schema=_EXTENDED_API_IN_SCHEMA,
)
v1_base_router.include_router(
    user_groups.router,
    tags=["User Groups"],
    include_in_schema=_EXTENDED_API_IN_SCHEMA,
)
v1_base_router.include_router(
    metrics.router, prefix="/metrics", include_in_schema=False
)
v1_base_router.include_router(
    model_routes.my_models_router,
    dependencies=[Depends(get_current_user)],
    prefix="/my-models",
    tags=["My Models"],
)
# BYO cluster: clusters / cloud-credentials / worker-pools live on the
# user-level router so Org owner / admin can CRUD their own infra. The
# routes themselves enforce per-row ownership via assert_cluster_writable
# and friends, so platform-only operations (e.g. set-default) still
# require is_admin inside the handler.
v1_base_router.include_router(clusters.router, prefix="/clusters", tags=["Clusters"])
v1_base_router.include_router(
    cloud_credentials.router,
    prefix="/cloud-credentials",
    tags=["Cloud Credentials"],
)
v1_base_router.include_router(
    worker_pools.router, prefix="/worker-pools", tags=["Worker Pools"]
)
# Workers are visible to anyone who can see their cluster; mutations gated
# by an explicit is_admin check inside each handler.
v1_base_router.include_router(workers.router, prefix="/workers", tags=["Workers"])

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

# Tenant-aware routers: any logged-in user can hit them; the handlers
# filter by TenantContext (owner_principal_id / cluster visibility).
tenant_routers = model_routers + [
    {"router": gpu_devices.router, "prefix": "/gpu-devices", "tags": ["GPU Devices"]},
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
    {
        "router": model_route_principals.router,
        "prefix": "/model-routes",
        "tags": ["Model Route Principals"],
        "include_in_schema": _EXTENDED_API_IN_SCHEMA,
    },
    {
        "router": model_evaluations.router,
        "prefix": "/model-evaluations",
        "tags": ["Model Evaluations"],
    },
    # Read-only platform catalogs (no tenant data) — every logged-in user
    # needs them to deploy models, including Org owners/managers.
    {"router": model_sets.router, "prefix": "/model-sets", "tags": ["Model Sets"]},
    {
        "router": draft_models.router,
        "prefix": "/draft-models",
        "tags": ["Draft Models"],
    },
    # Inference backends are platform-wide (admin curates) but every Org
    # owner/manager needs to read them to pick a backend at deploy time.
    # Worker / cluster system users also reach this through v1_base_router
    # since `get_current_user` accepts ``is_system=True`` callers.
    {
        "router": inference_backend.router,
        "prefix": "/inference-backends",
        "tags": ["Inference Backend"],
    },
]

# Platform-only routers — admin can manage globally; non-admin gets 403.
admin_routers = [
    {"router": dashboard.router, "prefix": "/dashboard", "tags": ["Dashboard"]},
    {"router": users.router, "prefix": "/users", "tags": ["Users"]},
    {
        "router": organizations.router,
        "prefix": "/organizations",
        "tags": ["Organizations"],
        "include_in_schema": _EXTENDED_API_IN_SCHEMA,
    },
    {
        "router": cluster_access.router,
        "tags": ["Cluster Access"],
        "include_in_schema": _EXTENDED_API_IN_SCHEMA,
    },
]

for tr in tenant_routers:
    v1_base_router.include_router(**tr)

v1_admin_router = APIRouter()
for admin_router in admin_routers:
    v1_admin_router.include_router(**admin_router)

# Order matters: FastAPI dispatches the FIRST router whose path matches.
# v1_base_router and worker_client_router register overlapping endpoints
# (e.g. /v2/models, /v2/workers) — putting v1_base_router first means
# regular user requests resolve through ``get_current_user`` (which also
# accepts worker / cluster system users), and only routes that are unique
# to the worker / cluster client paths fall through to those routers.
management_router.include_router(
    v1_base_router, dependencies=[Depends(get_current_user)], prefix=versioned_prefix
)
management_router.include_router(
    worker_client_router,
    dependencies=[Depends(get_worker_user)],
    prefix=versioned_prefix,
)
management_router.include_router(
    cluster_client_router,
    dependencies=[Depends(get_cluster_user)],
    prefix=versioned_prefix,
)
management_router.include_router(
    v1_admin_router, dependencies=[Depends(get_admin_user)], prefix=versioned_prefix
)
management_router.include_router(
    config.router,
    dependencies=[Depends(get_admin_user)],
    prefix=versioned_prefix,
    include_in_schema=False,
)
management_router.include_router(
    debug.router,
    dependencies=[Depends(get_admin_user)],
    prefix="/debug",
    include_in_schema=False,
)
management_router.include_router(
    update.router,
    dependencies=[Depends(get_admin_user)],
    prefix="/update",
    include_in_schema=False,
)
management_router.include_router(
    proxy.router,
    dependencies=[Depends(get_current_user)],
    prefix="/proxy",
    tags=["Server-Side Proxy"],
    include_in_schema=False,
)

inference_router = APIRouter(
    dependencies=[Depends(get_current_user), Depends(inference_scope)]
)

inference_router.include_router(
    openai.get_legacy_api_router(),
    prefix="/v1-openai",
    responses=openai_api_error_responses,
    tags=["OpenAI-Compatible APIs (Legacy alias)"],
)
inference_router.include_router(
    openai.get_api_router(),
    prefix="/v1",
    responses=openai_api_error_responses,
    tags=["OpenAI-Compatible APIs"],
)
inference_router.include_router(
    rerank.router,
    prefix="/v1",
    tags=["Rerank"],
)

# Following routes should not check api scope as it is publicly accessible and used for authentication by external services.
api_router.include_router(probes.router, tags=["Probes"])
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(
    router=token.router,
    prefix="/token-auth",
    include_in_schema=False,
)

api_router.include_router(management_router)
api_router.include_router(inference_router)
api_router.include_router(higress_plugins_router, include_in_schema=False)
api_router.include_router(
    gateway_metrics_router,
    prefix=f"{versioned_prefix}/usage",
    include_in_schema=False,
)
api_router.include_router(
    message_server_router,
    tags=["WebSocket Proxy"],
    include_in_schema=True,
)
