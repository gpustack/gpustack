import math
import random
import secrets
from typing import Any, Callable, Optional, Union
from urllib.parse import urlencode

import aiohttp
from fastapi import APIRouter, Depends, Request, Response, Query
from fastapi.responses import RedirectResponse, StreamingResponse
from enum import Enum
from sqlalchemy.orm import selectinload

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
    ConflictException,
    ServiceUnavailableException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.schemas import Principal
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.config import parse_base_model_to_env_vars
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    cluster_visibility_conditions,
    assert_cluster_visible,
    assert_cluster_writable,
    validate_owner_principal,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.services import create_user_with_principal
from gpustack.server.worker_request import stream_to_worker
from gpustack.schemas.clusters import (
    ClusterListParams,
    ClusterUpdate,
    ClusterCreate,
    ClusterPublic,
    ClustersPublic,
    Cluster,
    ClusterStateEnum,
    ClusterProvider,
    SensitiveRegistrationConfig,
    ClusterRegistrationTokenPublic,
    WorkerPoolCreate,
    WorkerPoolPublic,
    WorkerPool,
    CloudOptions,
)
from gpustack.schemas.principals import PrincipalType, platform_principal_id
from gpustack.schemas.users import system_name_prefix
from gpustack.schemas.api_keys import ApiKey
from gpustack.security import get_secret_hash, API_KEY_PREFIX
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.config.config import (
    get_global_config,
    get_cluster_image_name,
    get_cluster_operator_image_name,
)
from gpustack.utils.grafana import resolve_grafana_base_url
from gpustack_runtime.detector import ManufacturerEnum

CLUSTER_LOAD_OPTIONS = [
    selectinload(Cluster.cluster_workers),
    selectinload(Cluster.cluster_models),
]

router = APIRouter()


def get_server_url(request: Request, cluster_override: Optional[str]) -> str:
    """Construct the server URL based on request headers or fallback to default."""
    if cluster_override:
        return cluster_override.strip("/")
    url = get_global_config().server_external_url
    if not url:
        url = f"{request.url.scheme}://{request.url.hostname}"
        if request.url.port:
            url += f":{request.url.port}"
    return url


def _is_cluster_visible(cluster: Cluster, ctx: TenantContext) -> bool:
    """Python-side mirror of cluster_visibility_conditions for in-memory lists."""
    if bypass_tenant_filter(ctx):
        return True
    if (
        ctx.current_principal_id is not None
        and cluster.owner_principal_id == ctx.current_principal_id
    ):
        return True
    if cluster.id in ctx.accessible_cluster_ids:
        return True
    return False


@router.get("", response_model=ClustersPublic, response_model_exclude_none=True)
async def get_clusters(
    session: SessionDep,
    ctx: TenantContextDep,
    params: ClusterListParams = Depends(),
    name: str = None,
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {'deleted_at': None}
    if name:
        fields = {"name": name}

    if params.watch:
        return StreamingResponse(
            Cluster.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                options=CLUSTER_LOAD_OPTIONS,
                filter_func=lambda c: _is_cluster_visible(c, ctx),
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        # Push visibility filtering into the query — own-Org cluster OR
        # cluster_access grant — instead of fetching the whole table and
        # filtering in Python.
        items = await Cluster.all_by_fields(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            options=CLUSTER_LOAD_OPTIONS,
            extra_conditions=cluster_visibility_conditions(ctx, Cluster),
        )

        if not items:
            return PaginatedList[ClusterPublic](
                items=[],
                pagination=Pagination(
                    page=params.page,
                    perPage=params.perPage,
                    total=0,
                    totalPage=0,
                ),
            )

        if params.page < 1 or params.perPage < 1:
            # Return all items.
            pagination = Pagination(
                page=1,
                perPage=len(items),
                total=len(items),
                totalPage=1,
            )
            return PaginatedList[ClusterPublic](items=items, pagination=pagination)

        # sort in memory
        order_by = params.order_by
        if order_by:
            for field, direction in reversed(order_by):
                items.sort(
                    key=_make_sort_key(field),
                    reverse=direction == "desc",
                )

        # Paginate results.
        start = (params.page - 1) * params.perPage
        end = start + params.perPage
        paginated_items = items[start:end]

        count = len(items)
        total_page = math.ceil(count / params.perPage)
        pagination = Pagination(
            page=params.page,
            perPage=params.perPage,
            total=count,
            totalPage=total_page,
        )

        return PaginatedList[ClusterPublic](
            items=paginated_items, pagination=pagination
        )


def _make_sort_key(field: str) -> Callable[[Any], tuple]:
    """
    Returns a key function for sorting objects by a given field.
    Handles:
      - None values (placed at the end regardless of sort direction),
      - Enum instances (uses .value for comparison),
      - Other types (str, int, float, datetime, etc.) as long as they are comparable.
    """

    def key_func(obj: Any) -> tuple:
        val = getattr(obj, field, None)
        if val is None:
            # (1, None) ensures None is sorted after non-None values
            return (1, None)
        if isinstance(val, Enum):
            # Use the underlying value of the Enum for comparison
            sort_val = val.value
        else:
            sort_val = val
        # (0, sort_val) so non-None values come first
        return (0, sort_val)

    return key_func


@router.get("/{id}", response_model=ClusterPublic, response_model_exclude_none=True)
async def get_cluster(session: SessionDep, ctx: TenantContextDep, id: int):
    cluster = await Cluster.one_by_id(
        session,
        id,
        options=CLUSTER_LOAD_OPTIONS,
    )
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    return cluster


def create_update_check(
    provider: ClusterProvider, input: Union[ClusterCreate, ClusterUpdate]
):
    cfg = get_global_config()
    is_cloud_provider = provider not in [
        ClusterProvider.Kubernetes,
        ClusterProvider.Docker,
    ]
    if (
        is_cloud_provider
        and isinstance(input, ClusterCreate)
        and input.credential_id is None
    ):
        raise InvalidException(
            message=f"credential_id is required for provider {provider}"
        )
    server_url = input.server_url or cfg.server_external_url
    if is_cloud_provider and server_url is None:
        raise InvalidException(
            message=f"server_url is required for provider {provider}"
        )
    if provider == ClusterProvider.Kubernetes:
        # check for volume mounts
        if input.k8s_volume_mounts is None or len(input.k8s_volume_mounts) < 1:
            # at least one volume mount is required, and the default one is for gpustack data dir.
            raise InvalidException(
                message="At least one k8s_volume_mount is required, and the default one is for gpustack data dir."
            )
        if (
            input.k8s_volume_mounts[0].volume_source is None
            or input.k8s_volume_mounts[0].volume_source.host_path is None
        ):
            raise InvalidException(
                message="The first k8s_volume_mount must be for gpustack data dir with hostPath volume source."
            )


def enforce_data_dir_mounts(input: Union[ClusterCreate, ClusterUpdate]):
    """
    Assuming the first item of k8s_volume_mounts is for gpustack data dir, enforce that it is always present and has the correct settings.
    """
    # the first volume must exist as it's validated in create_update_check, and it must be for gpustack data dir, so we enforce it here.
    data_dir_mount = input.k8s_volume_mounts[0]
    data_dir_mount.name = "gpustack-data-dir"
    data_dir_mount.mount_path = "/var/lib/gpustack"
    data_dir_mount.read_only = False
    data_dir_mount.volume_source.host_path.type = "DirectoryOrCreate"
    data_dir_mount.volume_source.config_map = None
    data_dir_mount.volume_source.persistent_volume_claim = None


@router.post("", response_model=ClusterPublic, response_model_exclude_none=True)
async def create_cluster(
    session: SessionDep, ctx: TenantContextDep, input: ClusterCreate
):
    # Every cluster has an owner Org. Fill in a sensible default when the
    # caller omitted it: their current Org context, or the platform Org
    # for admin in "All" mode (admin's home is Default).
    if input.owner_principal_id is None:
        input.owner_principal_id = ctx.current_principal_id or platform_principal_id()
    validate_owner_principal(input.owner_principal_id, ctx, resource_label="cluster")

    # Cluster names are unique within their owning Org, not globally —
    # two Orgs can each have a "c1".
    existing = await Cluster.one_by_fields(
        session,
        {
            'deleted_at': None,
            "name": input.name,
            "owner_principal_id": input.owner_principal_id,
        },
    )
    if existing:
        raise AlreadyExistsException(message=f"cluster {input.name} already exists")

    create_update_check(input.provider, input)
    if input.provider == ClusterProvider.Kubernetes:
        enforce_data_dir_mounts(input)

    # Auto-promote the first cluster in an Org to that Org's default so
    # users don't have to flip a separate switch after onboarding.
    has_existing_in_org = await Cluster.first_by_field(
        session=session, field="owner_principal_id", value=input.owner_principal_id
    )
    auto_default = has_existing_in_org is None

    access_key = secrets.token_hex(8)
    secret_key = secrets.token_hex(16)
    target_state = ClusterStateEnum.READY
    state_message = None
    if input.provider not in [ClusterProvider.Kubernetes, ClusterProvider.Docker]:
        target_state = ClusterStateEnum.PENDING
        state_message = "No workers have been provisioned for this cluster yet."
    pools = input.worker_pools or []
    to_create_cluster = Cluster.model_validate(
        {
            **input.model_dump(exclude={"worker_pools"}),
            "state": target_state,
            "state_message": state_message,
            "hashed_suffix": secrets.token_hex(6),
            "registration_token": f"{API_KEY_PREFIX}_{access_key}_{secret_key}",
            "is_default": auto_default,
        }
    )
    to_create_principal = Principal(
        slug=f'{system_name_prefix}-{to_create_cluster.hashed_suffix}',
        kind=PrincipalType.SYSTEM,
    )
    to_create_apikey = ApiKey(
        name=f'{system_name_prefix}-{to_create_cluster.hashed_suffix}',
        access_key=access_key,
        hashed_secret_key=get_secret_hash(secret_key),
    )
    try:
        # create cluster
        cluster = await Cluster.create(session, to_create_cluster, auto_commit=False)
        # create pools
        for pool in pools:
            to_create_pool = WorkerPool.model_validate(
                {
                    **pool.model_dump(),
                    "cluster_id": cluster.id,
                    # Pool inherits its cluster's owner so list filters
                    # can scope without joining.
                    "owner_principal_id": cluster.owner_principal_id,
                    "cloud_options": (
                        pool.cloud_options if pool.cloud_options else CloudOptions()
                    ),
                }
            )
            to_create_pool.cluster = cluster
            await WorkerPool.create(
                session=session, source=to_create_pool, auto_commit=False
            )
        user = await create_user_with_principal(session, to_create_principal)
        cluster.system_principal_id = user.id
        await cluster.save(session=session, auto_commit=False)
        to_create_apikey.user_id = user.id
        await ApiKey.create(session=session, source=to_create_apikey, auto_commit=False)
        await session.commit()
        await session.refresh(cluster)
        return cluster
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create cluster: {e}")


@router.put("/{id}", response_model=ClusterPublic, response_model_exclude_none=True)
async def update_cluster(
    session: SessionDep, ctx: TenantContextDep, id: int, input: ClusterUpdate
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")
    assert_cluster_writable(ctx, cluster)

    create_update_check(cluster.provider, input)
    if cluster.provider == ClusterProvider.Kubernetes:
        enforce_data_dir_mounts(input)

    try:
        await cluster.update(session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update cluster: {e}")

    return await Cluster.one_by_id(
        session,
        id,
        options=CLUSTER_LOAD_OPTIONS,
    )


@router.delete("/{id}")
async def delete_cluster(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await Cluster.one_by_id(
        session,
        id,
        options=[
            selectinload(Cluster.cluster_workers),
            selectinload(Cluster.cluster_models),
            selectinload(Cluster.cluster_model_instances),
        ],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    assert_cluster_writable(ctx, existing)
    # check for workers, if any are present, prevent deletion
    if len(existing.cluster_workers) > 0:
        raise ConflictException(
            message=f"cluster {existing.name}(id: {id}) has workers, cannot be deleted"
        )
    # check for models, if any are present, prevent deletion
    if len(existing.cluster_models) > 0:
        raise ConflictException(
            message=f"cluster {existing.name}(id: {id}) has models, cannot be deleted"
        )
    # check for model instances, if any are present, prevent deletion
    if len(existing.cluster_model_instances) > 0:
        raise ConflictException(
            message=f"cluster {existing.name}(id: {id}) has model instances, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete cluster: {e}")


@router.post("/{id}/set-default")
async def set_default_cluster(session: SessionDep, ctx: TenantContextDep, id: int):
    # "Default cluster" is a per-Org concept now: each Org has at most
    # one default, and that's what its members' deploy form falls back
    # to. Writing it follows the standard cluster-write rule (admin
    # always; Org owner only on their own Org's clusters).
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")
    assert_cluster_writable(ctx, cluster)

    try:
        # Unset any existing default in this cluster's Org. The partial
        # unique index guarantees there's at most one to begin with.
        existing_defaults = await Cluster.all_by_fields(
            session,
            {
                'is_default': True,
                'deleted_at': None,
                'owner_principal_id': cluster.owner_principal_id,
            },
        )
        for dc in existing_defaults:
            if dc.id != cluster.id:
                await dc.update(
                    session=session,
                    source={"is_default": False},
                    auto_commit=False,
                )
        # Set this cluster as the Org's default.
        await cluster.update(
            session=session,
            source={"is_default": True},
            auto_commit=False,
        )
        await session.commit()
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to set default cluster: {e}"
        )


@router.post("/{id}/worker-pools", response_model=WorkerPoolPublic)
async def create_worker_pool(
    session: SessionDep, ctx: TenantContextDep, id: int, input: WorkerPoolCreate
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    assert_cluster_writable(ctx, cluster)
    if cluster.provider in [ClusterProvider.Docker, ClusterProvider.Kubernetes]:
        raise InvalidException(
            message=f"Cannot create worker pool for cluster {cluster.name}(id: {id}) with provider {cluster.provider}"
        )
    try:
        cloud_options = input.cloud_options or CloudOptions()
        worker_pool = WorkerPool.model_validate(
            {
                **input.model_dump(),
                "cluster_id": id,
                "owner_principal_id": cluster.owner_principal_id,
                "cloud_options": cloud_options,
            }
        )
        worker_pool.cluster = cluster
        return await WorkerPool.create(session, worker_pool)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create worker pool: {e}")


def get_registration_from_cluster(
    request: Request, cluster: Cluster
) -> ClusterRegistrationTokenPublic:
    config = cluster.worker_config.model_dump() if cluster.worker_config else {}
    sensitive_registration = SensitiveRegistrationConfig(
        token=cluster.registration_token, **config
    )
    return ClusterRegistrationTokenPublic(
        token=cluster.registration_token,
        server_url=get_server_url(request, cluster.server_url),
        image=get_cluster_image_name(
            cluster.worker_config
        ),  # Default image, can be customized
        env=parse_base_model_to_env_vars(sensitive_registration),
        args=[],
        # Below fields are used for configure GPUStack Operator.
        operator_image=get_cluster_operator_image_name(cluster.worker_config),
    )


@router.get("/{id}/registration-token", response_model=ClusterRegistrationTokenPublic)
async def get_registration_token(
    request: Request, session: SessionDep, ctx: TenantContextDep, id: int
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    # Registration token is a write-class secret (anyone holding it can
    # register a worker into this cluster) — gate with the writable check.
    assert_cluster_writable(ctx, cluster)
    return get_registration_from_cluster(request, cluster)


@router.get("/{id}/manifests")
async def get_cluster_manifests(
    request: Request,
    session: SessionDep,
    id: int,
    runtime: Optional[ManufacturerEnum] = Query(
        None, description="Optional runtime to include in the manifest"
    ),
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    if cluster.provider != ClusterProvider.Kubernetes:
        raise InvalidException(
            message=f"Cannot get manifests for cluster {cluster.name}(id: {id}) with provider {cluster.provider}"
        )
    # TODO: Redundant principal slugs at the cluster level to reduce multiple queries.
    principal = await Principal.one_by_id(session, cluster.owner_principal_id)
    if not principal:
        raise NotFoundException(
            message=(
                f"Owner principal (id: {cluster.owner_principal_id}) of cluster "
                f"{cluster.name}(id: {id}) not found"
            )
        )

    config = TemplateConfig(
        registration=get_registration_from_cluster(request, cluster),
        cluster_suffix=cluster.hashed_suffix,
        cluster_owner_principal_slug=principal.slug,
        namespace=getattr(cluster.worker_config, "namespace", None),
        runtime_enum=runtime,
        k8s_volume_mounts=cluster.k8s_volume_mounts,
    )
    yaml_content = config.render()
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=manifest.yaml"},
    )


@router.get("/{id}/dashboard")
async def get_cluster_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
):
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message="cluster not found")

    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_worker_dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )
    query_params = {"var-cluster_name": cluster.name}

    grafana_base = resolve_grafana_base_url(cfg, request)
    slug = "gpustack-worker"
    dashboard_url = f"{grafana_base}/d/{cfg.grafana_worker_dashboard_uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


# Hop-by-hop headers and other things we should not forward to the worker; the
# worker layer will inject its own Authorization, and the worker→k8s leg will
# inject the in-pod ServiceAccount token.
_CLUSTER_PROXY_REQUEST_HEADER_SKIP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    "authorization",
    "cookie",
    "x-api-key",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
}


@router.api_route(
    "/{id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def cluster_apiserver_proxy(
    request: Request,
    session: SessionDep,
    id: int,
    path: str,
):
    """
    Proxy a request to the Kubernetes API server of a Kubernetes-provider
    cluster, by forwarding it through one of the cluster's worker pods. The
    worker uses its in-pod ServiceAccount credentials to call the API server.
    """
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    if cluster.provider != ClusterProvider.Kubernetes:
        raise InvalidException(
            message=(
                f"cluster {cluster.name}(id: {id}) provider is "
                f"{cluster.provider.value}; API server proxy is only supported "
                "for Kubernetes-provider clusters."
            )
        )

    workers = await Worker.all_by_fields(
        session,
        fields={"cluster_id": id, "state": WorkerStateEnum.READY},
    )
    if not workers:
        raise ServiceUnavailableException(
            message=f"No reachable workers in cluster {cluster.name}(id: {id})"
        )
    worker = random.choice(workers)

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _CLUSTER_PROXY_REQUEST_HEADER_SKIP
    }

    body = None
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        body = await request.body()

    # request.query_params preserves order but a flat dict is sufficient for
    # the Kubernetes API surface we forward (no duplicate keys in practice).
    params = dict(request.query_params) or None

    # No total timeout — Kubernetes watch and log-follow streams may be open
    # indefinitely. Connect timeout still bounds the upstream connect step.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)

    return StreamingResponseWithStatusCode(
        stream_to_worker(
            worker=worker,
            method=request.method,
            path=f"cluster-proxy/{path}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params=params,
            data=body,
            headers=headers,
            timeout=timeout,
            raw=True,
        ),
    )
