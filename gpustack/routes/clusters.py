import math
import secrets
from typing import Any, Callable, Optional, Union
from urllib.parse import urlencode
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
)
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.config import parse_base_model_to_env_vars
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep
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
from gpustack.schemas.users import User, UserRole, system_name_prefix
from gpustack.schemas.api_keys import ApiKey
from gpustack.security import get_secret_hash, API_KEY_PREFIX
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.config.config import get_global_config, get_cluster_image_name
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


@router.get("", response_model=ClustersPublic, response_model_exclude_none=True)
async def get_clusters(
    session: SessionDep,
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
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        items = await Cluster.all_by_fields(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            options=CLUSTER_LOAD_OPTIONS,
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
async def get_cluster(session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(
        session,
        id,
        options=CLUSTER_LOAD_OPTIONS,
    )
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")
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


@router.post("", response_model=ClusterPublic, response_model_exclude_none=True)
async def create_cluster(session: SessionDep, input: ClusterCreate):
    existing = await Cluster.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(message=f"cluster {input.name} already exists")

    create_update_check(input.provider, input)

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
        }
    )
    to_create_user = User(
        username=f'{system_name_prefix}-{to_create_cluster.hashed_suffix}',
        is_system=True,
        role=UserRole.Cluster,
        hashed_password="",
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
                    "cloud_options": (
                        pool.cloud_options if pool.cloud_options else CloudOptions()
                    ),
                }
            )
            to_create_pool.cluster = cluster
            await WorkerPool.create(
                session=session, source=to_create_pool, auto_commit=False
            )
        to_create_user.cluster = cluster
        user = await User.create(
            session=session, source=to_create_user, auto_commit=False
        )
        to_create_apikey.user_id = user.id
        to_create_apikey.user = user
        await ApiKey.create(session=session, source=to_create_apikey, auto_commit=False)
        await session.commit()
        await session.refresh(cluster)
        return cluster
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create cluster: {e}")


@router.put("/{id}", response_model=ClusterPublic, response_model_exclude_none=True)
async def update_cluster(session: SessionDep, id: int, input: ClusterUpdate):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")

    create_update_check(cluster.provider, input)

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
async def delete_cluster(session: SessionDep, id: int):
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
async def set_default_cluster(session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")

    try:
        # unset other default clusters
        default_clusters = await Cluster.all_by_fields(
            session,
            {'is_default': True, 'deleted_at': None},
        )
        for dc in default_clusters:
            if dc.id != cluster.id:
                await dc.update(
                    session=session,
                    source={"is_default": False},
                    auto_commit=False,
                )
        # set this cluster as default
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
async def create_worker_pool(session: SessionDep, id: int, input: WorkerPoolCreate):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
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
    )


@router.get("/{id}/registration-token", response_model=ClusterRegistrationTokenPublic)
async def get_registration_token(request: Request, session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
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
    config = TemplateConfig(
        registration=get_registration_from_cluster(request, cluster),
        cluster_suffix=cluster.hashed_suffix,
        namespace=getattr(cluster.worker_config, "namespace", None),
        runtime_enum=runtime,
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
    id: int,
    request: Request,
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message="cluster not found")

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
