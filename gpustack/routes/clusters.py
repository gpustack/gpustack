import secrets
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
    ForbiddenException,
)
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack.schemas.clusters import (
    ClusterUpdate,
    ClusterCreate,
    ClusterPublic,
    ClustersPublic,
    Cluster,
    ClusterStateEnum,
    ClusterProvider,
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
from gpustack.config.config import get_global_config

router = APIRouter()


def get_server_url(request: Request) -> str:
    """Construct the server URL based on request headers or fallback to default."""
    url = get_global_config().server_external_url
    if not url:
        url = f"{request.url.scheme}://{request.url.hostname}"
        if request.url.port:
            url += f":{request.url.port}"
    return url


@router.get("", response_model=ClustersPublic)
async def get_clusters(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
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
            Cluster.streaming(engine, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await Cluster.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=ClusterPublic)
async def get_cluster(session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")
    return cluster


@router.post("", response_model=ClusterPublic)
async def create_cluster(session: SessionDep, input: ClusterCreate):
    existing = await Cluster.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(message=f"cluster {input.name} already exists")
    if (
        input.provider not in [ClusterProvider.Kubernetes, ClusterProvider.Docker]
        and input.credential_id is None
    ):
        raise InvalidException(
            message=f"credential_id is required for provider {input.provider}"
        )
    access_key = secrets.token_hex(8)
    secret_key = secrets.token_hex(16)
    target_state = ClusterStateEnum.PROVISIONING
    if input.provider in [ClusterProvider.Kubernetes, ClusterProvider.Docker]:
        target_state = ClusterStateEnum.READY
    pools = input.worker_pools or []
    to_create_cluster = Cluster.model_validate(
        {
            **input.model_dump(exclude={"worker_pools"}),
            "state": target_state,
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


@router.put("/{id}", response_model=ClusterPublic)
async def update_cluster(session: SessionDep, id: int, input: ClusterUpdate):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")

    try:
        await cluster.update(session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update cluster: {e}")

    return await Cluster.one_by_id(session, id)


@router.delete("/{id}")
async def delete_cluster(session: SessionDep, id: int):
    existing = await Cluster.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    # check for workers, if any are present, prevent deletion
    if len(existing.cluster_workers) > 0:
        raise ForbiddenException(
            message=f"cluster {existing.name}(id: {id}) has workers, cannot be deleted"
        )
    # check for models, if any are present, prevent deletion
    if len(existing.cluster_models) > 0:
        raise ForbiddenException(
            message=f"cluster {existing.name}(id: {id}) has models, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete cluster: {e}")


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


@router.get("/{id}/registration-token", response_model=ClusterRegistrationTokenPublic)
async def get_registration_token(request: Request, session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    url = get_server_url(request)

    return ClusterRegistrationTokenPublic(
        token=cluster.registration_token,
        server_url=url,
        image=get_global_config().get_image_name(),  # Default image, can be customized
    )


@router.get("/{id}/manifests")
async def get_cluster_manifests(request: Request, session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    if cluster.provider != ClusterProvider.Kubernetes:
        raise InvalidException(
            message=f"Cannot get manifests for cluster {cluster.name}(id: {id}) with provider {cluster.provider}"
        )
    url = get_server_url(request)
    config = TemplateConfig(
        cluster_suffix=cluster.hashed_suffix,
        token=cluster.registration_token,
        image=get_global_config().get_image_name(),
        server_url=url,
    )
    yaml_content = config.render()
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=manifest.yaml"},
    )
