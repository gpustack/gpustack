import secrets
import datetime
import base64
from typing import Optional
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    ForbiddenException,
)
from gpustack.config.config import get_global_config
from gpustack.server.deps import (
    SessionDep,
    EngineDep,
    CurrentUserDep,
)
from gpustack.schemas.workers import (
    WorkerCreate,
    WorkerListParams,
    WorkerPublic,
    WorkerUpdate,
    WorkersPublic,
    Worker,
    WorkerRegistrationPublic,
    WorkerStatusPublic,
    WorkerStateEnum,
)
from gpustack.schemas.clusters import Cluster, Credential, ClusterStateEnum
from gpustack.schemas.users import User, UserRole
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.config import (
    SensitivePredefinedConfig,
    PredefinedConfigNoDefaults,
)
from gpustack.security import get_secret_hash, API_KEY_PREFIX
from gpustack.server.services import WorkerService
from gpustack.cloud_providers.common import key_bytes_to_openssh_pem

router = APIRouter()
system_name_prefix = "system/worker"


def to_worker_public(input: Worker, me: bool) -> WorkerPublic:
    data = input.model_dump()
    if me:
        data['me'] = me
    return WorkerPublic.model_validate(data)


@router.get("", response_model=WorkersPublic)
async def get_workers(
    user: CurrentUserDep,
    engine: EngineDep,
    session: SessionDep,
    params: WorkerListParams = Depends(),
    name: str = None,
    search: str = None,
    uuid: str = None,
    me: Optional[bool] = None,
    cluster_id: Optional[int] = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if name:
        fields = {"name": name}
    if uuid:
        fields["worker_uuid"] = uuid
    if cluster_id:
        fields["cluster_id"] = cluster_id

    if params.watch:
        return StreamingResponse(
            Worker.streaming(engine, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )
    if me and user.worker is not None:
        # me query overrides all other filters
        fields = {"id": user.worker.id}
        fuzzy_fields = {}

    order_by = params.order_by
    if order_by:
        new_order_by = []
        for field, direction in order_by:
            # maps gpus (gpu count) to the internal representation for JSON array length
            if field == "gpus":
                new_order_by.append(("status.gpu_devices[]", direction))
            else:
                new_order_by.append((field, direction))
        order_by = new_order_by

    worker_list = await Worker.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=order_by,
    )
    if not user.worker:
        return worker_list
    public_list = []
    for worker in worker_list.items:
        public_list.append(to_worker_public(worker, user.worker.id == worker.id))
    return WorkersPublic(items=public_list, pagination=worker_list.pagination)


@router.get("/{id}", response_model=WorkerPublic)
async def get_worker(
    user: CurrentUserDep,
    session: SessionDep,
    id: int,
):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")
    if user.worker is not None and user.worker.id == worker.id:
        return to_worker_public(worker, True)
    return worker


def update_worker_data(
    worker_in: WorkerCreate,
    existing: Optional[Worker] = None,
    **kwargs,
) -> Worker:
    to_create_worker = None
    if existing is not None:
        to_create_worker = Worker.model_validate(
            {
                **existing.model_dump(),
                **worker_in.model_dump(),
                "labels": {
                    **existing.labels,
                    **worker_in.labels,
                },
                "cluster_id": existing.cluster_id,
                "state": WorkerStateEnum.READY,
            }
        )
    else:
        to_create_worker = Worker.model_validate(
            {
                **worker_in.model_dump(),
                "state": WorkerStateEnum.READY,
                **kwargs,
            }
        )
        if kwargs.get("cluster") is not None:
            to_create_worker.cluster = kwargs.get("cluster")
    to_create_worker.compute_state()
    return to_create_worker


async def get_existing_worker(
    session, cluster_id: int, worker_in: WorkerCreate
) -> Optional[Worker]:
    static_fields = {
        "deleted_at": None,
        "cluster_id": cluster_id,
    }
    # find existing worker by external_id or worker_uuid
    for field in ["external_id", "worker_uuid"]:
        value = getattr(worker_in, field, None)
        if value is None:
            continue
        fields = {**static_fields, field: value}
        existing_worker = await Worker.one_by_fields(session, fields)
        if existing_worker is not None:
            return existing_worker

    # find existing worker by name
    if worker_in.labels and worker_in.labels.get("gpustack.existence-check"):
        fields = {"name": worker_in.name}
        existing_worker = await Worker.one_by_fields(session, fields)
        if existing_worker is not None:
            return existing_worker

    # no existing worker found, find duplicated name worker
    name_conflict_fields = {**static_fields, "name": worker_in.name}
    name_conflict_worker = await Worker.one_by_fields(session, name_conflict_fields)
    if name_conflict_worker is not None:
        raise AlreadyExistsException(
            message=f"worker with name {worker_in.name} already exists"
        )
    return None


@router.post("", response_model=WorkerRegistrationPublic)
async def create_worker(
    user: CurrentUserDep, session: SessionDep, worker_in: WorkerCreate
):
    cluster_id = (
        worker_in.cluster_id if worker_in.cluster_id is not None else user.cluster_id
    )
    if cluster_id is None:
        raise ForbiddenException(message="Missing cluster_id for worker registration")

    existing_worker = await get_existing_worker(session, cluster_id, worker_in)
    if existing_worker is None:
        if worker_in.external_id is not None:
            # avoid creating a worker with a non-existent external_id
            raise NotFoundException(
                message=f"worker with external_id {worker_in.external_id} not found"
            )

    # needed a session bond cluster object here
    cluster = await Cluster.one_by_id(session, cluster_id)
    if cluster is None or cluster.deleted_at is not None:
        raise NotFoundException(message="Cluster not found")

    sensitive_fields = set(SensitivePredefinedConfig.model_fields.keys())

    worker_config = (
        {}
        if cluster.worker_config is None
        else cluster.worker_config.model_dump(exclude=sensitive_fields)
    )
    cfg = get_global_config()
    if (
        cfg.system_default_container_registry is not None
        and len(cfg.system_default_container_registry) > 0
    ):
        worker_config.setdefault(
            "system_default_container_registry", cfg.system_default_container_registry
        )

    hashed_suffix = secrets.token_hex(6)
    access_key = secrets.token_hex(8)
    secret_key = secrets.token_hex(16)
    new_token = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

    new_worker = update_worker_data(
        worker_in,
        existing=existing_worker,
        # following args are only used when creating a new worker
        provider=cluster.provider,
        cluster=cluster,
        token=new_token,
    )

    # determine if existing worker already has an user and api key
    existing_user = (
        await User.one_by_field(
            session=session, field="worker_id", value=existing_worker.id
        )
        if existing_worker
        else None
    )

    to_create_user = (
        User(
            username=f'{system_name_prefix}-{hashed_suffix}',
            is_system=True,
            role=UserRole.Worker,
            hashed_password="",
            cluster=cluster,
        )
        if not existing_user
        else None
    )

    existing_api_key = (
        existing_user.api_keys[0]
        if existing_user and existing_user.api_keys and len(existing_user.api_keys) > 0
        else None
    )

    to_create_apikey = (
        ApiKey(
            name=f'{system_name_prefix}-{hashed_suffix}',
            access_key=access_key,
            hashed_secret_key=get_secret_hash(secret_key),
        )
        if not existing_api_key
        else None
    )

    try:
        worker = None
        if existing_worker is not None:
            if to_create_apikey is not None:
                new_worker.token = new_token
            await WorkerService(session).update(
                existing_worker, new_worker, auto_commit=False
            )
            worker = existing_worker
        else:
            worker = await Worker.create(session, new_worker, auto_commit=False)
        created_user = None
        if to_create_user is not None:
            to_create_user.worker = worker
            created_user = await User.create(
                session=session, source=to_create_user, auto_commit=False
            )
        if to_create_apikey is not None:
            to_create_apikey.user = existing_user or created_user
            to_create_apikey.user_id = (existing_user or created_user).id
            await ApiKey.create(
                session=session, source=to_create_apikey, auto_commit=False
            )
        if cluster.state != ClusterStateEnum.READY:
            cluster.state = ClusterStateEnum.READY
            await cluster.update(session=session, auto_commit=False)
        await session.commit()
        await session.refresh(worker)
        worker_dump = worker.model_dump()
        worker_dump["token"] = worker.token
        worker_dump["worker_config"] = PredefinedConfigNoDefaults.model_validate(
            worker_config
        )

        return WorkerRegistrationPublic.model_validate(worker_dump)
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to create worker: {e}")


@router.put("/{id}", response_model=WorkerPublic)
async def update_worker(session: SessionDep, id: int, worker_in: WorkerUpdate):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")

    patch = worker_in.model_dump()
    if worker_in.maintenance is not None:
        worker.maintenance = worker_in.maintenance
        worker.compute_state()
        patch["state"] = worker.state
    try:
        await WorkerService(session).update(worker, patch)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update worker: {e}")

    return worker


@router.delete("/{id}")
async def delete_worker(session: SessionDep, id: int):
    worker = await Worker.one_by_id(session, id)
    if not worker or worker.deleted_at is not None:
        raise NotFoundException(message="worker not found")
    try:
        soft = worker.external_id is not None
        if soft:
            worker.state = WorkerStateEnum.DELETING
        await WorkerService(session).delete(worker, soft=soft)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete worker: {e}")


async def create_worker_status(
    user: CurrentUserDep, session: SessionDep, input: WorkerStatusPublic
):
    if user.worker is None:
        raise ForbiddenException(message="Failed to find related worker")
    # query a session bound worker
    worker: Worker = await Worker.one_by_id(session, user.worker.id)
    if not worker or worker.deleted_at is not None:
        raise NotFoundException(message="Worker not found")
    cluster: Cluster = await Cluster.one_by_id(session, worker.cluster_id)
    heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    input_dict = input.model_dump(exclude_unset=True)
    input_dict["heartbeat_time"] = heartbeat_time
    worker_dict = worker.model_dump()
    worker_dict.update(input_dict)
    try:
        to_update = worker.model_validate(worker_dict)
        to_update.compute_state()
        await WorkerService(session).update(worker, to_update)
        if input.gateway_endpoint is not None:
            # no need to use transaction here
            cluster.reported_gateway_endpoint = input.gateway_endpoint
            await cluster.update(session=session)
        return Response(status_code=204)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update worker: {e}")


async def heartbeat(user: CurrentUserDep, session: SessionDep):
    if user.worker is None:
        raise ForbiddenException(message="Failed to find related worker")
    # query a session bound worker
    worker: Worker = await Worker.one_by_id(session, user.worker.id)
    if not worker or worker.deleted_at is not None:
        raise NotFoundException(message="Worker not found")

    worker.heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    )
    try:
        await worker.update(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update worker: {e}")

    return Response(status_code=204)


@router.get("/{id}/privatekey")
async def get_worker_privatekey(
    session: SessionDep,
    id: int,
):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")
    if worker.ssh_key_id is None:
        raise NotFoundException(message="worker ssh key not found")
    ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
    if not ssh_key:
        raise NotFoundException(message="worker ssh key not found")
    private_key_bytes = base64.b64decode(ssh_key.encoded_private_key)
    private_key_pem = key_bytes_to_openssh_pem(
        private_key_bytes, ssh_key.ssh_key_options.algorithm
    )

    return Response(
        content=private_key_pem,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=worker-{id}-private_key.pem"
        },
    )
