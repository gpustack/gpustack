import secrets
import datetime
import base64
from typing import Optional
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    ForbiddenException,
)
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep, CurrentUserDep
from gpustack.schemas.workers import (
    WorkerCreate,
    WorkerPublic,
    WorkerUpdate,
    WorkersPublic,
    Worker,
    WorkerRegistrationPublic,
    WorkerStatusPublic,
    WorkerStateEnum,
)
from gpustack.schemas.clusters import Cluster, Credential, ClusterState
from gpustack.schemas.users import User, UserRole
from gpustack.schemas.api_keys import ApiKey
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
    params: ListParamsDep,
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
    worker_list = await Worker.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
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


def get_create_worker(
    cluster: Cluster, existing: Optional[Worker], worker_in: WorkerCreate
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
                "cluster_id": cluster.id,
                "state": WorkerStateEnum.READY,
            }
        )
    else:
        to_create_worker = Worker.model_validate(
            {
                **worker_in.model_dump(),
                "provider": cluster.provider,
                "state": WorkerStateEnum.READY,
            }
        )
        to_create_worker.cluster = cluster
    to_create_worker.compute_state()
    return to_create_worker


@router.post("", response_model=WorkerRegistrationPublic)
async def create_worker(
    user: CurrentUserDep, session: SessionDep, worker_in: WorkerCreate
):
    if worker_in.cluster_id is None and user.cluster_id is None:
        raise ForbiddenException(message="User does not belong to any cluster")
    fields = {
        "deleted_at": None,
        "cluster_id": user.cluster_id,
    }
    if worker_in.external_id is not None:
        fields["external_id"] = worker_in.external_id
    else:
        fields["name"] = worker_in.name
    existing = await Worker.one_by_fields(session, fields)
    if existing and worker_in.external_id is None:
        # avoid duplicate workers with the same name
        raise AlreadyExistsException(message=f"worker f{worker_in.name} already exists")
    elif existing is None and worker_in.external_id is not None:
        # avoid creating a worker with a non-existent external_id
        raise NotFoundException(
            message=f"worker with external_id {worker_in.external_id} not found"
        )

    cluster_id = (
        worker_in.cluster_id if worker_in.cluster_id is not None else user.cluster_id
    )
    # needed a session bond cluster object here
    cluster = await Cluster.one_by_id(session, cluster_id)
    if cluster is None or cluster.deleted_at is not None:
        raise NotFoundException(message="Cluster not found")

    access_key = secrets.token_hex(8)
    secret_key = secrets.token_hex(16)

    to_create_worker = get_create_worker(cluster, existing, worker_in)
    to_create_worker.token = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

    hashed_suffix = secrets.token_hex(6)
    to_create_user = User(
        username=f'{system_name_prefix}-{hashed_suffix}',
        is_system=True,
        role=UserRole.Worker,
        hashed_password="",
        cluster=cluster,
    )

    to_create_apikey = ApiKey(
        name=f'{system_name_prefix}-{hashed_suffix}',
        access_key=access_key,
        hashed_secret_key=get_secret_hash(secret_key),
    )

    try:
        worker = None
        if existing is not None:
            await WorkerService(session).update(existing, to_create_worker)
            worker = existing
        else:
            worker = await Worker.create(session, to_create_worker, auto_commit=False)
        to_create_user.worker = worker
        created_user = await User.create(
            session=session, source=to_create_user, auto_commit=False
        )
        to_create_apikey.user_id = created_user.id
        to_create_apikey.user = created_user
        await ApiKey.create(session=session, source=to_create_apikey, auto_commit=False)
        if (cluster.state & ClusterState.READY) != ClusterState.READY:
            cluster.state |= ClusterState.READY
            await cluster.update(session=session, auto_commit=False)
        await session.commit()
        await session.refresh(worker)
        worker_dump = worker.model_dump()
        worker_dump["token"] = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

        return WorkerRegistrationPublic.model_validate(worker_dump)
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to create worker: {e}")


@router.put("/{id}", response_model=WorkerPublic)
async def update_worker(session: SessionDep, id: int, worker_in: WorkerUpdate):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")

    try:
        await WorkerService(session).update(worker, worker_in)
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
    # query a session bond worker
    worker: Worker = await Worker.one_by_id(session, user.worker.id)
    if not worker or worker.deleted_at is not None:
        raise NotFoundException(message="Worker not found")
    heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    input_dict = input.model_dump(exclude_unset=True)
    input_dict["heartbeat_time"] = heartbeat_time
    worker_dict = worker.model_dump()
    worker_dict.update(input_dict)
    try:
        to_update = worker.model_validate(worker_dict)
        to_update.compute_state()
        await WorkerService(session).update(worker, to_update)
        return Response(status_code=204)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update worker: {e}")


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
