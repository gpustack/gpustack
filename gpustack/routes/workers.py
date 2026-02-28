import secrets
import datetime
import base64
import uuid
import logging
import asyncio
from typing import Optional, List, Dict, Any, Set
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, Response, Request
from fastapi.responses import StreamingResponse, RedirectResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    ForbiddenException,
    InvalidException,
)
from gpustack.config.config import get_global_config
from gpustack.server.deps import (
    SessionDep,
    CurrentUserDep,
)
from gpustack.server.db import async_session
from gpustack.server.worker_status_buffer import (
    heartbeat_flush_buffer,
    heartbeat_flush_buffer_lock,
    worker_status_flush_buffer,
    worker_status_flush_buffer_lock,
)
from gpustack.schemas.workers import (
    WorkerCreate,
    WorkerListParams,
    WorkerPublic,
    WorkerUpdate,
    WorkersPublic,
    Worker,
    WorkerRegistrationPublic,
    WorkerStatusStored,
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
from gpustack.utils.grafana import resolve_grafana_base_url

router = APIRouter()
system_name_prefix = "system/worker"
logger = logging.getLogger(__name__)


def to_worker_public(input: Worker, me: bool) -> WorkerPublic:
    data = input.model_dump()
    if me:
        data['me'] = me
    return WorkerPublic.model_validate(data)


@router.get("", response_model=WorkersPublic)
async def get_workers(
    user: CurrentUserDep,
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
            Worker.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
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

    async with async_session() as session:
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


@router.get("/{id}/dashboard")
async def get_worker_dashboard(
    session: SessionDep,
    id: int,
    request: Request,
):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")

    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_worker_dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if worker.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, worker.cluster_id)

    query_params = {}
    if cluster is not None:
        query_params["var-cluster_name"] = cluster.name
    query_params["var-worker_name"] = worker.name

    grafana_base = resolve_grafana_base_url(cfg, request)
    slug = "gpustack-worker"
    dashboard_url = f"{grafana_base}/d/{cfg.grafana_worker_dashboard_uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


def update_worker_data(
    worker_in: WorkerCreate,
    existing: Optional[Worker] = None,
    **kwargs,
) -> Worker:
    to_create_worker = None
    if existing is not None:
        # Preserve maintenance field from existing worker if not explicitly set in worker_in
        incoming_data = worker_in.model_dump()
        if (
            incoming_data.get("maintenance") is None
            and existing.maintenance is not None
        ):
            incoming_data["maintenance"] = existing.maintenance

        to_create_worker = Worker.model_validate(
            {
                **existing.model_dump(),
                **incoming_data,
                "labels": {
                    **existing.labels,
                    **worker_in.labels,
                },
                "cluster_id": existing.cluster_id,
                "state": WorkerStateEnum.READY,
            }
        )
    else:
        # new worker should ignore the reported worker_uuid
        to_create_worker = Worker.model_validate(
            {
                **worker_in.model_dump(exclude={"name", "worker_uuid"}),
                "name": worker_in.name or worker_in.hostname,
                "worker_uuid": "",
                "state": WorkerStateEnum.READY,
                **kwargs,
            }
        )
        if kwargs.get("cluster") is not None:
            to_create_worker.cluster = kwargs.get("cluster")
    to_create_worker.compute_state()
    return to_create_worker


def filter_workers_by_fields(
    workers: List[Worker],
    fields: Optional[Dict[str, Any]],
    fuzzy_fields: Dict[str, str] = {},
) -> List[Worker]:
    if not fields and not fuzzy_fields:
        return workers

    to_return = []
    for worker in workers:
        match = True
        if fields:
            for k, v in fields.items():
                if getattr(worker, k, None) != v:
                    match = False
                    break
        if not match:
            continue

        if fuzzy_fields:
            for k, v in fuzzy_fields.items():
                attr = getattr(worker, k, None)
                if not isinstance(attr, str) or v.lower() not in attr.lower():
                    match = False
                    break

        if match:
            to_return.append(worker)
    return to_return


def get_existing_worker(
    cluster_id: int, worker_in: WorkerCreate, workers: List[Worker]
) -> Optional[Worker]:
    static_fields = {
        "deleted_at": None,
        "cluster_id": cluster_id,
    }

    if worker_in.name == "":
        return None

    # find existing worker by worker_id (for worker restarts)
    if worker_in.worker_id is not None:
        fields = {**static_fields, "id": worker_in.worker_id}
        existing_worker = next(iter(filter_workers_by_fields(workers, fields)), None)
        if existing_worker is not None:
            return existing_worker

    # find existing worker by external_id or worker_uuid
    for field in ["external_id", "worker_uuid"]:
        value = getattr(worker_in, field, None)
        if value is None:
            continue
        fields = {**static_fields, field: value}
        existing_worker = next(iter(filter_workers_by_fields(workers, fields)), None)
        if existing_worker is not None:
            return existing_worker

    # find existing worker by name
    if worker_in.labels and worker_in.labels.get("gpustack.existence-check"):
        fields = {"name": worker_in.name}
        existing_worker = next(iter(filter_workers_by_fields(workers, fields)), None)
        if existing_worker is not None:
            if existing_worker.cluster_id != cluster_id:
                raise AlreadyExistsException(
                    message=f"worker with name {worker_in.name} already exists in another cluster"
                )
            return existing_worker

    return None


def check_worker_name_conflict(
    name: str, workers: List[Worker], existing_id: Optional[int] = None
):
    if name == "":
        if existing_id is not None:
            raise InvalidException(message="worker name cannot be empty")
        return
    workers = [worker for worker in workers if worker.id != existing_id]
    name_conflict_fields = {"name": name}
    name_conflict_worker = next(
        iter(filter_workers_by_fields(workers, name_conflict_fields)), None
    )
    if name_conflict_worker is not None:
        raise AlreadyExistsException(message=f"worker with name {name} already exists")


def find_available_worker_name(
    original_name: str, current_name: str, related_names: Set[str]
) -> str:
    if original_name not in related_names:
        return original_name

    index = 1
    if current_name.startswith(f"{original_name}-"):
        suffix = current_name[len(original_name) + 1 :]
        if suffix.isdigit():
            index = int(suffix) + 1

    new_name = f"{original_name}-{index}"
    while new_name in related_names:
        index += 1
        new_name = f"{original_name}-{index}"
    return new_name


async def retry_create_worker(
    session: AsyncSession, to_create: Worker, workers: List[Worker]
) -> Worker:
    related_workers = filter_workers_by_fields(
        workers,
        fields={
            "deleted_at": None,
        },
        fuzzy_fields={"name": to_create.name},
    )
    related_names = set(worker.name for worker in related_workers)
    original_name = to_create.name
    current_name = to_create.name
    for i in range(5):
        try:
            current_name = find_available_worker_name(
                original_name, current_name, related_names
            )
            to_create.name = current_name
            to_create.labels["worker-name"] = current_name
            new_worker = await Worker.create(session, to_create, auto_commit=False)
            return new_worker
        except IntegrityError:
            logger.warning(
                f"Worker name collision detected for worker name {to_create.name}, retrying... (attempt {i + 1}/5)"
            )
            related_names.add(current_name)
            await asyncio.sleep(0.1)  # small delay before retrying to reduce contention
    raise InternalServerErrorException(
        message="Failed to create worker with unique name after multiple attempts"
    )


def retry_create_unique_worker_uuid(workers: List[Worker]) -> str:
    current_uuids = set(
        worker.worker_uuid for worker in workers if worker.worker_uuid != ""
    )
    for i in range(5):
        new_uuid = str(uuid.uuid4())
        if new_uuid not in current_uuids:
            return new_uuid
        logger.warning(
            f"UUID collision detected for worker_uuid {new_uuid}, retrying... (attempt {i + 1}/5)"
        )
    # might not be necessary to retry so many times, but just in case, we want to make sure
    # the system can recover from such a rare event without manual intervention
    raise InternalServerErrorException(
        message="Failed to generate unique worker UUID after multiple attempts"
    )


@router.post("", response_model=WorkerRegistrationPublic)
async def create_worker(
    user: CurrentUserDep, session: SessionDep, worker_in: WorkerCreate
):
    cluster_id = (
        worker_in.cluster_id if worker_in.cluster_id is not None else user.cluster_id
    )
    if cluster_id is None:
        raise ForbiddenException(message="Missing cluster_id for worker registration")
    all_workers = await Worker.all_by_fields(session, {"deleted_at": None})
    existing_worker = get_existing_worker(cluster_id, worker_in, all_workers)
    check_worker_name_conflict(
        worker_in.name,
        all_workers,
        existing_id=existing_worker.id if existing_worker else None,
    )
    if existing_worker is None:
        if worker_in.external_id is not None:
            # avoid creating a worker with a non-existent external_id
            raise NotFoundException(
                message=f"worker with external_id {worker_in.external_id} not found"
            )
    else:
        existing_worker = await Worker.one_by_id(
            session=session, id=existing_worker.id, for_update=True
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
    if new_worker.worker_uuid == "":
        new_worker.worker_uuid = retry_create_unique_worker_uuid(all_workers)

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
            worker = await retry_create_worker(session, new_worker, all_workers)
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


async def create_worker_status(user: CurrentUserDep, input: WorkerStatusStored):
    if user.worker is None:
        raise ForbiddenException(message="Failed to find related worker")

    heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    input_dict = input.model_dump(exclude_unset=True)
    input_dict["heartbeat_time"] = heartbeat_time

    # Add worker status to buffer for batch update
    async with worker_status_flush_buffer_lock:
        worker_status_flush_buffer[user.worker.id] = input_dict

    return Response(status_code=204)


async def heartbeat(user: CurrentUserDep):
    if user.worker is None:
        raise ForbiddenException(message="Failed to find related worker")

    # Add worker ID to buffer for batch update
    async with heartbeat_flush_buffer_lock:
        heartbeat_flush_buffer.add(user.worker.id)

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
