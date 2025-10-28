import math
from typing import List, Optional, Union
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from gpustack_runtime.detector import ManufacturerEnum
from sqlalchemy import bindparam, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.mysql import JSON
from sqlmodel import col, or_, func
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstancesPublic,
    is_audio_model,
    BackendEnum,
)
from gpustack.schemas.workers import GPUDeviceInfo, Worker
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
)
from gpustack.schemas.models import (
    ModelAccessUpdate,
    ModelUserAccessExtended,
    ModelAccessList,
    MyModel,
)
from gpustack.schemas.users import User
from gpustack.server.services import ModelService, WorkerService
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id

router = APIRouter()


@router.get("", response_model=ModelsPublic)
async def get_models(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    cluster_id: int = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id

    if params.watch:
        return StreamingResponse(
            Model.streaming(
                engine,
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: categories_filter(data, categories),
            ),
            media_type="text/event-stream",
        )

    extra_conditions = []
    if categories:
        conditions = build_category_conditions(session, categories)
        extra_conditions.append(or_(*conditions))

    return await Model.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
        fields=fields,
    )


def build_pg_category_condition(target_class: Union[Model, MyModel], category: str):
    if category == "":
        return cast(target_class.categories, JSONB).op('@>')(cast('[]', JSONB))
    return cast(target_class.categories, JSONB).op('?')(
        bindparam(f"category_{category}", category)
    )


# Add MySQL category condition construction function
def build_mysql_category_condition(target_class: Union[Model, MyModel], category: str):
    if category == "":
        return func.json_length(target_class.categories) == 0
    return func.json_contains(
        target_class.categories, func.cast(func.json_quote(category), JSON), '$'
    )


def build_category_conditions(session, target_class: Union[Model, MyModel], categories):
    dialect = session.bind.dialect.name
    if dialect == "postgresql":
        return [
            build_pg_category_condition(target_class, category)
            for category in categories
        ]
    elif dialect == "mysql":
        return [
            build_mysql_category_condition(target_class, category)
            for category in categories
        ]
    else:
        raise NotImplementedError(f'Unsupported database {dialect}')


def categories_filter(data: Model, categories: Optional[List[str]]):
    if not categories:
        return True

    data_categories = data.categories or []
    if not data_categories and "" in categories:
        return True

    return any(category in data_categories for category in categories)


@router.get("/{id}", response_model=ModelPublic)
async def get_model(session: SessionDep, id: int):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return model


@router.get("/{id}/instances", response_model=ModelInstancesPublic)
async def get_model_instances(
    engine: EngineDep, session: SessionDep, id: int, params: ListParamsDep
):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    if params.watch:
        fields = {"model_id": id}
        return StreamingResponse(
            ModelInstance.streaming(engine, fields=fields),
            media_type="text/event-stream",
        )

    instances = model.instances
    count = len(instances)
    total_page = math.ceil(count / params.perPage)
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return ModelInstancesPublic(items=instances, pagination=pagination)


async def validate_model_in(
    session: SessionDep, model_in: Union[ModelCreate, ModelUpdate]
):
    if model_in.gpu_selector is not None and model_in.replicas > 0:
        await validate_gpu_ids(session, model_in)

    if model_in.backend_parameters:
        param_gpu_layers = find_parameter(
            model_in.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
        )

        if param_gpu_layers:
            int_param_gpu_layers = safe_int(param_gpu_layers, None)
            if (
                not param_gpu_layers.isdigit()
                or int_param_gpu_layers < 0
                or int_param_gpu_layers > 999
            ):
                raise BadRequestException(
                    message="Invalid backend parameter --gpu-layers. Please provide an integer in the range 0-999 (inclusive)."
                )

            if (
                int_param_gpu_layers == 0
                and model_in.gpu_selector is not None
                and len(model_in.gpu_selector.gpu_ids) > 0
            ):
                raise BadRequestException(
                    message="Cannot set --gpu-layers to 0 and manually select GPUs at the same time. Setting --gpu-layers to 0 means running on CPU only."
                )

        param_port = find_parameter(model_in.backend_parameters, ["port"])

        if param_port:
            raise BadRequestException(
                message="Setting the port using --port is not supported."
            )


async def validate_gpu_ids(  # noqa: C901
    session: SessionDep, model_in: Union[ModelCreate, ModelUpdate]
):

    if (
        model_in.gpu_selector
        and model_in.gpu_selector.gpu_ids
        and model_in.gpu_selector.gpus_per_replica
    ):
        if len(model_in.gpu_selector.gpu_ids) < model_in.gpu_selector.gpus_per_replica:
            raise BadRequestException(
                message="The number of selected GPUs must be greater than or equal to gpus_per_replica."
            )

    audio_model = is_audio_model(model_in)
    if audio_model and len(model_in.gpu_selector.gpu_ids) > 1:
        raise BadRequestException(
            message="Audio models are restricted to execution on a single NVIDIA GPU."
        )

    model_backend = model_in.backend

    worker_name_set = set()
    for gpu_id in model_in.gpu_selector.gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise BadRequestException(message=f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        gpu_index = safe_int(matched.get("gpu_index"), -1)
        worker_name_set.add(worker_name)

        worker = await WorkerService(session).get_by_name(worker_name)
        if not worker:
            raise BadRequestException(message=f"Worker {worker_name} not found")

        gpu = (
            next(
                (gpu for gpu in worker.status.gpu_devices if gpu.index == gpu_index),
                None,
            )
            if worker.status and worker.status.gpu_devices
            else None
        )
        if gpu:
            validate_gpu(gpu, is_audio_model=audio_model, model_backend=model_backend)

        worker_os = (
            worker.labels.get("os", "unknown")
            if worker.labels is not None
            else "unknown"
        )
        if model_backend == BackendEnum.VLLM and worker_os != "linux":
            raise BadRequestException(
                message=f'vLLM backend is only supported on Linux, but the selected worker "{worker.name}" is running on {worker_os.capitalize()}.'
            )

        if model_backend == BackendEnum.VLLM and len(worker_name_set) > 1:
            await validate_distributed_vllm_limit_per_worker(session, model_in, worker)

    if model_backend == BackendEnum.LLAMA_BOX:
        ts = find_parameter(model_in.backend_parameters, ["ts", "tensor-split"])
        if ts:
            raise BadRequestException(
                message="Use tensor-split and gpu-selector at the same time is not allowed."
            )


def validate_gpu(
    gpu_device: GPUDeviceInfo, is_audio_model: bool = False, model_backend: str = ""
):
    if is_audio_model and gpu_device.vendor != ManufacturerEnum.NVIDIA.value:
        raise BadRequestException(
            "Audio models are supported only on NVIDIA GPUs and CPUs."
        )

    if (
        model_backend == BackendEnum.ASCEND_MINDIE
        and gpu_device.vendor != ManufacturerEnum.ASCEND.value
    ):
        raise BadRequestException(
            f"Ascend MindIE backend requires Ascend NPUs. Selected {gpu_device.vendor} GPU is not supported."
        )


async def validate_distributed_vllm_limit_per_worker(
    session: AsyncSession, model: Union[ModelCreate, ModelUpdate], worker: Worker
):
    """
    Validate that there is no more than one distributed vLLM instance per worker.
    """
    instances = await ModelInstance.all_by_field(session, "worker_id", worker.id)
    for instance in instances:
        if (
            instance.distributed_servers
            and instance.distributed_servers.subordinate_workers
            and instance.model_name != model.name
        ):
            raise BadRequestException(
                message=f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
            )


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    existing = await Model.one_by_field(session, "name", model_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Model '{model_in.name}' already exists. "
            "Please choose a different name or check the existing model."
        )

    await validate_model_in(session, model_in)

    try:
        model = await Model.create(session, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


@router.put("/{id}", response_model=ModelPublic)
async def update_model(session: SessionDep, id: int, model_in: ModelUpdate):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    await validate_model_in(session, model_in)

    try:
        await ModelService(session).update(model, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return model


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        await ModelService(session).delete(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")


def model_access_list(model: Model) -> List[ModelUserAccessExtended]:
    return [
        ModelUserAccessExtended(
            id=access.id,
            username=access.username,
            full_name=access.full_name,
            avatar_url=access.avatar_url,
            # Add more user fields here if needed
        )
        for access in model.users
    ]


@router.get("/{id}/access", response_model=ModelAccessList)
async def get_model_access(session: SessionDep, id: int):
    model: Model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return ModelAccessList(items=model_access_list(model))


@router.post("/{id}/access", response_model=ModelAccessList)
async def add_model_access(
    session: SessionDep, id: int, access_request: ModelAccessUpdate
):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")
    extra_conditions = [
        col(User.id).in_([user.id for user in access_request.users]),
    ]

    users = await User.all_by_fields(
        session=session, fields={}, extra_conditions=extra_conditions
    )
    if len(users) != len(access_request.users):
        existing_user_ids = {user.id for user in users}
        for req_user in access_request.users:
            if req_user.id not in existing_user_ids:
                raise NotFoundException(message=f"User ID {req_user.id} not found")

    model.users = list(users)
    if access_request.access_policy is not None:
        model.access_policy = access_request.access_policy
    try:
        await ModelService(session).update(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to add model access: {e}")
    await session.refresh(model)
    return ModelAccessList(items=model_access_list(model))
