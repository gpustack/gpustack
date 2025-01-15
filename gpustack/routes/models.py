import math
from typing import List, Optional, Union
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import bindparam, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import col, or_

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
from gpustack.schemas.workers import VendorEnum, Worker
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.gpu import parse_gpu_id

router = APIRouter()


@router.get("", response_model=ModelsPublic)
async def get_models(
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    if params.watch:
        return StreamingResponse(
            Model.streaming(
                session,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: categories_filter(data, categories),
            ),
            media_type="text/event-stream",
        )

    extra_conditions = []
    if categories:
        if session.bind.dialect.name == "sqlite":
            category_conditions = [
                (
                    col(Model.categories) == []
                    if category == ""
                    else col(Model.categories).contains(category)
                )
                for category in categories
            ]
            extra_conditions.append(or_(*category_conditions))
        else:  # For PostgreSQL
            category_conditions = [
                build_pg_category_condition(category) for category in categories
            ]
            extra_conditions.append(or_(*category_conditions))

    return await Model.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
    )


def build_pg_category_condition(category: str):
    if category == "":
        return cast(Model.categories, JSONB).op('@>')(cast('[]', JSONB))
    return cast(Model.categories, JSONB).op('?')(
        bindparam(f"category_{category}", category)
    )


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
async def get_model_instances(session: SessionDep, id: int, params: ListParamsDep):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    if params.watch:
        fields = {"model_id": id}
        return StreamingResponse(
            ModelInstance.streaming(session=session, fields=fields),
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


async def validate_gpu_ids(  # noqa: C901
    session: SessionDep, model_in: Union[ModelCreate, ModelUpdate]
):
    audio_model = is_audio_model(model_in)
    if audio_model and len(model_in.gpu_selector.gpu_ids) > 1:
        raise BadRequestException(
            message="Audio models are restricted to execution on a single NVIDIA GPU."
        )

    worker_name_set = set()
    for gpu_id in model_in.gpu_selector.gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise BadRequestException(message=f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        gpu_index = matched.get("gpu_index")
        worker_name_set.add(worker_name)

        worker = await Worker.one_by_field(session, "name", worker_name)
        if not worker:
            raise BadRequestException(message=f"Worker {worker_name} not found")

        if audio_model:
            for worker_gpu in worker.status.gpu_devices:
                if (
                    worker_gpu.index == gpu_index
                    and worker_gpu.type != VendorEnum.NVIDIA.value
                ):
                    raise BadRequestException(
                        "Audio models are supported only on NVIDIA GPUs and CPUs."
                    )

    if model_in.backend == BackendEnum.VLLM.value:
        if len(worker_name_set) > 1:
            raise BadRequestException(
                message="Model deployment with the vLLM backend is currently not supported on GPUs across different workers."
            )

        tp = find_parameter(model_in.backend_parameters, ["tensor-parallel-size", "tp"])
        if tp:
            raise BadRequestException(
                message="Use tensor-parallel-size and gpu-selector at the same time is not allowed."
            )

    if model_in.backend == BackendEnum.LLAMA_BOX.value:
        ts = find_parameter(model_in.backend_parameters, ["ts", "tensor-split"])
        if ts:
            raise BadRequestException(
                message="Use tensor-split and gpu-selector at the same time is not allowed."
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
        await model.update(session, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return model


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        await model.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
