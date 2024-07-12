import math
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.models import ModelInstance, ModelInstancesPublic
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
)

router = APIRouter()


@router.get("", response_model=ModelsPublic)
async def get_models(session: SessionDep, params: ListParamsDep, search: str = None):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    if params.watch:
        return StreamingResponse(
            Model.streaming(session, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await Model.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


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


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    existing = await Model.one_by_field(session, "name", model_in.name)
    if existing:
        raise AlreadyExistsException(message=f"Model f{model_in.name} already exists")

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
