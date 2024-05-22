from fastapi import APIRouter

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
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
async def get_models(session: SessionDep, params: ListParamsDep):
    fields = {}
    if params.query:
        fields = {"name": params.query}
    return Model.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=ModelPublic)
async def get_model(session: SessionDep, id: int):
    model = Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return model


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    model = Model.model_validate(model_in)

    existing = Model.one_by_field(session, "name", model.name)
    if existing:
        raise AlreadyExistsException(message=f"Model f{model.name} already exists")

    try:
        model.save(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


@router.put("/{id}", response_model=ModelPublic)
async def update_model(session: SessionDep, id: int, model_in: ModelUpdate):
    model = Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        model.update(session, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return model


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        model.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
