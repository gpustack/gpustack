from fastapi import APIRouter, HTTPException


from ..core.deps import ListParamsDep, SessionDep
from ..schemas.models import Model, ModelCreate, ModelUpdate, ModelPublic, ModelsPublic

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
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    model = Model.model_validate(model_in)

    return model.save(session)


@router.put("/{id}", response_model=ModelPublic)
async def update_model(session: SessionDep, model_in: ModelUpdate):
    model = Model.model_validate(model_in)
    return model.save(session)


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = Model.one_by_id(session, id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model.delete(session)
