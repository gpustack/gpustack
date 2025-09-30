from typing import List, Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import bindparam, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.mysql import JSON
from sqlmodel import col, or_, func
from gpustack.api.exceptions import (
    NotFoundException,
)

from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep, CurrentUserDep
from gpustack.schemas.models import (
    Model,
    MyModel,
    ModelPublic,
    ModelsPublic,
)

router = APIRouter()


@router.get("", response_model=ModelsPublic)
async def get_models(
    user: CurrentUserDep,
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

    target_class = Model
    if not user.is_admin:
        target_class = MyModel
        fields["user_id"] = user.id

    if params.watch:
        return StreamingResponse(
            target_class.streaming(
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

    return await target_class.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
        fields=fields,
    )


def build_pg_category_condition(category: str):
    if category == "":
        return cast(MyModel.categories, JSONB).op('@>')(cast('[]', JSONB))
    return cast(MyModel.categories, JSONB).op('?')(
        bindparam(f"category_{category}", category)
    )


# Add MySQL category condition construction function
def build_mysql_category_condition(category: str):
    if category == "":
        return func.json_length(MyModel.categories) == 0
    return func.json_contains(
        MyModel.categories, func.cast(func.json_quote(category), JSON), '$'
    )


def build_category_conditions(session, categories):
    dialect = session.bind.dialect.name
    if dialect == "sqlite":
        return [
            (
                col(MyModel.categories) == []
                if category == ""
                else col(MyModel.categories).contains(category)
            )
            for category in categories
        ]
    elif dialect == "postgresql":
        return [build_pg_category_condition(category) for category in categories]
    elif dialect == "mysql":
        return [build_mysql_category_condition(category) for category in categories]
    else:
        raise NotImplementedError(f'Unsupported database {dialect}')


def categories_filter(data: MyModel, categories: Optional[List[str]]):
    if not categories:
        return True

    data_categories = data.categories or []
    if not data_categories and "" in categories:
        return True

    return any(category in data_categories for category in categories)


@router.get("/{id}", response_model=ModelPublic)
async def get_model(
    session: SessionDep,
    id: int,
    user: CurrentUserDep,
):
    fields = {
        "id": id,
    }
    target_class = Model
    if not user.is_admin:
        target_class = MyModel
        fields["user_id"] = user.id
    model = await target_class.one_by_fields(session=session, fields=fields)
    if not model:
        raise NotFoundException(message="Model not found")

    return model
