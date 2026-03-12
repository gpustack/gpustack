from typing import List, Optional, Union
from sqlalchemy import bindparam, cast
from sqlmodel import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.mysql import JSON
from gpustack.schemas.models import Model
from gpustack.schemas.model_routes import ModelRoute, MyModel

category_classes = Union[
    Model,
    ModelRoute,
    MyModel,
]


def build_pg_category_condition(target_class: category_classes, category: str):
    if category == "":
        return cast(target_class.categories, JSONB).op('@>')(cast('[]', JSONB))
    return cast(target_class.categories, JSONB).op('?')(
        bindparam(f"category_{category}", category)
    )


# Add MySQL category condition construction function
def build_mysql_category_condition(target_class: category_classes, category: str):
    if category == "":
        return func.json_length(target_class.categories) == 0
    return func.json_contains(
        target_class.categories, func.cast(func.json_quote(category), JSON), '$'
    )


def build_category_conditions(session, target_class: category_classes, categories):
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


def categories_filter(data: category_classes, categories: Optional[List[str]]):
    if not categories:
        return True

    data_categories = data.categories or []
    if not data_categories and "" in categories:
        return True

    return any(category in data_categories for category in categories)
