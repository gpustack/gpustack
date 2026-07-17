from enum import Enum
from typing import Any, List, Optional, Union
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


class ModelStateFilterEnum(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    STOPPED = "stopped"


def state_stream_filter(
    data: Any,
    state: Optional[ModelStateFilterEnum],
    ready_attr: str,
    total_attr: str,
) -> bool:
    """Python-side mirror of the SQL readiness filter for watch streams.

    ``ready_attr``/``total_attr`` name the readiness counters that differ
    per resource (``ready_replicas``/``replicas`` for a Model,
    ``ready_targets``/``targets`` for a route)."""
    if state is None:
        return True
    ready = getattr(data, ready_attr, None)
    total = getattr(data, total_attr, None)
    # Partial payloads (e.g. ID-only DELETED events shaped ``{"id": ...}``)
    # don't carry readiness counters. Let them through so watch clients can
    # still drop the row by ID instead of holding a stale copy forever.
    if ready is None or total is None:
        return True
    if state == ModelStateFilterEnum.READY:
        return ready > 0
    if state == ModelStateFilterEnum.NOT_READY:
        return ready == 0 and total > 0
    if state == ModelStateFilterEnum.STOPPED:
        return total == 0
    return True


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

    # Partial payloads (e.g. ID-only DELETED events shaped ``{"id": ...}``)
    # don't carry categories. Let them through so watch clients can drop the
    # row by ID instead of the stream erroring on a missing attribute.
    if not hasattr(data, "categories"):
        return True

    data_categories = data.categories or []
    if not data_categories and "" in categories:
        return True

    return any(category in data_categories for category in categories)
