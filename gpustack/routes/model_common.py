from enum import Enum
from typing import Any, List, Optional, Union
from sqlalchemy import and_, bindparam, cast, or_
from sqlmodel import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.mysql import JSON
from gpustack.schemas.models import Model, ModelStateEnum
from gpustack.schemas.model_routes import ModelRoute, MyModel

category_classes = Union[
    Model,
    ModelRoute,
    MyModel,
]


class ModelStateFilterEnum(str, Enum):
    """Values of the ``?state=`` list filter.

    Not to be confused with ``ModelStateEnum``, the resource field on the
    Model row: this is a *filter* over readiness, it predates that field, and
    its name, values and meaning are frozen for backward compatibility.
    What changed is where it gets its answer from — see
    ``model_state_condition``.
    """

    READY = "ready"
    NOT_READY = "not_ready"
    STOPPED = "stopped"


# ``Model.state`` is written by ``sync_model_status``
# (``server/controllers.py``), the single owner of the model's status fields.
#
# This filter asks "is anything up", which is *not* the same question as the
# servability gate's "can this serve" — pre-PD the two coincided, and under PD
# they do not. The filter's meaning is frozen (V28), so it stays on the first
# question: RUNNING and PARTIAL are exactly the values that mean at least one
# member is ready, PENDING and ERROR are exactly the values that mean none is.
# A group whose router is down therefore still lists as ready, which is the
# point — hiding it would hide the very row the user needs to look at, and the
# counter-based filter it replaces listed it too.
#
# For a role-less model PARTIAL is unreachable (one ready replica serves), so
# this degenerates to ``state == RUNNING`` there, which is equivalent to
# ``ready_replicas > 0``: the filter's answer does not shift a byte for any
# model that exists today.
_READY_MODEL_STATES = (ModelStateEnum.RUNNING, ModelStateEnum.PARTIAL)
_NOT_READY_MODEL_STATES = (ModelStateEnum.PENDING, ModelStateEnum.ERROR)


def model_state_condition(state: Optional[ModelStateFilterEnum]):
    """SQL condition for ``GET /v2/models?state=``, derived from
    ``Model.state``.

    ``stopped`` stays on ``replicas`` on purpose: it asks about intent, not
    status, and ``ModelStateEnum`` deliberately has no stopped value — a
    stopped model is PENDING with nothing to be ready.

    A NULL ``state`` falls back to the replica counters. The migration
    backfills existing rows, so this covers a newly created model and any row
    a reconcile has not reached yet — and those rows have to keep answering
    both ``ready`` and ``not_ready``, since a three-valued ``IN`` would drop
    them out of *both* filters and make the list page look like it lost data.
    """
    if state == ModelStateFilterEnum.READY:
        return or_(
            Model.state.in_(_READY_MODEL_STATES),
            and_(Model.state.is_(None), Model.ready_replicas > 0),
        )
    if state == ModelStateFilterEnum.NOT_READY:
        return and_(
            Model.replicas > 0,
            or_(
                Model.state.in_(_NOT_READY_MODEL_STATES),
                and_(Model.state.is_(None), Model.ready_replicas == 0),
            ),
        )
    if state == ModelStateFilterEnum.STOPPED:
        return Model.replicas == 0
    return None


def model_state_stream_filter(
    data: Any,
    state: Optional[ModelStateFilterEnum],
) -> bool:
    """Python-side mirror of :func:`model_state_condition` for watch streams.

    Kept next to the SQL it mirrors so the two can't drift, and tolerant of
    partial payloads for the same reason :func:`state_stream_filter` is:
    ID-only DELETED events carry no status at all, and dropping them would
    leave watch clients holding a stale row forever.
    """
    if state is None:
        return True
    total = getattr(data, "replicas", None)
    if total is None:
        return True
    if state == ModelStateFilterEnum.STOPPED:
        return total == 0

    model_state = getattr(data, "state", None)
    if model_state is None:
        # Same fallback as the SQL: a row from before the column existed.
        ready_replicas = getattr(data, "ready_replicas", None)
        if ready_replicas is None:
            return True
        is_ready = ready_replicas > 0
    else:
        is_ready = model_state in _READY_MODEL_STATES

    if state == ModelStateFilterEnum.READY:
        return is_ready
    if state == ModelStateFilterEnum.NOT_READY:
        return not is_ready and total > 0
    return True


def state_stream_filter(
    data: Any,
    state: Optional[ModelStateFilterEnum],
    ready_attr: str,
    total_attr: str,
) -> bool:
    """Python-side mirror of the counter-based SQL readiness filter for watch
    streams.

    ``ready_attr``/``total_attr`` name the readiness counters that differ per
    resource (``ready_targets``/``targets`` for a route). A Model answers the
    same filter from its own status field instead — see
    :func:`model_state_stream_filter`."""
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
