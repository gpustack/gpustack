"""Immutable resource bindings shared by ordinary ORM write paths."""

from contextlib import contextmanager
from typing import Any, ClassVar, Dict, Iterable, Iterator, Optional, Tuple

from sqlalchemy import event, inspect, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper, Session

from gpustack.api.exceptions import InvalidException


class ImmutableFieldsMixin:
    # Ownership is protected automatically; models declare additional bindings.
    __immutable_fields__: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _immutable_fields(cls) -> Tuple[str, ...]:
        return tuple(
            name
            for name in dict.fromkeys(("owner_principal_id", *cls.__immutable_fields__))
            if name in cls.__mapper__.column_attrs
        )

    def _assert_immutable_update(self, source: Dict[str, Any]) -> None:
        """Validate the entire patch before assigning any of its fields."""
        state = inspect(self)
        for name in self._immutable_fields():
            # Unloaded values are checked against storage during flush without
            # triggering synchronous lazy loads from an async update call.
            if name in source and name in state.dict:
                _assert_unchanged(self, name, state.dict[name], source[name])


def _assert_unchanged(resource, name, old, new):
    if old == new:
        return
    allowed = inspect(resource).info.get("immutable_field_allowance", {})
    if name in allowed and allowed[name] == new:
        return
    raise InvalidException(message=f"{name} cannot be changed")


@contextmanager
def allow_immutable_field_update(
    resource: ImmutableFieldsMixin, name: str, value: Any
) -> Iterator[None]:
    """Permit one server-derived binding value on one object until scope exit.

    Internal reconciliation must flush inside this scope. Callers must derive
    the value from an authorized parent, never directly from a request body.
    """
    if name not in resource._immutable_fields():
        raise ValueError(f"{name} is not an immutable field")
    state = inspect(resource)
    previous = state.info.get("immutable_field_allowance")
    state.info["immutable_field_allowance"] = {**(previous or {}), name: value}
    try:
        yield
    finally:
        if previous is None:
            state.info.pop("immutable_field_allowance", None)
        else:
            state.info["immutable_field_allowance"] = previous


def _assert_immutable_history(resource, connection=None):
    state = inspect(resource)
    if not state.persistent:
        return
    for name in resource._immutable_fields():
        history = state.attrs[name].history
        if not history.has_changes():
            continue
        if history.deleted:
            old = history.deleted[0]
        else:
            # Assignments to expired or deferred attributes have no old value
            # in ORM history. Compare with the row instead of assuming a change.
            if connection is None:
                connection = state.session.connection(
                    bind_arguments={"mapper": state.mapper}
                )
            column = state.mapper.column_attrs[name].columns[0]
            statement = select(column).where(
                *(
                    key == value
                    for key, value in zip(state.mapper.primary_key, state.identity)
                )
            )
            old = connection.execute(statement).scalar_one()
        _assert_unchanged(resource, name, old, state.dict[name])


@event.listens_for(Session, "before_flush")
def validate_immutable_fields(
    session: Session, flush_context: Any, instances: Optional[Iterable[Any]]
) -> None:
    """Protect direct assignments, saves and batches before emitting writes."""
    targets = None if instances is None else {id(obj) for obj in instances}
    for resource in session.dirty:
        if not isinstance(resource, ImmutableFieldsMixin):
            continue
        if targets is not None and id(resource) not in targets:
            continue
        _assert_immutable_history(resource)


@event.listens_for(Mapper, "before_update")
def validate_immutable_relationships(
    mapper: Mapper, connection: Connection, resource: Any
) -> None:
    """Check foreign keys synchronized from relationships during the flush."""
    if isinstance(resource, ImmutableFieldsMixin):
        _assert_immutable_history(resource, connection)
