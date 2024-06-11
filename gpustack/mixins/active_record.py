import asyncio
import json
import logging
import math
from typing import Any, AsyncGenerator

from fastapi.encoders import jsonable_encoder
from sqlalchemy import func
from sqlmodel import SQLModel, col, select, Session
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm.exc import FlushError
from sqlalchemy import event as sa_event

from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.bus import Event, EventType, event_bus


logger = logging.getLogger(__name__)


class ActiveRecordMixin:
    """ActiveRecordMixin provides a set of methods to interact with the database."""

    __config__ = None

    @property
    def primary_key(self):
        """Return the primary key of the object."""

        return self.__mapper__.primary_key_from_instance(self)

    @classmethod
    def first(cls, session: Session):
        """Return the first object of the model."""

        statement = select(cls)
        return session.exec(statement).first()

    @classmethod
    def one_by_id(cls, session: Session, id: int):
        """Return the object with the given id. Return None if not found."""

        obj = session.get(cls, id)
        return obj

    @classmethod
    def first_by_field(cls, session: Session, field: str, value: Any):
        """Return the first object with the given field and value. Return None if not found."""

        return cls.first_by_fields(session, {field: value})

    @classmethod
    def one_by_field(cls, session: Session, field: str, value: Any):
        """Return the object with the given field and value. Return None if not found."""
        return cls.one_by_fields(session, {field: value})

    @classmethod
    def first_by_fields(cls, session: Session, fields: dict):
        """
        Return the first object with the given fields and values.
        Return None if not found.
        """

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        return session.exec(statement).first()

    @classmethod
    def one_by_fields(cls, session: Session, fields: dict):
        """Return the object with the given fields and values. Return None if not found."""

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        return session.exec(statement).first()

    @classmethod
    def all_by_field(cls, session: Session, field: str, value: Any):
        """
        Return all objects with the given field and value.
        Return an empty list if not found.
        """

        statement = select(cls).where(getattr(cls, field) == value)
        return session.exec(statement).all()

    @classmethod
    def all_by_fields(cls, session: Session, fields: dict):
        """
        Return all objects with the given fields and values.
        Return an empty list if not found.
        """

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)
        return session.exec(statement).all()

    @classmethod
    def paginated_by_query(
        cls, session: Session, fields: dict, page: int, per_page: int
    ) -> PaginatedList[SQLModel]:
        """
        Return a paginated list of objects match the given fields and values.
        Return an empty list if not found.
        """

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(col(getattr(cls, key)).contains(value))

        if page is not None and per_page is not None:
            statement = statement.offset((page - 1) * per_page).limit(per_page)
        items = session.exec(statement).all()

        statement = select(func.count(cls.id))
        for key, value in fields.items():
            statement = statement.where(col(getattr(cls, key)).contains(value))

        count = session.exec(statement).one()
        total_page = math.ceil(count / per_page)
        pagination = Pagination(
            page=page,
            perPage=per_page,
            total=count,
            totalPage=total_page,
        )

        return PaginatedList[cls](items=items, pagination=pagination)

    @classmethod
    def convert_without_saving(
        cls, source: dict | SQLModel, update: dict | None = None
    ) -> SQLModel:
        """
        Convert the source to the model without saving to the database.
        Return None if failed.
        """

        if isinstance(source, SQLModel):
            obj = cls.from_orm(source, update=update)
        elif isinstance(source, dict):
            obj = cls.parse_obj(source, update=update)
        return obj

    @classmethod
    async def create(
        cls, session: Session, source: dict | SQLModel, update: dict | None = None
    ) -> SQLModel | None:
        """Create and save a new record for the model."""

        obj = cls.convert_without_saving(source, update)
        if obj is None:
            return None

        obj.save(session)
        return obj

    @classmethod
    async def create_or_update(
        cls, session: Session, source: dict | SQLModel, update: dict | None = None
    ) -> SQLModel | None:
        """Create or update a record for the model."""

        obj = cls.convert_without_saving(source, update)
        if obj is None:
            return None
        pk = cls.__mapper__.primary_key_from_instance(obj)
        if pk[0] is not None:
            existing = session.get(cls, pk)
            if existing is None:
                return None
            else:
                existing.update(session, obj)
                return existing
        else:
            return cls.create(session, obj)

    @classmethod
    def count(cls, session: Session) -> int:
        """Return the number of records in the model."""

        return len(cls.all(session))

    def refresh(self, session: Session):
        """Refresh the object from the database."""

        session.refresh(self)

    def save(self, session: Session):
        """Save the object to the database. Raise exception if failed."""

        session.add(self)
        try:
            session.commit()
            session.refresh(self)
        except (IntegrityError, OperationalError, FlushError) as e:
            session.rollback()
            raise e

    async def update(self, session: Session, source: dict | SQLModel):
        """Update the object with the source and save to the database."""

        if isinstance(source, SQLModel):
            source = source.model_dump(exclude_unset=True)

        for key, value in source.items():
            setattr(self, key, value)
        self.save(session)

    async def delete(self, session: Session):
        """Delete the object from the database."""

        session.delete(self)
        session.commit()

    @classmethod
    def all(cls, session: Session):
        """Return all objects of the model."""

        return session.exec(select(cls)).all()

    @classmethod
    async def delete_all(cls, session: Session):
        """Delete all objects of the model."""

        for obj in cls.all(session):
            obj.delete(session)
            await cls._publish_event(EventType.DELETED, obj)

    @classmethod
    async def _publish_event(cls, event_type: str, data: Any):
        try:
            await event_bus.publish(
                cls.__name__.lower(), Event(type=event_type, data=data)
            )
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    @classmethod
    async def subscribe(cls, session: Session) -> AsyncGenerator[Event, None]:
        items = cls.all(session)
        for item in items:
            yield Event(type=EventType.CREATED, data=item)

        subscriber = event_bus.subscribe(cls.__name__.lower())

        try:
            while True:
                event = await subscriber.receive()
                yield event
        finally:
            event_bus.unsubscribe(cls.__name__.lower(), subscriber)

    @classmethod
    async def streaming(cls, session: Session) -> AsyncGenerator[str, None]:
        async for event in cls.subscribe(session):
            yield json.dumps(jsonable_encoder(event), separators=(",", ":")) + "\n\n"

    @classmethod
    def __declare_last__(cls):
        """
        Automatically add hooks to the model class to publish events
        after creation, update, and deletion.

        Reference:
            https://docs.sqlalchemy.org/en/20/orm/declarative_config.html#declare-last
            https://docs.sqlalchemy.org/en/20/orm/events.html#mapper-events
        """

        sa_event.listen(cls, "after_insert", cls.after_create_hook)
        sa_event.listen(cls, "after_update", cls.after_update_hook)
        sa_event.listen(cls, "after_delete", cls.after_delete_hook)

    @staticmethod
    def after_create_hook(mapper, connection, target):
        asyncio.create_task(target._publish_event(EventType.CREATED, target))

    @staticmethod
    def after_update_hook(mapper, connection, target):
        asyncio.create_task(target._publish_event(EventType.UPDATED, target))

    @staticmethod
    def after_delete_hook(mapper, connection, target):
        asyncio.create_task(target._publish_event(EventType.DELETED, target))
