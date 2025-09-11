import asyncio
from datetime import datetime, timedelta, timezone
import importlib
import json
import logging
import math
from typing import Any, AsyncGenerator, Callable, List, Optional, Union, Tuple

from fastapi.encoders import jsonable_encoder
from sqlalchemy import func, event as sa_event, inspect
from sqlmodel import SQLModel, and_, asc, col, desc, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.ext.asyncio import AsyncEngine
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.bus import Event, EventType, event_bus


logger = logging.getLogger(__name__)


class CommitEvent:
    name: str
    event: Event

    def __init__(self, name: str, type: EventType, data: Any):
        self.name = name
        self.event = Event(type=type, data=data)


@sa_event.listens_for(Session, "before_flush")
def find_history(session: AsyncSession, flush_context, instances):
    events: List[CommitEvent] = session.info.get("pending_events", [])
    if len(events) == 0:
        return
    dirty_objs = [obj for obj in session.dirty if instances is None or obj in instances]
    for event in events:
        if event.event.data not in dirty_objs:
            continue
        obj = event.event.data
        state = inspect(obj)
        for attr in state.attrs:
            hist = attr.history
            if hist.has_changes():
                event.event.changed_fields[attr.key] = (hist.deleted, hist.added)


@sa_event.listens_for(Session, "before_commit")
def refresh_objects_before_commit(session: AsyncSession):
    events: List[CommitEvent] = session.info.get("pending_events", [])
    for event in events:
        bus_event = event.event
        # refresh the object before commit to ensure the data is up-to-date
        if (
            isinstance(bus_event.data, SQLModel)
            and bus_event.data not in session.deleted
        ):
            try:
                session.refresh(bus_event.data)
            except Exception as e:
                logger.error(f"Failed to refresh object {bus_event.data}: {e}")


# commit hook to send events after a database commit
@sa_event.listens_for(Session, "after_commit")
def send_post_commit_events(session: AsyncSession):
    events: List[CommitEvent] = session.info.pop("pending_events", [])
    for event in events:
        # copy before submit to avoid mutation
        id = getattr(event.event.data, "id", None)
        logger.debug(f"Sending event {event.name} of type {event.event.type}, id {id}")
        bus_event = event.event
        copied_dict = bus_event.data.model_dump()
        bus_event.data = type(bus_event.data).model_validate(copied_dict)
        try:
            asyncio.create_task(event_bus.publish(event.name, bus_event))
        except Exception as e:
            logger.error(f"Failed to publish events: {e}")


class ActiveRecordMixin:
    """ActiveRecordMixin provides a set of methods to interact with the database."""

    __config__ = None

    @property
    def primary_key(self):
        """Return the primary key of the object."""

        return self.__mapper__.primary_key_from_instance(self)

    @classmethod
    async def first(cls, session: AsyncSession):
        """Return the first object of the model."""

        statement = select(cls)
        result = await session.exec(statement)
        return result.first()

    @classmethod
    async def one_by_id(cls, session: AsyncSession, id: int):
        """Return the object with the given id. Return None if not found."""

        return await session.get(cls, id)

    @classmethod
    async def first_by_field(cls, session: AsyncSession, field: str, value: Any):
        """Return the first object with the given field and value. Return None if not found."""

        return await cls.first_by_fields(session, {field: value})

    @classmethod
    async def one_by_field(cls, session: AsyncSession, field: str, value: Any):
        """Return the object with the given field and value. Return None if not found."""

        return await cls.one_by_fields(session, {field: value})

    @classmethod
    async def first_by_fields(cls, session: AsyncSession, fields: dict):
        """
        Return the first object with the given fields and values.
        Return None if not found.
        """

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        result = await session.exec(statement)
        return result.first()

    @classmethod
    async def one_by_fields(cls, session: AsyncSession, fields: dict):
        """Return the object with the given fields and values. Return None if not found."""

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        result = await session.exec(statement)
        return result.first()

    @classmethod
    async def all_by_field(cls, session: AsyncSession, field: str, value: Any):
        """
        Return all objects with the given field and value.
        Return an empty list if not found.
        """
        statement = select(cls).where(getattr(cls, field) == value)
        result = await session.exec(statement)
        return result.all()

    @classmethod
    async def all_by_fields(
        cls,
        session: AsyncSession,
        fields: dict = {},
        extra_conditions: Optional[List] = None,
    ):
        """
        Return all objects with the given fields and values.
        Return an empty list if not found.
        """

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        if extra_conditions:
            statement = statement.where(and_(*extra_conditions))

        result = await session.exec(statement)
        return result.all()

    @classmethod
    async def paginated_by_query(
        cls,
        session: AsyncSession,
        fields: Optional[dict] = None,
        fuzzy_fields: Optional[dict] = None,
        extra_conditions: Optional[List] = None,
        page: int = 1,
        per_page: int = 100,
        order_by: Optional[List[Tuple[str, str]]] = None,
    ) -> PaginatedList[SQLModel]:
        """
        Return a paginated and optionally sorted list of objects matching the given query criteria.

        Args:
            session (AsyncSession): The SQLAlchemy async session used to interact with the database.
            fields (Optional[dict]): Exact match filters as key-value pairs.
            fuzzy_fields (Optional[dict]): Fuzzy match filters using the SQL `LIKE` operator.
            page (int): Page number for pagination, starting from 1. Default is 1.
            per_page (int): Number of items per page. Default is 100.
            order_by (Optional[List[Tuple[str, str]]]): Sorting criteria as a list of tuples,
                each containing a field name and sort direction ("asc" or "desc").
                If not provided, defaults to `created_at DESC`.

        Returns:
            PaginatedList[SQLModel]: A paginated list of matching objects with pagination metadata.
        """

        statement = select(cls)
        if fields:
            conditions = [
                col(getattr(cls, key)) == value for key, value in fields.items()
            ]
            statement = statement.where(and_(*conditions))

        if fuzzy_fields:
            fuzzy_conditions = [
                func.lower(getattr(cls, key)).like(f"%{str(value).lower()}%")
                for key, value in fuzzy_fields.items()
            ]
            statement = statement.where(or_(*fuzzy_conditions))

        if extra_conditions:
            statement = statement.where(and_(*extra_conditions))

        if not order_by:
            order_by = [("created_at", "desc")]

        for field, direction in order_by:
            column = col(getattr(cls, field))
            statement = statement.order_by(
                asc(column) if direction.lower() == "asc" else desc(column)
            )

        if page is not None and per_page is not None:
            statement = statement.offset((page - 1) * per_page).limit(per_page)
        items = (await session.exec(statement)).all()

        count_statement = select(func.count(cls.id))
        if fields:
            conditions = [
                col(getattr(cls, key)) == value for key, value in fields.items()
            ]
            count_statement = count_statement.where(and_(*conditions))

        if fuzzy_fields:
            fuzzy_conditions = [
                col(getattr(cls, key)).like(f"%{value}%")
                for key, value in fuzzy_fields.items()
            ]
            count_statement = count_statement.where(or_(*fuzzy_conditions))

        if extra_conditions:
            count_statement = count_statement.where(and_(*extra_conditions))

        count = (await session.exec(count_statement)).one()
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
        cls, source: Union[dict, SQLModel], update: Optional[dict] = None
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
        cls,
        session: AsyncSession,
        source: Union[dict, SQLModel],
        update: Optional[dict] = None,
        auto_commit: bool = True,
    ) -> Optional[SQLModel]:
        """Create and save a new record for the model."""

        obj = cls.convert_without_saving(source, update)
        if obj is None:
            return None

        cls._publish_event_after_commit(session, EventType.CREATED, obj)
        await obj.save(session, auto_commit=auto_commit)
        return obj

    @classmethod
    async def create_or_update(
        cls,
        session: AsyncSession,
        source: Union[dict, SQLModel],
        update: Optional[dict] = None,
        auto_commit: bool = True,
    ) -> Optional[SQLModel]:
        """Create or update a record for the model."""

        obj = cls.convert_without_saving(source, update)
        if obj is None:
            return None
        pk = cls.__mapper__.primary_key_from_instance(obj)
        if pk[0] is not None:
            existing = await session.get(cls, pk)
            if existing is None:
                return None
            else:
                await existing.update(session, obj, auto_commit=auto_commit)
                return existing
        else:
            return await cls.create(session, obj, auto_commit=auto_commit)

    @classmethod
    async def count(cls, session: AsyncSession) -> int:
        """Return the number of records in the model."""

        return len(await cls.all(session))

    async def refresh(self, session: AsyncSession):
        """Refresh the object from the database."""

        await session.refresh(self)

    async def save(self, session: AsyncSession, auto_commit=True):
        """Save the object to the database. Raise exception if failed."""

        session.add(self)
        await session.flush()

        if not auto_commit:
            return

        try:
            await session.commit()
            await session.refresh(self)
        except (IntegrityError, OperationalError, FlushError) as e:
            await session.rollback()
            raise e
        except Exception as e:
            await session.rollback()
            raise e

    async def update(
        self,
        session: AsyncSession,
        source: Union[dict, SQLModel, None] = None,
        auto_commit=True,
    ):
        """Update the object with the source and save to the database."""

        if isinstance(source, SQLModel):
            source = source.model_dump(exclude_unset=True)
        elif source is None:
            source = {}

        for key, value in source.items():
            setattr(self, key, value)
        self._publish_event_after_commit(session, EventType.UPDATED, self)
        await self.save(session, auto_commit=auto_commit)

    async def delete(self, session: AsyncSession, soft=False, auto_commit=True):
        """Delete the object from the database."""
        self._publish_event_after_commit(session, EventType.DELETED, self)

        if soft or self._has_cascade_delete():
            if hasattr(self, "deleted_at"):
                # timestamp is stored without timezone in db
                self.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
                await self.save(session, auto_commit=auto_commit)
            await self._handle_cascade_delete(
                session, soft=soft, auto_commit=auto_commit
            )
        if not soft:
            await session.delete(self)
        if not auto_commit:
            return
        await session.commit()

    async def _handle_cascade_delete(
        self, session: AsyncSession, soft=False, auto_commit=True
    ):
        """Handle cascading deletes for all defined relationships."""
        for rel in self.__mapper__.relationships:
            if rel.cascade.delete:
                # Load the related objects
                await session.refresh(self)
                related_objects = getattr(self, rel.key)

                # Delete each related object
                if isinstance(related_objects, list):
                    for related_object in related_objects:
                        await related_object.delete(
                            session, soft=soft, auto_commit=auto_commit
                        )
                elif related_objects:
                    await related_objects.delete(
                        session, soft=soft, auto_commit=auto_commit
                    )

    def _has_cascade_delete(self):
        """Check if the model has cascade delete relationships."""
        return any(rel.cascade.delete for rel in self.__mapper__.relationships)

    @classmethod
    async def all(cls, session: AsyncSession):
        """Return all objects of the model."""

        result = await session.exec(select(cls))
        return result.all()

    @classmethod
    async def delete_all(cls, session: AsyncSession, soft=False):
        """Delete all objects of the model."""

        for obj in await cls.all(session):
            cls._publish_event_after_commit(session, EventType.DELETED, obj)
            await obj.delete(session, soft=soft, auto_commit=False)
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete all objects of {cls.__name__}: {e}")

    @classmethod
    def _publish_event_after_commit(
        cls, session: AsyncSession, event_type: str, data: Any
    ):
        session.info.setdefault("pending_events", []).append(
            CommitEvent(
                name=cls.__name__.lower(),
                type=event_type,
                data=data,
            )
        )

    @classmethod
    async def subscribe(cls, engine: AsyncEngine) -> AsyncGenerator[Event, None]:
        subscriber = event_bus.subscribe(cls.__name__.lower())
        async with AsyncSession(engine) as session:
            items = await cls.all(session)
            for item in items:
                yield Event(type=EventType.CREATED, data=item)

        heartbeat_interval = timedelta(seconds=15)
        last_event_time = datetime.now(timezone.utc)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        subscriber.receive(), timeout=heartbeat_interval.total_seconds()
                    )
                    yield event
                except asyncio.TimeoutError:
                    if (
                        datetime.now(timezone.utc) - last_event_time
                        >= heartbeat_interval
                    ):
                        yield Event(type=EventType.HEARTBEAT, data=None)
                        last_event_time = datetime.now(timezone.utc)
        finally:
            event_bus.unsubscribe(cls.__name__.lower(), subscriber)

    @classmethod
    async def streaming(
        cls,
        engine: AsyncEngine,
        fields: Optional[dict] = None,
        fuzzy_fields: Optional[dict] = None,
        filter_func: Optional[Callable[[Any], bool]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream events matching the given criteria as JSON strings."""
        try:
            async for event in cls.subscribe(engine):
                if event.type == EventType.HEARTBEAT:
                    yield "\n\n"
                    continue

                if not cls._match_fields(event, fields):
                    continue

                if not cls._match_fuzzy_fields(event, fuzzy_fields):
                    continue

                if filter_func and not filter_func(event.data):
                    continue

                event.data = cls._convert_to_public_class(event.data)
                yield cls._format_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in streaming {cls.__name__}: {e}")

    @classmethod
    def _match_fields(cls, event: Any, fields: Optional[dict]) -> bool:
        """Match fields using AND condition."""
        for key, value in (fields or {}).items():
            if getattr(event.data, key, None) != value:
                return False
        return True

    @classmethod
    def _match_fuzzy_fields(cls, event: Any, fuzzy_fields: Optional[dict]) -> bool:
        """Match fuzzy fields using OR condition."""
        for key, value in (fuzzy_fields or {}).items():
            attr_value = str(getattr(event.data, key, "")).lower()
            if str(value).lower() in attr_value:
                return True
        return not fuzzy_fields

    @classmethod
    def _convert_to_public_class(cls, data: Any) -> Any:
        """Convert the instance to the corresponding Public class if it exists."""
        class_module = importlib.import_module(cls.__module__)
        public_class = getattr(class_module, f"{cls.__name__}Public", None)
        return public_class.model_validate(data) if public_class else data

    @staticmethod
    def _format_event(event: Any) -> str:
        """Format the event as a JSON string."""
        return json.dumps(jsonable_encoder(event), separators=(",", ":")) + "\n\n"
