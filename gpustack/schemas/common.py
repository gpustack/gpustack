from datetime import timezone
import json
from typing import ClassVar, Generic, List, Optional, Tuple, Type, TypeVar

from fastapi import Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, TypeAdapter, computed_field, field_validator
import sqlalchemy as sa
from sqlalchemy import JSON as SQLAlchemyJSON, TypeDecorator

from gpustack.api.exceptions import InvalidException

T = TypeVar("T", bound=BaseModel)


class Pagination(BaseModel):
    page: int
    perPage: int
    total: int
    totalPage: int


class ListParams(BaseModel):
    page: int = Query(default=1)
    # FIXME It uses camelCase but most APIs use snake_case. We might want to migrate to snake_case later.
    perPage: int = Query(default=100)
    watch: bool = Query(default=False)
    sort_by: Optional[str] = Query(
        default=None,
        description="Sorting in the format: field1,-field2,field3. A leading '-' indicates descending order.",
    )

    sortable_fields: ClassVar[List[str]] = []

    @field_validator('sort_by')
    def validate_sort_by(cls, v: Optional[str]) -> Optional[str]:
        """Validates the sort_by string format."""
        if not v:
            return v

        if not cls.sortable_fields:
            return v

        for field in v.split(','):
            field = field.strip()
            if not field:
                continue

            field_name = field[1:] if field.startswith('-') else field

            # Verify if the field is in the allowed sortable fields
            if field_name not in cls.sortable_fields:
                raise InvalidException(
                    f"Field '{field_name}' is not sortable. "
                    f"Allowed fields: {', '.join(cls.sortable_fields)}"
                )

        return v

    @computed_field
    @property
    def order_by(self) -> Optional[List[Tuple[str, str]]]:
        """
        Parses the sort_by string into a list of (field, direction) tuples.

        For example, "name,-created_at,status" will be parsed to:
        [("name", "asc"), ("created_at", "desc"), ("status", "asc")]

        Returns None if sort_by is not set.
        """
        if self.sort_by is None:
            return None

        order_by = []
        for field in self.sort_by.split(','):
            field = field.strip()
            if not field:
                continue

            if field.startswith('-'):
                direction = "desc"
                field_name = field[1:]
            else:
                direction = "asc"
                field_name = field

            order_by.append((field_name, direction))

        return order_by


class ItemList(BaseModel, Generic[T]):
    items: list[T]


class PaginatedList(ItemList[T]):
    pagination: Pagination


class JSON(SQLAlchemyJSON):
    pass


class UTCDateTime(sa.TypeDecorator):
    impl = sa.TIMESTAMP(timezone=False)

    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            # Ensure the datetime is in UTC and clear tzinfo before storing
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # Assume stored datetime is in UTC and attach tzinfo
            value = value.replace(tzinfo=timezone.utc)
        return value


def pydantic_column_type(
    pydantic_type: Type[T],
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    exclude_unset: bool = False,
):  # noqa: C901
    class PydanticJSONType(TypeDecorator, Generic[T]):
        impl = JSON()

        # https://docs.sqlalchemy.org/en/20/core/type_api.html#sqlalchemy.types.ExternalType.cache_ok
        cache_ok = True

        def __init__(self, json_encoder=json):
            self.json_encoder = json_encoder
            super(PydanticJSONType, self).__init__()

        def bind_processor(self, dialect):
            impl_processor = self.impl.bind_processor(dialect)
            dumps = self.json_encoder.dumps

            def process(value: T):
                if value is not None:
                    value_to_dump = self._prepare_value_for_dump(value)
                    value = jsonable_encoder(
                        value_to_dump,
                        exclude_defaults=exclude_defaults,
                        exclude_none=exclude_none,
                        exclude_unset=exclude_unset,
                    )
                return (
                    impl_processor(value)
                    if impl_processor
                    else dumps(
                        jsonable_encoder(
                            value_to_dump,
                            exclude_defaults=exclude_defaults,
                            exclude_none=exclude_none,
                            exclude_unset=exclude_unset,
                        )
                    )
                )

            return process

        def result_processor(self, dialect, coltype) -> T:
            impl_processor = self.impl.result_processor(dialect, coltype)

            def process(value):
                if impl_processor:
                    value = impl_processor(value)
                if value is None:
                    return None
                return TypeAdapter(pydantic_type).validate_python(value)

            return process

        def compare_values(self, x, y):
            return x == y

        def _prepare_value_for_dump(self, value):
            return pydantic_type.model_validate(value)

        def __repr__(self):
            return "JSON()"

        def __str__(self):
            return "JSON()"

    return PydanticJSONType
