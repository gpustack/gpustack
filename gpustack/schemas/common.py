import json
from typing import Generic, Optional, Type, TypeVar

from fastapi import Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, TypeAdapter
from sqlalchemy import JSON as SQLAlchemyJSON, TypeDecorator

T = TypeVar("T", bound=BaseModel)


class Pagination(BaseModel):
    page: int
    perPage: int
    total: int
    totalPage: int


class ListParams(BaseModel):
    query: Optional[str] = None
    page: int = Query(default=1, ge=1)
    perPage: int = Query(default=100, ge=1, le=100)
    watch: bool = Query(default=False)


class PaginatedList(BaseModel, Generic[T]):
    items: list[T]
    pagination: Pagination


class JSON(SQLAlchemyJSON):
    pass


def pydantic_column_type(pydantic_type: Type[T]):  # noqa: C901
    class PydanticJSONType(TypeDecorator, Generic[T]):
        impl = JSON()

        def __init__(self, json_encoder=json):
            self.json_encoder = json_encoder
            super(PydanticJSONType, self).__init__()

        def bind_processor(self, dialect):
            impl_processor = self.impl.bind_processor(dialect)
            dumps = self.json_encoder.dumps

            def process(value: T):
                if value is not None:
                    value_to_dump = self._prepare_value_for_dump(value)
                    value = jsonable_encoder(value_to_dump)
                return (
                    impl_processor(value)
                    if impl_processor
                    else dumps(jsonable_encoder(value_to_dump))
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
