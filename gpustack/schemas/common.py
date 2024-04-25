from typing import Generic, TypeVar

from fastapi import Query
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class Pagination(BaseModel):
    page: int
    perPage: int
    total: int
    totalPage: int


class ListParams(BaseModel):
    query: str | None = None
    page: int = Query(default=1, ge=1)
    perPage: int = Query(default=100, ge=1, le=100)


class PaginatedList(BaseModel, Generic[T]):
    items: list[T]
    pagination: Pagination
