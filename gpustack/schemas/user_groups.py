"""Pydantic surface for the User Group API.

Backed by ``principals`` rows where ``kind == GROUP`` and
``parent_principal_id`` points at the owning ORG-principal. There is
no dedicated ``user_groups`` table — UserGroup is a Pydantic-only DTO
over ``Principal``. Group membership lives in
``principal_memberships``.
"""

from datetime import datetime
from typing import ClassVar, List, Optional

from sqlmodel import Field, SQLModel

from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.principals import Principal


class UserGroupUpdate(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(default=None, nullable=True)


class UserGroupCreate(UserGroupUpdate):
    pass


class UserGroupListParams(ListParams):
    organization_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "created_at",
        "updated_at",
    ]


class UserGroupPublic(SQLModel):
    id: int
    organization_id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_principal(cls, p: Principal) -> "UserGroupPublic":
        return cls(
            id=p.id,
            organization_id=p.parent_principal_id,
            name=p.name,
            description=p.description,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


UserGroupsPublic = PaginatedList[UserGroupPublic]


class UserGroupMembershipPublic(SQLModel):
    user_id: int
    group_id: int
    created_at: datetime
    username: Optional[str] = None
    full_name: Optional[str] = None
