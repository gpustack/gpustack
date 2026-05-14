"""Pydantic surface for the User Group API.

Backed by ``principals`` rows where ``kind == GROUP``. Groups are
peer-level principals (no structural parent). A Group's relationship
to an Org, when one exists, is one or more rows in
``principal_memberships`` with ``parent=Org, member=Group``; those
confer the membership's role on every active user in the Group.
There is no dedicated ``user_groups`` table — UserGroup is a
Pydantic-only DTO over ``Principal``. User membership in a Group
lives in ``principal_memberships`` too (``parent=Group, member=User``,
``role=NULL``).
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
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "created_at",
        "updated_at",
    ]


class UserGroupPublic(SQLModel):
    id: int
    name: str
    description: Optional[str] = None
    # Count of active users in this group — denormalized on the
    # response so admin listings can render "Members" at a glance
    # without an N+1 fan-out to ``/groups/{id}/members``. Always
    # populated by the list/get routes (zero for empty groups).
    member_count: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_principal(
        cls, p: Principal, *, member_count: int = 0
    ) -> "UserGroupPublic":
        return cls(
            id=p.id,
            name=p.name,
            description=p.description,
            member_count=member_count,
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
