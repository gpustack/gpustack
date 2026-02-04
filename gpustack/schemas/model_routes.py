import re
from enum import Enum
from typing import ClassVar, Optional, Dict, Any, List, Set
from pydantic import BaseModel, field_validator, model_validator
from sqlmodel import (
    Field,
    Relationship,
    Column,
    SQLModel,
    Integer,
    ForeignKey,
    JSON,
)

from typing import TYPE_CHECKING
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    PublicFields,
    ItemList,
)
from gpustack.schemas.links import UserModelRouteLink

if TYPE_CHECKING:
    from gpustack.schemas.users import User
    from gpustack.schemas.models import Model
    from gpustack.schemas.model_provider import ModelProvider


name_pattern = r'^[A-Za-z][A-Za-z0-9_\-\.]*[A-Za-z0-9]$'


class AccessPolicyEnum(str, Enum):
    PUBLIC = "public"
    AUTHED = "authed"
    ALLOWED_USERS = "allowed_users"


class TargetStateEnum(str, Enum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"


class FallbackStatusEnum(str, Enum):
    ERROR_400 = "4xx"
    ERROR_500 = "5xx"


class ModelRouteTargetUpdate(SQLModel):
    provider_model_name: Optional[str] = Field(default=None, nullable=True)
    weight: int = Field(default=0, nullable=False, ge=0)
    model_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey(
                "models.id",
                ondelete="CASCADE",
            ),
            nullable=True,
        ),
    )
    provider_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey(
                "model_providers.id",
                ondelete="CASCADE",
            ),
            nullable=True,
        ),
    )

    @model_validator(mode="after")
    def check_provider_or_model(self):
        both_set = self.provider_id is not None and self.model_id is not None
        both_none = self.provider_id is None and self.model_id is None
        name_missing = self.provider_model_name is None and self.provider_id is not None
        invalid_name = (
            self.provider_model_name is not None and self.model_id is not None
        )

        if both_none:
            raise ValueError("Either provider_id or model_id must be provided.")
        if both_set:
            raise ValueError("Only one of provider_id or model_id can be provided.")
        if name_missing:
            raise ValueError(
                "provider_model_name must be provided when provider_id is set."
            )
        if invalid_name:
            raise ValueError("provider_model_name must be None when model_id is set.")
        return self


class ModelRouteTargetCreate(ModelRouteTargetUpdate):
    fallback_status_codes: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(
            JSON,
            nullable=True,
        ),
    )

    @field_validator("fallback_status_codes", mode="before")
    def validate_fallback_status_codes(cls, v):
        if v is None:
            return v
        deduped: Set[str] = set(v)
        for status in deduped:
            if status not in [
                FallbackStatusEnum.ERROR_400,
                FallbackStatusEnum.ERROR_500,
            ]:
                raise ValueError(f"Invalid fallback status code: {status}")
        return list(deduped)


class ModelRouteTargetBase(ModelRouteTargetCreate):
    name: str = Field(nullable=False)
    route_name: str = Field(nullable=False)
    route_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey(
                "model_routes.id",
                ondelete="CASCADE",
            ),
            nullable=False,
        )
    )
    state: TargetStateEnum = Field(default=TargetStateEnum.ACTIVE, nullable=False)

    @field_validator("route_name", mode="before")
    def validate_route_name(cls, v):
        if not isinstance(v, str):
            raise ValueError("route_name must be a string")
        if not re.match(name_pattern, v):
            raise ValueError(
                "route_name must start with a letter, only contain letters, numbers, hyphens, underscores, and not end with hyphen or underscore"
            )
        return v


class ModelRouteTarget(ModelRouteTargetBase, BaseModelMixin, table=True):
    __tablename__: ClassVar[str] = "model_route_targets"
    id: Optional[int] = Field(default=None, primary_key=True)
    model_route: "ModelRoute" = Relationship(
        back_populates="route_targets",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    provider: Optional["ModelProvider"] = Relationship(
        back_populates="model_route_targets",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    model: Optional["Model"] = Relationship(
        back_populates="model_route_targets",
        sa_relationship_kwargs={"lazy": "noload"},
    )


class ModelRouteTargetPublic(ModelRouteTargetBase, PublicFields):
    pass


ModelRouteTargetsPublic = PaginatedList[ModelRouteTargetPublic]


class ModelRouteTargetListParams(ListParams):
    route_id: Optional[int] = None
    route_name: Optional[str] = None
    model_id: Optional[int] = None
    provider_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "created_at",
        "updated_at",
        "name",
        "weight",
        "state",
    ]


class ModelRouteTargetUpdateItem(ModelRouteTargetCreate):
    id: Optional[int] = None


class ModelRouteUpdateBase(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(default=None, nullable=True)
    categories: List[str] = Field(sa_type=JSON, default=[])
    meta: Optional[Dict[str, Any]] = Field(sa_type=JSON, default={})
    generic_proxy: Optional[bool] = Field(default=False)

    @field_validator("categories", mode="before")
    def validate_categories(cls, v):
        if v is None:
            return v
        for category in v:
            if category not in [
                "llm",
                "embedding",
                "image",
                "reranker",
                "speech_to_text",
                "text_to_speech",
                "unknown",
            ]:
                raise ValueError(f"Invalid category: {category}")
        return v

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        if not isinstance(v, str):
            raise ValueError("name must be a string")
        if not re.match(name_pattern, v):
            raise ValueError(
                "name must start with a letter, only contain letters, numbers, hyphens, underscores, and not end with hyphen or underscore"
            )
        return v


class ModelRouteUpdate(ModelRouteUpdateBase):
    targets: Optional[List[ModelRouteTargetUpdateItem]] = Field(
        default=None, nullable=True
    )


class ModelRouteCreate(ModelRouteUpdate):
    pass


class ModelRouteBase(ModelRouteUpdateBase):
    created_by_model: Optional[bool] = Field(default=False, nullable=False)
    targets: int = Field(default=0, nullable=False, ge=0)
    ready_targets: int = Field(default=0, nullable=False, ge=0)
    access_policy: AccessPolicyEnum = Field(default=AccessPolicyEnum.AUTHED)


class ModelRoute(ModelRouteBase, BaseModelMixin, table=True):
    __tablename__: ClassVar[str] = "model_routes"
    id: Optional[int] = Field(default=None, primary_key=True)

    route_targets: List[ModelRouteTarget] = Relationship(
        back_populates="model_route",
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
    )

    users: List["User"] = Relationship(
        back_populates="routes",
        link_model=UserModelRouteLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    models: List["Model"] = Relationship(
        back_populates="model_routes",
        link_model=ModelRouteTarget,
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "route_targets,model_route,model",
        },
    )


class ModelRoutePublic(ModelRouteBase, PublicFields):
    pass


ModelRoutesPublic = PaginatedList[ModelRoutePublic]


class ModelRouteListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "created_at",
        "updated_at",
        "name",
        "targets",
        "ready_targets",
    ]


class SetFallbackTargetInput(BaseModel):
    fallback_status_codes: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(
            JSON,
            nullable=True,
        ),
    )

    @field_validator("fallback_status_codes", mode="before")
    def validate_fallback_status_codes(cls, v):
        if v is None:
            return v
        deduped: Set[FallbackStatusEnum] = set(v)
        for status in deduped:
            if status not in [
                FallbackStatusEnum.ERROR_400,
                FallbackStatusEnum.ERROR_500,
            ]:
                raise ValueError(f"Invalid fallback status code: {status}")
        return list(deduped)


class ModelUserAccess(BaseModel):
    id: int
    # More custom fields can be added here, e.g., quota, rate_limit, etc.


class ModelAuthorizationUpdate(BaseModel):
    access_policy: Optional[AccessPolicyEnum] = None
    users: List[ModelUserAccess]


class ModelUserAccessExtended(ModelUserAccess):
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    # More user fields can be added here. e.g. quota, rate_limit, etc.


ModelAuthorizationList = ItemList[ModelUserAccessExtended]


class MyModel(ModelRouteBase, BaseModelMixin, table=True):
    __tablename__ = 'non_admin_user_models'
    __mapper_args__ = {'primary_key': ["pid"]}
    pid: str
    id: int
    user_id: int = Field(default=0)


class MyModelPublic(ModelRoutePublic):
    pass
