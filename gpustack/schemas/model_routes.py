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
from gpustack.schemas.principals import _platform_principal_id

if TYPE_CHECKING:
    from gpustack.schemas.models import Model
    from gpustack.schemas.model_provider import ModelProvider


# Route names intentionally exclude `/` — the dispatch parser
# (`ModelRouteService.resolve_route_targets`) splits the inbound
# `model` string on the first `/` to separate the owner's ``name``
# (k8s-style identifier) from the raw route name. Allowing `/` inside
# route names would create irresolvable ambiguity (e.g. literal route
# "a/b" in platform Org vs. route "b" in Org with ``name="a"``). `:`
# IS allowed — LoRA child routes use the form `<base>:<lora>`, which
# is unambiguous since the owner-prefix split happens before `:` is
# ever consulted.
name_pattern = r'^[A-Za-z](?:[A-Za-z0-9_\-\.:]*[A-Za-z0-9])?$'


def effective_route_name(
    route_name: str,
    owner_name: Optional[str],
    is_platform_org: bool,
) -> str:
    """The model name clients see and gateways route on.

    The platform Org keeps unprefixed names (backward compat — existing
    clients calling `model: "qwen3-0.6b"` keep working). Other Orgs get
    an owner-name prefix (`org1/qwen3-0.6b`) so two Orgs can use the
    same route name without colliding in Higress's AI proxy match
    rules.

    Format follows the OpenAI / HuggingFace / OpenRouter convention
    (`namespace/model`); the owner's ``name`` column is constrained to
    `^[a-z](?:[a-z0-9\\-]*[a-z0-9])?$` and route names exclude `/` (see
    ``name_pattern``) so the joined string parses unambiguously.
    """
    if is_platform_org or not owner_name:
        return route_name
    return f"{owner_name}/{route_name}"


class AccessPolicyEnum(str, Enum):
    PUBLIC = "public"
    AUTHED = "authed"
    # ORG = scoped to members of the route's owning Organization. The
    # default for new routes in non-platform Orgs — semantically the
    # "team-private" scope, no principal table involvement.
    ORG = "org"
    # Per-user grants. Rows are stored in ``model_route_principals``
    # with ``principal_id`` pointing at a USER-kind principal.
    ALLOWED_USERS = "allowed_users"
    # Per-principal grants (user / org / group) via
    # ``model_route_principals``. Mutually exclusive with
    # ``ALLOWED_USERS`` — pick the policy whose granularity matches
    # the deployment's identity model.
    ALLOWED_PRINCIPALS = "allowed_principals"


class TargetStateEnum(str, Enum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"


class FallbackStatusEnum(str, Enum):
    ERROR_400 = "4xx"
    ERROR_500 = "5xx"


class ModelRouteTargetUpdate(SQLModel):
    overridden_model_name: Optional[str] = Field(default=None, nullable=True)
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

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_provider_model_name(cls, data):
        # Accept the pre-v2.2.0 ``provider_model_name`` key as a
        # deprecated alias for ``overridden_model_name``. Conflicting
        # values on both keys are rejected to avoid silent picking.
        if not isinstance(data, dict) or "provider_model_name" not in data:
            return data
        legacy = data.pop("provider_model_name")
        if "overridden_model_name" in data:
            canonical = data["overridden_model_name"]
            if canonical != legacy:
                raise ValueError(
                    "Got both 'overridden_model_name' and the legacy alias "
                    "'provider_model_name' with different values. Drop "
                    "'provider_model_name' (deprecated) and send only "
                    "'overridden_model_name'."
                )
        else:
            data["overridden_model_name"] = legacy
        return data

    @field_validator("overridden_model_name", mode="before")
    def validate_overridden_model_name(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("overridden_model_name must be a string")
        if not re.match(name_pattern, v):
            raise ValueError(
                "overridden_model_name must start with a letter and contain only "
                "letters, numbers, hyphens, underscores, dots, or colons"
            )
        return v

    @model_validator(mode="after")
    def check_provider_or_model(self):
        both_set = self.provider_id is not None and self.model_id is not None
        both_none = self.provider_id is None and self.model_id is None

        if both_none:
            raise ValueError("Either provider_id or model_id must be provided.")
        if both_set:
            raise ValueError("Only one of provider_id or model_id can be provided.")
        if self.provider_id is not None and self.overridden_model_name is None:
            raise ValueError(
                "overridden_model_name must be provided when provider_id is set."
            )
        # Local-model targets only accept overridden_model_name shaped like
        # "<base_model_name>:<lora_name>". The full base-prefix check lives in
        # the service layer (lora_route_name_for); schema only enforces shape.
        if (
            self.model_id is not None
            and self.overridden_model_name is not None
            and ":" not in self.overridden_model_name
        ):
            raise ValueError(
                "overridden_model_name for a local-model target must be of the "
                "form '<base_model_name>:<lora_name>'."
            )
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
                "route_name must start with a letter, only contain letters, numbers, hyphens, "
                "underscores, or dots, and not end with hyphen or underscore"
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

    @model_validator(mode="after")
    def validate_name(self):
        name = self.name
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        if not re.match(name_pattern, name):
            raise ValueError(
                "name must start with a letter and contain only letters, numbers, hyphens, "
                "underscores, or dots, and not end with hyphen or underscore"
            )
        return self


class ModelRouteUpdate(ModelRouteUpdateBase):
    targets: Optional[List[ModelRouteTargetUpdateItem]] = Field(
        default=None, nullable=True
    )


class ModelRouteCreate(ModelRouteUpdate):
    pass


class ModelRouteBase(ModelRouteUpdateBase):
    # NULL for hand-created routes.
    created_model_id: Optional[int] = Field(default=None, nullable=True)
    targets: int = Field(default=0, nullable=False, ge=0)
    ready_targets: int = Field(default=0, nullable=False, ge=0)
    access_policy: AccessPolicyEnum = Field(default=AccessPolicyEnum.AUTHED)
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        foreign_key="principals.id",
        nullable=False,
    )


class ModelRoute(ModelRouteBase, BaseModelMixin, table=True):
    __tablename__: ClassVar[str] = "model_routes"
    id: Optional[int] = Field(default=None, primary_key=True)

    route_targets: List[ModelRouteTarget] = Relationship(
        back_populates="model_route",
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
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
    # The model name clients should send in their request body. Equals
    # `name` for the platform Org (backward compat); for other Orgs it
    # is `<owner-name>/<name>`. Frontends currently derive this
    # themselves via `effectiveRouteName(name, org)` since they have
    # the owning Org row in hand from a separate fetch — the field is
    # reserved here so a future server-side enrichment can populate it
    # without breaking consumers.
    effective_name: Optional[str] = None


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


class ModelAuthorizationList(ItemList[ModelUserAccessExtended]):
    # The route's current access_policy is returned alongside the
    # grants list so a single GET refreshes both halves of the
    # Access Settings dialog (some clients open it from a stale list
    # snapshot where the row's policy may not reflect the latest
    # save).
    access_policy: Optional[AccessPolicyEnum] = None


class MyModel(ModelRouteBase, BaseModelMixin, table=True):
    __tablename__ = 'non_admin_user_models'
    __mapper_args__ = {'primary_key': ["pid"]}
    pid: str
    id: int
    user_id: int = Field(default=0)
    # Records which principal granted this (user, route) row visibility.
    # NULL on PUBLIC/AUTHED rows (not principal-mediated). The kind is
    # stored as plain text — see ``model_user_after_create_view_stmt``
    # for why it isn't the ``principaltype`` enum. Server-side only;
    # not surfaced via ``MyModelPublic``.
    via_principal_id: Optional[int] = None
    via_principal_kind: Optional[str] = None


class MyModelPublic(ModelRoutePublic):
    pass
