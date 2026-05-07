from typing import Optional

from pydantic import ConfigDict
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    UniqueConstraint,
)

from gpustack.mixins import BaseModelMixin


class ModelInstanceModelFileLink(SQLModel, table=True):
    model_instance_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())


class ModelInstanceDraftModelFileLink(SQLModel, table=True):
    model_instance_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())


class ModelRoutePrincipalLinkBase(SQLModel):
    route_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("model_routes.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )


class ModelRoutePrincipalLink(ModelRoutePrincipalLinkBase, BaseModelMixin, table=True):
    """Per-route principal grants for ``ALLOWED_USERS`` and
    ``ALLOWED_PRINCIPALS`` access policies.

    Each row references a single ``principals`` row — kind (USER / ORG
    / GROUP) is read from the joined principals row at evaluation
    time. This collapses the previous polymorphic three-FK design and
    matches the unified principal model used everywhere else.
    """

    __tablename__ = 'model_route_principals'
    __table_args__ = (
        UniqueConstraint('route_id', 'principal_id', name='uix_route_principal'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
