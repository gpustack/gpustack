from datetime import datetime
from typing import ClassVar, List, Optional

from sqlalchemy import BigInteger
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    UniqueConstraint,
)

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList


class TenantQuotaUpdate(SQLModel):
    gpu: Optional[int] = Field(default=None, ge=0)
    cpu_milli: Optional[int] = Field(default=None, ge=0)
    memory_bytes: Optional[int] = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )
    gpu_instance: Optional[int] = Field(default=None, ge=0)


class TenantQuotaBase(TenantQuotaUpdate):
    cluster_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    owner_principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )


class TenantQuota(TenantQuotaBase, BaseModelMixin, table=True):
    __tablename__ = 'tenant_quotas'
    __table_args__ = (
        UniqueConstraint(
            'cluster_id', 'owner_principal_id', name='uix_tenant_quota_cluster_org'
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)


class TenantQuotaListParams(ListParams):
    cluster_id: Optional[int] = None
    owner_principal_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "cluster_id",
        "owner_principal_id",
        "created_at",
        "updated_at",
    ]


class TenantQuotaPublic(TenantQuotaBase):
    id: int
    created_at: datetime
    updated_at: datetime


TenantQuotasPublic = PaginatedList[TenantQuotaPublic]
