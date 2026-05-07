from datetime import datetime
from typing import Optional

from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    UniqueConstraint,
)

from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.principals import PrincipalType


class ClusterAccess(SQLModel, table=True):
    """Grant a single principal access to a cluster.

    Polymorphic ``(principal_type, principal_id)`` was collapsed into a
    single ``principal_id`` FK pointing at ``principals``; the kind is
    available via the joined principals row when callers need to render
    or branch on it.
    """

    __tablename__ = 'cluster_access'
    __table_args__ = (
        UniqueConstraint(
            'cluster_id', 'principal_id', name='uix_cluster_access_cluster_principal'
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    cluster_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    granted_by: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(UTCDateTime, nullable=False),
    )


class ClusterAccessPublic(SQLModel):
    cluster_id: int
    principal_id: int
    # Discriminator kept on the public payload so the UI can render
    # "User Alice" vs "Org Acme" vs "Group Engineers" without doing a
    # second principals lookup. Resolved server-side from the principals
    # row.
    principal_type: PrincipalType
    # Human-readable label for the principal — username for USER,
    # name for ORG / GROUP. Same rationale: server has the join cheaply.
    principal_name: Optional[str] = None
    # For GROUP principals, the parent ORG. NULL for USER and ORG.
    principal_parent_id: Optional[int] = None
    granted_by: Optional[int] = None
    created_at: datetime
