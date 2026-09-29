"""Saved deployment configuration, independent of runtime state."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import Column, ForeignKey, Integer, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.schemas.common import PaginatedList, UTCDateTime
from gpustack.schemas.deployment_document import DeploymentChange


class ModelRevision(SQLModel, table=True):
    __tablename__ = "model_revisions"
    __table_args__ = (UniqueConstraint("model_id", "revision"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
        )
    )
    revision: int
    spec: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(UTCDateTime, nullable=False),
    )
    created_by: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("principals.id", ondelete="SET NULL")),
    )


class ModelRevisionSummary(BaseModel):
    id: int
    model_id: int
    revision: int
    created_at: datetime
    created_by: Optional[int] = None


class ModelRevisionPublic(ModelRevisionSummary):
    spec: Dict[str, Any]


ModelRevisionsPublic = PaginatedList[ModelRevisionSummary]


class ModelRollbackRequest(BaseModel):
    target_revision: int = Field(gt=0)


class ModelRollbackPreview(BaseModel):
    current_revision: int
    target_revision: int
    current: Dict[str, Any]
    desired: Dict[str, Any]
    changes: List[DeploymentChange]
    changed: bool
