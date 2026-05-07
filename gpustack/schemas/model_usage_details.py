from datetime import date, datetime
from typing import ClassVar, Optional

from pydantic import ConfigDict
from sqlalchemy import BigInteger, Column, Integer
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.model_usage import OperationEnum


class ModelUsageDetails(SQLModel, BaseModelMixin, table=True):
    """
    Per-request inference usage audit row.

    Reference id columns (``user_id`` / ``model_id`` / ``model_route_id`` /
    ``provider_id`` / ``cluster_id`` / ``api_key_id``) are plain integers,
    not foreign keys. Audit rows must outlive the entities they describe;
    losing the historical id (which ``SET NULL`` would do on parent delete)
    is a worse audit outcome than losing the live join, so ids stay as
    reported and ``*_name`` columns hold mutable display snapshots
    alongside.

    Relationship to ``ModelUsage``: details and rollup are NOT 1:1 — the
    rollup aggregates many requests per (model, user, key, operation, day)
    into one row, while details preserves every report. They are populated
    from the same ingest path but serve different read patterns:
        * ``ModelUsage``    — dashboard / per-day analytics, FK-friendly
        * ``ModelUsageDetails`` — quota reconciliation / per-request audit,
                                  FK-less so historical ids survive deletes

    Both rows are constructed with the same ``build_model_usage_snapshot``
    keys plus table-specific extras; see that helper's docstring for the
    shared-snapshot contract.
    """

    __tablename__: ClassVar[str] = "model_usage_details"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_name: str = Field(default=...)
    model_route_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_route_name: Optional[str] = Field(default=None)
    provider_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    prompt_cached_token_count: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    operation: Optional[OperationEnum] = Field(default=None)
    # Wall-clock anchors reported by the proxy (UnixMilli on the wire,
    # stored as naive UTC). Distinct from ``created_at`` so quota
    # reconciliation / cache rebuild can key off the request's actual
    # completion time even after rows are archived or migrated.
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())


class ModelUsageDetailsArchive(SQLModel, BaseModelMixin, table=True):
    """
    Cold-storage archive for ``model_usage_details``.

    Same column layout as the hot table; ``id`` is a plain primary key with
    no sequence/autoincrement — rows are archived from
    ``model_usage_details`` and reuse the source ``id``.
    """

    __tablename__: ClassVar[str] = "model_usage_details_archive"
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_name: str = Field(default=...)
    model_route_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_route_name: Optional[str] = Field(default=None)
    provider_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    prompt_cached_token_count: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    operation: Optional[OperationEnum] = Field(default=None)
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())
