from datetime import datetime
from typing import Optional

from sqlalchemy import Column
from sqlmodel import Field, SQLModel
from gpustack.mixins.active_record import ActiveRecordMixin
from gpustack.schemas.common import pydantic_column_type
from gpustack.schemas.workers import UtilizationInfo


class SystemLoad(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'system_loads'
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    cpu: Optional[UtilizationInfo] = Field(
        sa_column=Column(pydantic_column_type(UtilizationInfo))
    )
    memory: Optional[UtilizationInfo] = Field(
        sa_column=Column(pydantic_column_type(UtilizationInfo))
    )
    gpu: Optional[UtilizationInfo] = Field(
        sa_column=Column(pydantic_column_type(UtilizationInfo))
    )
    gpu_memory: Optional[UtilizationInfo] = Field(
        sa_column=Column(pydantic_column_type(UtilizationInfo))
    )
