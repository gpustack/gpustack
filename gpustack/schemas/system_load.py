from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel
from gpustack.mixins.active_record import ActiveRecordMixin


class SystemLoad(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'system_loads'
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    # average cpu utilization rate per worker
    cpu: Optional[float] = Field(default=None)

    # average memory utilization rate per worker
    memory: Optional[float] = Field(default=None)

    # average gpu utilization rate per gpu device
    gpu: Optional[float] = Field(default=None)

    # average gpu memory utilization rate per gpu device
    gpu_memory: Optional[float] = Field(default=None)
