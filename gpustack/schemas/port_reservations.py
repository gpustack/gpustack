from typing import Annotated, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class WorkerPortAvailability(BaseModel):
    worker_id: int
    generation: UUID
    claim_version: Optional[UUID] = None
    port_range: str
    unavailable_ports: List[int] = Field(default_factory=list)
    pending_cleanup: bool = False


class ModelPortClaim(BaseModel):
    token: UUID
    generations: Dict[int, UUID] = Field(default_factory=dict)
    replaces: Dict[int, UUID] = Field(default_factory=dict)
    ports: List[Annotated[int, Field(ge=1, le=65535)]] = Field(
        min_length=1, max_length=65535
    )


class PortClaimResult(BaseModel):
    accepted: bool
    unavailable_ports: List[int] = Field(default_factory=list)
