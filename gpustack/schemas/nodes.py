from pydantic import BaseModel

from .common import PaginatedList


class ResourceSummary(BaseModel):
    total: float | None = None
    used: float | None = None
    free: float | None = None
    percent: float | None = None


class Node(BaseModel):
    id: str
    name: str
    hostname: str
    address: str
    alive: bool
    resources: dict[str, ResourceSummary] = {}
    labels: dict[str, str] = {}


NodePublic = Node
NodesPublic = PaginatedList[Node]
