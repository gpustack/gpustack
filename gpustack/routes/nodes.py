from fastapi import APIRouter, HTTPException

from ..core.runtime.ray import RayRuntime
from ..core.deps import ListParamsDep
from ..schemas.common import Pagination
from ..schemas.nodes import NodePublic, NodesPublic

router = APIRouter()


@router.get("", response_model=NodesPublic)
async def get_nodes(params: ListParamsDep):
    nodes = RayRuntime().get_nodes()
    total_items = len(nodes)
    total_page = (total_items + params.perPage - 1) // params.perPage
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=total_items,
        totalPage=total_page,
    )
    return NodesPublic(items=nodes, pagination=pagination)


@router.get("/{id}", response_model=NodePublic)
async def get_node(id: str):
    nodes = RayRuntime().get_nodes()
    for node in nodes:
        if node.id == id:
            return node

    raise HTTPException(status_code=404, detail="Node not found")
