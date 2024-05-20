from fastapi import APIRouter, HTTPException

from ..core.deps import ListParamsDep, SessionDep
from ..schemas.nodes import NodeCreate, NodePublic, NodeUpdate, NodesPublic, Node

router = APIRouter()


@router.get("", response_model=NodesPublic)
async def get_nodes(session: SessionDep, params: ListParamsDep):
    fields = {}
    if params.query:
        fields = {"name": params.query}
    return Node.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=NodePublic)
async def get_node(session: SessionDep, id: int):
    node = Node.one_by_id(session, id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.post("", response_model=NodePublic)
async def create_node(session: SessionDep, node_in: NodeCreate):
    node = Node.model_validate(node_in)

    return node.save(session)


@router.put("/{id}", response_model=NodePublic)
async def update_node(session: SessionDep, node_in: NodeUpdate):
    node = Node.model_validate(node_in)
    return node.save(session)


@router.delete("/{id}")
async def delete_node(session: SessionDep, id: int):
    node = Node.one_by_id(session, id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    return node.delete(session)
