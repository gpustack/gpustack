from fastapi import APIRouter

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.nodes import NodeCreate, NodePublic, NodeUpdate, NodesPublic, Node

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
        raise NotFoundException(message="Node not found")

    return node


@router.post("", response_model=NodePublic)
async def create_node(session: SessionDep, node_in: NodeCreate):
    node = Node.model_validate(node_in)

    existing = Node.one_by_field(session, "name", node.name)
    if existing:
        raise AlreadyExistsException(message=f"Node f{node.name} already exists")

    try:
        node.save(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create node: {e}")

    return node


@router.put("/{id}", response_model=NodePublic)
async def update_node(session: SessionDep, id: int, node_in: NodeUpdate):
    node = Node.one_by_id(session, id)
    if not node:
        raise NotFoundException(message="Node not found")

    try:
        node.update(session, node_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update node: {e}")

    return node


@router.delete("/{id}")
async def delete_node(session: SessionDep, id: int):
    node = Node.one_by_id(session, id)
    if not node:
        raise NotFoundException(message="Node not found")

    try:
        node.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete node: {e}")
