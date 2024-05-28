from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
import httpx

from gpustack.agent.logs import LogOptionsDep
from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.nodes import Node
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.tasks import Task, TaskCreate, TaskPublic, TaskUpdate, TasksPublic

router = APIRouter()


@router.get("", response_model=TasksPublic)
async def get_tasks(session: SessionDep, params: ListParamsDep):
    fields = {}
    if params.query:
        fields = {"name": params.query}

    if params.watch:
        return StreamingResponse(Task.streaming(), media_type="text/event-stream")

    return Task.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=TaskPublic)
async def get_task(session: SessionDep, id: int):
    task = Task.one_by_id(session, id)
    if not task:
        raise NotFoundException(message="Task not found")
    return task


@router.get("/{id}/logs")
async def get_task_logs(
    request: Request, session: SessionDep, id: int, log_options: LogOptionsDep
):
    task = Task.one_by_id(session, id)
    if not task:
        raise NotFoundException(message="Task not found")

    if not task.node_id:
        raise NotFoundException(message="Task not assigned to a node")

    # proxy to node agent's task logs endpoint
    node = Node.one_by_id(session, task.node_id)
    if not node:
        raise NotFoundException(message="Task's node not found")

    task_log_url = (
        f"http://{node.address}:10050/taskLogs/{task.id}?{log_options.url_encode()}"
    )

    client: httpx.AsyncClient = request.app.state.http_client

    if log_options.follow:

        async def proxy_stream():
            async with client.stream("GET", task_log_url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Error fetching task logs",
                    )
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(proxy_stream(), media_type="application/octet-stream")
    else:
        response = await client.get(task_log_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Error fetching task logs",
            )

        return PlainTextResponse(
            content=response.text, status_code=response.status_code
        )


@router.post("", response_model=TaskPublic)
async def create_task(session: SessionDep, task_in: TaskCreate):
    try:
        task = await Task.create(session, task_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create task: {e}")

    return task


@router.put("/{id}", response_model=TaskPublic)
async def update_task(session: SessionDep, id: int, task_in: TaskUpdate):
    task = Task.one_by_id(session, id)
    if not task:
        raise NotFoundException(message="Task not found")

    try:
        await task.update(session, task_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update task: {e}")

    return task


@router.delete("/{id}")
async def delete_task(session: SessionDep, id: int):
    task = Task.one_by_id(session, id)
    if not task:
        raise NotFoundException(message="Task not found")

    try:
        await task.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete task: {e}")
