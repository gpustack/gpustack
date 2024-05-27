from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.tasks import Task, TaskCreate, TaskPublic, TaskUpdate, TasksPublic
from gpustack.server.bus import Event, event_bus

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
        task.update(session, task_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update task: {e}")

    return task


@router.delete("/{id}")
async def delete_task(session: SessionDep, id: int):
    task = Task.one_by_id(session, id)
    if not task:
        raise NotFoundException(message="Task not found")

    try:
        task.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete task: {e}")
