from fastapi import APIRouter, HTTPException


from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.tasks import Task, TaskCreate, TaskUpdate, TaskPublic, TasksPublic

router = APIRouter()


@router.get("", response_model=TasksPublic)
async def get_tasks(session: SessionDep, params: ListParamsDep):
    fields = {}
    if params.query:
        fields = {"name": params.query}
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
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("", response_model=TaskPublic)
async def create_task(session: SessionDep, task_in: TaskCreate):
    task = Task.model_validate(task_in)

    return task.save(session)


@router.put("/{id}", response_model=TaskPublic)
async def update_task(session: SessionDep, task_in: TaskUpdate):
    task = Task.model_validate(task_in)
    return task.save(session)


@router.delete("/{id}")
async def delete_task(session: SessionDep, id: int):
    task = Task.one_by_id(session, id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task.delete(session)
