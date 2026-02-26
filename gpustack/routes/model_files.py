from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import String, cast, func, or_
from pathlib import Path
from sqlalchemy.orm import selectinload

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileCreate,
    ModelFileListParams,
    ModelFilePublic,
    ModelFileStateEnum,
    ModelFileUpdate,
    ModelFilesPublic,
)

router = APIRouter()


@router.get("", response_model=ModelFilesPublic)
async def get_model_files(
    params: ModelFileListParams = Depends(),
    search: str = None,
    worker_id: int = None,
):
    fields = {}

    if worker_id:
        fields["worker_id"] = worker_id

    def get_filter_func(search):
        if search:
            return lambda data: search_model_file_filter(data, search)
        return None

    if params.watch:
        return StreamingResponse(
            ModelFile.streaming(
                fields=fields,
                filter_func=get_filter_func(search),
            ),
            media_type="text/event-stream",
        )

    extra_conditions = []
    if search:
        lower_search = search.lower()
        extra_conditions.append(
            or_(
                *[
                    func.lower(cast(ModelFile.resolved_paths, String)).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.huggingface_repo_id).like(f"%{lower_search}%"),
                    func.lower(ModelFile.huggingface_filename).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.model_scope_model_id).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.model_scope_file_path).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.local_path).like(f"%{lower_search}%"),
                ]
            )
        )

    order_by = params.order_by

    order_by = params.order_by
    if order_by:
        new_order_by = []
        for field, direction in order_by:
            if field == "source":
                # When sorting by "source", add additional sorting fields for deterministic ordering
                new_order_by.append((field, direction))
                new_order_by.append(("huggingface_repo_id", direction))
                new_order_by.append(("huggingface_filename", direction))
                new_order_by.append(("model_scope_model_id", direction))
                new_order_by.append(("model_scope_file_path", direction))
                new_order_by.append(("local_path", direction))
            elif field == "resolved_paths":
                # resolved_paths is a JSON field, replace resolved_paths ordering with expression
                new_order_by.append((cast(ModelFile.resolved_paths, String), direction))
            else:
                new_order_by.append((field, direction))
        order_by = new_order_by

    async with async_session() as session:
        return await ModelFile.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=order_by,
        )


def search_model_file_filter(data: ModelFile, search: str) -> bool:
    if (
        (
            data.huggingface_repo_id
            and search.lower() in data.huggingface_repo_id.lower()
        )
        or (
            data.huggingface_filename
            and search.lower() in data.huggingface_filename.lower()
        )
        or (
            data.model_scope_model_id
            and search.lower() in data.model_scope_model_id.lower()
        )
        or (
            data.model_scope_file_path
            and search.lower() in data.model_scope_file_path.lower()
        )
        or (data.local_path and search.lower() in data.local_path.lower())
        or (data.resolved_paths and search.lower() in data.resolved_paths[0].lower())
    ):
        return True

    return False


@router.get("/{id}", response_model=ModelFilePublic)
async def get_model_file(session: SessionDep, id: int):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")
    return model_file


@router.post("", response_model=ModelFilePublic)
async def create_model_file(session: SessionDep, model_file_in: ModelFileCreate):
    fields = {
        "worker_id": model_file_in.worker_id,
        "source_index": model_file_in.model_source_index,
        "local_dir": model_file_in.local_dir,
    }
    existing = await ModelFile.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(
            message="Model file with the same model source already exists on the worker."
        )

    if model_file_in.local_dir is not None:
        fields = {
            "worker_id": model_file_in.worker_id,
            "local_dir": model_file_in.local_dir,
        }
        worker_existing_files = await ModelFile.all_by_field(
            session, field="worker_id", value=model_file_in.worker_id
        )
        if worker_existing_files:
            for file in worker_existing_files:
                if (
                    file.local_dir is not None
                    and file.huggingface_filename is None
                    and file.model_scope_file_path is None
                    and Path(file.local_dir).resolve()
                    == Path(model_file_in.local_dir).resolve()
                ):
                    raise AlreadyExistsException(
                        message=f"The local directory {model_file_in.local_dir} is already occupied by {file.readable_source} on this worker."
                    )
    try:
        model_file = ModelFile(
            **model_file_in.model_dump(), source_index=model_file_in.model_source_index
        )
        model_file = await ModelFile.create(session, model_file)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create model file: {e}")

    return model_file


@router.put("/{id}", response_model=ModelFilePublic)
async def update_model_file(
    session: SessionDep, id: int, model_file_in: ModelFileUpdate
):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    try:
        await model_file.update(session, model_file_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model file: {e}")

    return model_file


@router.delete("/{id}")
async def delete_model_file(
    session: SessionDep, id: int, cleanup: Optional[bool] = None
):
    model_file = await ModelFile.one_by_id(
        session, id, options=[selectinload(ModelFile.instances)]
    )
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    if model_file.instances:
        model_instance_names = ", ".join(
            [model_instance.name for model_instance in model_file.instances]
        )
        raise ConflictException(
            message=f"Cannot delete the model file. It's being used by model instances: {model_instance_names}.",
        )

    try:
        if cleanup is not None and model_file.cleanup_on_delete != cleanup:
            model_file.cleanup_on_delete = cleanup
            await model_file.update(session)

        await model_file.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model file: {e}")


@router.post("/{id}/reset", response_model=ModelFilePublic)
async def reset_model_file(session: SessionDep, id: int):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    try:
        model_file.state = ModelFileStateEnum.DOWNLOADING
        model_file.download_progress = 0
        model_file.state_message = ""

        await model_file.update(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model file: {e}")

    return model_file
