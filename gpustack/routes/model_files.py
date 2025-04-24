from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlmodel import String, cast, func, or_

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileCreate,
    ModelFilePublic,
    ModelFileStateEnum,
    ModelFileUpdate,
    ModelFilesPublic,
)

router = APIRouter()


@router.get("", response_model=ModelFilesPublic)
async def get_model_files(
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    worker_id: int = None,
):
    fields = {}

    if worker_id:
        fields["worker_id"] = worker_id

    if params.watch:
        return StreamingResponse(
            ModelFile.streaming(
                session,
                fields=fields,
                filter_func=lambda data: (
                    search_model_file_filter(data, search) if search else None
                ),
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
                    func.lower(ModelFile.ollama_library_model_name).like(
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

    return await ModelFile.paginated_by_query(
        session=session,
        fields=fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
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
            data.ollama_library_model_name
            and search.lower() in data.ollama_library_model_name.lower()
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
    model_file = await ModelFile.one_by_id(session, id)
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
