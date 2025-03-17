from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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

    fuzzy_fields = {}
    if search:
        fuzzy_fields = {
            "huggingface_repo_id": search,
            "huggingface_filename": search,
            "ollama_library_model_name": search,
            "model_scope_model_id": search,
            "model_scope_file_path": search,
            "local_path": search,
        }

    if params.watch:
        return StreamingResponse(
            ModelFile.streaming(session, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await ModelFile.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


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
async def delete_model_file(session: SessionDep, id: int):
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
