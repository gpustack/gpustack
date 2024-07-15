from datetime import datetime, timedelta, timezone
import secrets
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.security import API_KEY_PREFIX, get_secret_hash
from gpustack.server.deps import CurrentUserDep, ListParamsDep, SessionDep
from gpustack.schemas.api_keys import ApiKey, ApiKeyCreate, ApiKeyPublic, ApiKeysPublic

router = APIRouter()


@router.get("", response_model=ApiKeysPublic)
async def get_api_keys(
    session: SessionDep, user: CurrentUserDep, params: ListParamsDep, search: str = None
):
    fields = {"user_id": user.id}

    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    if params.watch:
        return StreamingResponse(
            ApiKey.streaming(session, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await ApiKey.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.post("", response_model=ApiKeyPublic)
async def create_api_key(
    session: SessionDep, user: CurrentUserDep, key_in: ApiKeyCreate
):
    fields = {"user_id": user.id, "name": key_in.name}
    existing = await ApiKey.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(message=f"Api key {key_in.name} already exists")

    try:
        access_key = secrets.token_hex(8)
        secret_key = secrets.token_hex(16)

        current = datetime.now(timezone.utc)
        expires_at = None
        if key_in.expires_in and key_in.expires_in > 0:
            expires_at = current + timedelta(seconds=key_in.expires_in)

        api_key = ApiKey(
            name=key_in.name,
            description=key_in.description,
            user_id=user.id,
            access_key=access_key,
            hashed_secret_key=get_secret_hash(secret_key),
            expires_at=expires_at,
        )
        api_key = await ApiKey.create(session, api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create api key: {e}")

    return ApiKeyPublic(
        name=api_key.name,
        description=api_key.description,
        id=api_key.id,
        value=f"{API_KEY_PREFIX}_{access_key}_{secret_key}",
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
        expires_at=api_key.expires_at,
    )


@router.delete("/{id}")
async def delete_api_key(session: SessionDep, user: CurrentUserDep, id: int):
    api_key = await ApiKey.one_by_id(session, id)
    if not api_key or api_key.user_id != user.id:
        raise NotFoundException(message="Api key not found")

    try:
        await api_key.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete api key: {e}")
