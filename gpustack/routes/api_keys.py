from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.security import API_KEY_PREFIX, get_secret_hash
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, SessionDep
from gpustack.schemas.api_keys import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyListParams,
    ApiKeyPublic,
    ApiKeysPublic,
    ApiKeyUpdate,
)
from gpustack.schemas.users import User
from gpustack.server.services import APIKeyService

router = APIRouter()


@router.get("", response_model=ApiKeysPublic)
async def get_api_keys(
    user: CurrentUserDep,
    params: ApiKeyListParams = Depends(),
    search: str = None,
):
    fields = {"user_id": user.id}

    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    if params.watch:
        return StreamingResponse(
            ApiKey.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await ApiKey.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.post("", response_model=ApiKeyPublic)
async def create_api_key(
    session: SessionDep, user: CurrentUserDep, key_in: ApiKeyCreate
):
    fields = {"user_id": user.id, "name": key_in.name}
    existing = await ApiKey.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(message=f"Api key {key_in.name} already exists")
    selected_user = await User.one_by_id(session, user.id)
    if not selected_user:
        raise NotFoundException(message="User not found")
    try:
        current = datetime.now(timezone.utc)
        expires_at = None
        if key_in.expires_in and key_in.expires_in > 0:
            expires_at = current + timedelta(seconds=key_in.expires_in)

        # Handle custom API key vs auto-generated
        if key_in.custom_key is not None and key_in.custom_key.strip() != "":
            # Custom API key
            custom_key = key_in.custom_key.strip()
            # Use MD5 hash of custom key as access_key to prevent duplicates
            access_key = hashlib.md5(custom_key.encode()).hexdigest()
            secret_key = custom_key  # Use custom key as secret key
            is_custom = True
            hashed_secret_key = get_secret_hash(secret_key)
        else:
            # Auto-generated API key
            access_key = secrets.token_hex(8)
            secret_key = secrets.token_hex(16)
            is_custom = False
            hashed_secret_key = get_secret_hash(secret_key)

        api_key = ApiKey(
            name=key_in.name,
            description=key_in.description,
            user=selected_user,
            user_id=selected_user.id,
            access_key=access_key,
            hashed_secret_key=hashed_secret_key,
            is_custom=is_custom,
            custom_key=custom_key if is_custom else None,
            expires_at=expires_at,
            allowed_model_names=key_in.allowed_model_names,
        )
        api_key = await ApiKey.create(session, api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create api key: {e}")

    # Return the appropriate value based on whether it's custom or auto-generated
    if key_in.custom_key is not None and key_in.custom_key.strip() != "":
        # For custom keys, return the custom key value
        api_key_value = key_in.custom_key.strip()
    else:
        # For auto-generated keys, return the standard format
        api_key_value = f"{API_KEY_PREFIX}_{access_key}_{secret_key}"

    return ApiKeyPublic(
        name=api_key.name,
        description=api_key.description,
        id=api_key.id,
        value=api_key_value,
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
        expires_at=api_key.expires_at,
        allowed_model_names=api_key.allowed_model_names,
    )


@router.delete("/{id}")
async def delete_api_key(session: SessionDep, user: CurrentUserDep, id: int):
    api_key = await ApiKey.one_by_id(session, id)
    if not api_key or api_key.user_id != user.id:
        raise NotFoundException(message="Api key not found")

    try:
        await APIKeyService(session).delete(api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete api key: {e}")


@router.put("/{id}", response_model=ApiKeyPublic)
async def update_api_key(
    session: SessionDep, user: CurrentUserDep, id: int, key_in: ApiKeyUpdate
):
    api_key = await ApiKey.one_by_id(session, id)
    if not api_key or api_key.user_id != user.id:
        raise NotFoundException(message="Api key not found")
    try:
        await api_key.update(
            session=session, source=key_in.model_dump(exclude_unset=True)
        )
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update api key: {e}")
    return api_key
