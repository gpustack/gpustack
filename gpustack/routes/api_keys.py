from datetime import datetime, timedelta, timezone
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import selectinload
from sqlmodel import col

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.security import API_KEY_PREFIX, get_secret_hash, get_key_pair
from gpustack.server.deps import CurrentUserDep, SessionDep
from gpustack.schemas.api_keys import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyListParams,
    ApiKeyPublic,
    ApiKeysPublic,
    ApiKeyUpdate,
)
from gpustack.server.services import APIKeyService
from gpustack.utils.api_keys import get_masked_api_key_value

router = APIRouter()

SYSTEM_API_KEY_NAME_PREFIX = "system/"


def _api_key_to_public(
    api_key: ApiKey, value: str = None, user_name: str = None
) -> ApiKeyPublic:
    """Convert an ApiKey object to an ApiKeyPublic object."""
    return ApiKeyPublic(
        name=api_key.name,
        description=api_key.description,
        id=api_key.id,
        user_name=user_name or api_key.user_name,
        value=value,
        masked_value=get_masked_api_key_value(api_key.access_key, api_key.is_custom),
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
        expires_at=api_key.expires_at,
        allowed_model_names=api_key.allowed_model_names,
        is_custom=api_key.is_custom,
        scope=api_key.scope,
    )


def _is_system_api_key(api_key: ApiKey) -> bool:
    return api_key.name.startswith(SYSTEM_API_KEY_NAME_PREFIX)


@router.get("", response_model=ApiKeysPublic)
async def get_api_keys(
    session: SessionDep,
    user: CurrentUserDep,
    params: ApiKeyListParams = Depends(),
    user_id: Optional[str] = Query(
        None, description="Filter by user_id. Admin can use '*' to list all users."
    ),
    search: str = None,
):
    fields = {"user_id": user.id}
    if user.is_admin and user_id is not None:
        if user_id == "*":
            fields = {}
        else:
            try:
                fields = {"user_id": int(user_id)}
            except ValueError:
                raise InvalidException(message="user_id must be an integer or '*'")

    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    extra_conditions = [col(ApiKey.name).not_like(f"{SYSTEM_API_KEY_NAME_PREFIX}%")]

    if params.watch:
        return StreamingResponse(
            ApiKey.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda api_key: not _is_system_api_key(api_key),
                options=[selectinload(ApiKey.user)],
            ),
            media_type="text/event-stream",
        )

    result = await ApiKey.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
        options=[selectinload(ApiKey.user)],
    )

    # Convert ApiKey to ApiKeyPublic
    items = [_api_key_to_public(item) for item in result.items]
    result.items = items
    return result


@router.post("", response_model=ApiKeyPublic)
async def create_api_key(
    session: SessionDep, user: CurrentUserDep, key_in: ApiKeyCreate
):
    fields = {"user_id": user.id, "name": key_in.name}
    existing = await ApiKey.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(message=f"Api key {key_in.name} already exists")

    if key_in.custom is None:
        access_key, secret_key = secrets.token_hex(8), secrets.token_hex(16)
    else:
        access_key, secret_key = get_key_pair(key_in.custom)
        existing_key = await ApiKey.one_by_field(
            session=session, field="access_key", value=access_key
        )
        if existing_key:
            expired = (
                existing_key.expires_at is not None
                and existing_key.expires_at <= datetime.now(timezone.utc)
            )
            message = (
                "Custom API Key duplicate with existing key "
                f"{existing_key.name} (id: {existing_key.id}, expired: {expired})"
            )
            raise AlreadyExistsException(message=message)

    current = datetime.now(timezone.utc)
    expires_at = None
    if key_in.expires_in and key_in.expires_in > 0:
        expires_at = current + timedelta(seconds=key_in.expires_in)

    try:
        api_key = ApiKey(
            name=key_in.name,
            description=key_in.description,
            user_id=user.id,
            access_key=access_key,
            hashed_secret_key=get_secret_hash(secret_key),
            expires_at=expires_at,
            allowed_model_names=key_in.allowed_model_names,
            is_custom=key_in.custom is not None,
            scope=key_in.scope,
        )
        api_key = await ApiKey.create(session, api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create api key: {e}")

    value = (
        key_in.custom
        if key_in.custom
        else f"{API_KEY_PREFIX}_{access_key}_{secret_key}"
    )
    return _api_key_to_public(api_key, value=value, user_name=user.username)


@router.delete("/{id}")
async def delete_api_key(session: SessionDep, user: CurrentUserDep, id: int):
    api_key = await ApiKey.one_by_id(session, id)
    if not api_key or (api_key.user_id != user.id and not user.is_admin):
        raise NotFoundException(message="Api key not found")

    try:
        await APIKeyService(session).delete(api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete api key: {e}")


@router.put("/{id}", response_model=ApiKeyPublic)
async def update_api_key(
    session: SessionDep, user: CurrentUserDep, id: int, key_in: ApiKeyUpdate
):
    api_key = await ApiKey.one_by_id(session, id, options=[selectinload(ApiKey.user)])
    user_name = api_key.user.username if api_key and api_key.user else None
    if not api_key or (api_key.user_id != user.id and not user.is_admin):
        raise NotFoundException(message="Api key not found")
    try:
        await APIKeyService(session).update(
            api_key=api_key,
            source=key_in.model_dump(exclude_unset=True),
        )
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update api key: {e}")
    return _api_key_to_public(api_key, user_name=user_name)
