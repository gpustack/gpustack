from datetime import datetime, timedelta, timezone
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import selectinload
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.security import API_KEY_PREFIX, get_secret_hash, get_key_pair
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.api_keys import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyListParams,
    ApiKeyPublic,
    ApiKeysPublic,
    ApiKeyUpdate,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.users import User
from gpustack.server.services import APIKeyService
from gpustack.utils.api_keys import get_masked_api_key_value

router = APIRouter()


def _is_system_owned(api_key: ApiKey) -> bool:
    """API keys whose owner is a SYSTEM principal (workers, cluster
    sync, etc.).

    Filters by the owner's ``kind`` rather than the api_key name
    because not every system-managed key follows the ``system/``
    naming scheme (``Legacy Cluster Token``, ``Default Cluster Token``,
    …). Requires ``selectinload(ApiKey.user)`` on the query feeding
    this so the relationship is hydrated before access.
    """
    return bool(api_key.user and api_key.user.kind == PrincipalType.SYSTEM)


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
        owner_principal_id=api_key.owner_principal_id,
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
        expires_at=api_key.expires_at,
        allowed_model_names=api_key.allowed_model_names,
        is_custom=api_key.is_custom,
        scope=api_key.scope,
    )


def _is_hidden_api_key(api_key: ApiKey) -> bool:
    return _is_system_owned(api_key)


def _can_manage_org_api_keys(ctx) -> bool:
    if ctx.user.is_admin:
        return True
    return ctx.current_principal_id is not None and ctx.org_role == OrgRole.OWNER


def _api_key_list_fields(ctx, user_id: Optional[str]) -> dict:
    """Build list filters for the caller's API key management scope.

    Org owners manage all API keys in the current Org. Regular members only
    manage their own keys in that Org. Platform admins keep the historical
    cross-org behavior: no org context lists their own keys unless user_id is
    supplied, with "*" meaning all users.
    """
    user = ctx.user
    fields = {}
    if ctx.current_principal_id is not None:
        fields["owner_principal_id"] = ctx.current_principal_id

    if _can_manage_org_api_keys(ctx):
        if user_id is None:
            if user.is_admin and ctx.current_principal_id is None:
                fields["user_id"] = user.id
            return fields
        if user_id == "*":
            return fields
        try:
            fields["user_id"] = int(user_id)
        except ValueError:
            raise InvalidException(message="user_id must be an integer or '*'")
        return fields

    fields["user_id"] = user.id
    return fields


@router.get("", response_model=ApiKeysPublic)
async def get_api_keys(
    session: SessionDep,
    ctx: TenantContextDep,
    params: ApiKeyListParams = Depends(),
    user_id: Optional[str] = Query(
        None, description="Filter by user_id. Admin can use '*' to list all users."
    ),
    search: str = None,
):
    fields = _api_key_list_fields(ctx, user_id)

    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    # Hide system-owned keys (workers, cluster sync, legacy/default
    # cluster tokens). The set isn't covered by a clean name prefix —
    # entries like "Default Cluster Token" / "Legacy Cluster Token"
    # exist alongside the "system/..." names — so we filter by the
    # owning principal's kind, which catches every variant.
    extra_conditions = [
        ApiKey.user_id.notin_(select(User.id).where(User.kind == PrincipalType.SYSTEM)),
    ]

    if params.watch:
        return StreamingResponse(
            ApiKey.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda api_key: not _is_hidden_api_key(api_key),
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
    session: SessionDep, ctx: TenantContextDep, key_in: ApiKeyCreate
):
    user = ctx.user
    # Admin "All" mode (no Org context) creates an untenant-pinned key:
    # ``owner_principal_id`` stays NULL so ``_resolve_requested_principal_id``
    # falls through to user-based resolution on each request and admin
    # gets the same cross-principal ``bypass_tenant_filter`` reach as
    # their cookie session. For every other caller — Org act-as, Org
    # member, personal scope — ``current_principal_id`` is already
    # non-NULL by ``_resolve_requested_principal_id`` design and pins
    # the key to that principal.
    target_org_id = ctx.current_principal_id
    fields = {
        "user_id": user.id,
        "owner_principal_id": target_org_id,
        "name": key_in.name,
    }
    existing = await ApiKey.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(
            message=f"API key with name '{key_in.name}' already exists."
        )

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
            owner_principal_id=target_org_id,
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
    return _api_key_to_public(api_key, value=value, user_name=user.name)


def _api_key_in_scope(api_key: ApiKey, ctx) -> bool:
    """An api_key is in the caller's scope if the caller is its owner, or a
    platform/org admin acting either across all orgs or in the key's org.
    """
    user = ctx.user
    if api_key.user_id == user.id and (
        ctx.current_principal_id is None
        or api_key.owner_principal_id == ctx.current_principal_id
    ):
        return True
    if user.is_admin and (
        ctx.current_principal_id is None
        or api_key.owner_principal_id == ctx.current_principal_id
    ):
        return True
    if (
        ctx.current_principal_id is not None
        and ctx.org_role == OrgRole.OWNER
        and api_key.owner_principal_id == ctx.current_principal_id
    ):
        return True
    return False


@router.delete("/{id}")
async def delete_api_key(session: SessionDep, ctx: TenantContextDep, id: int):
    api_key = await ApiKey.one_by_id(session, id)
    if not api_key or not _api_key_in_scope(api_key, ctx):
        raise NotFoundException(message="Api key not found")

    try:
        await APIKeyService(session).delete(api_key)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete api key: {e}")


@router.put("/{id}", response_model=ApiKeyPublic)
async def update_api_key(
    session: SessionDep, ctx: TenantContextDep, id: int, key_in: ApiKeyUpdate
):
    api_key = await ApiKey.one_by_id(session, id, options=[selectinload(ApiKey.user)])
    user_name = api_key.user.name if api_key and api_key.user else None
    if not api_key or not _api_key_in_scope(api_key, ctx):
        raise NotFoundException(message="Api key not found")
    try:
        await APIKeyService(session).update(
            api_key=api_key,
            source=key_in.model_dump(exclude_unset=True),
        )
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update api key: {e}")
    return _api_key_to_public(api_key, user_name=user_name)
