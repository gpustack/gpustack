from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
    ConflictException,
)
from gpustack.schemas.organizations import OrganizationPublic
from gpustack.schemas.principals import (
    OrgRole,
    PLATFORM_PRINCIPAL_ID,
    Principal,
    PrincipalMembership,
    PrincipalType,
)
from gpustack.schemas.credentials import (
    get_password_credential,
    require_password_change as credential_requires_password_change,
    set_password,
)
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep
from gpustack.schemas.users import (
    User,
    UserActivationUpdate,
    UserCreate,
    UserListParams,
    UserUpdate,
    UserMePublic,
    UserPublic,
    UsersPublic,
    UserSelfUpdate,
)
from gpustack.server.services import UserService, create_user_with_principal

router = APIRouter()


class UserMembership(BaseModel):
    organization: OrganizationPublic
    role: OrgRole

    model_config = {"from_attributes": True}


@router.get("", response_model=UsersPublic)
async def get_users(
    params: UserListParams = Depends(),
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"username": search, "full_name": search}

    if params.watch:
        return StreamingResponse(
            User.streaming(fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await User.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            fields={
                "deleted_at": None,
                "is_system": False,
            },
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")
    return user


@router.get("/{id}/memberships", response_model=List[UserMembership])
async def list_user_memberships(session: SessionDep, id: int):
    """Admin-only: list the Orgs a user belongs to.

    Only ORG-kind principals are returned — the user's own
    USER-principal is intrinsic to them, not a "membership" anyone
    grants or revokes.
    """
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    stmt = (
        select(PrincipalMembership, Principal)
        .join(
            Principal,
            Principal.id == PrincipalMembership.parent_principal_id,
        )
        .where(
            PrincipalMembership.member_principal_id == user.principal_id,
            PrincipalMembership.deleted_at.is_(None),
            Principal.kind == PrincipalType.ORG,
            Principal.deleted_at.is_(None),
        )
    )
    rows = (await session.exec(stmt)).all()
    return [
        UserMembership(
            organization=OrganizationPublic.from_principal(org),
            role=membership.role or OrgRole.MEMBER,
        )
        for membership, org in rows
    ]


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    existing = await User.one_by_field(session, "username", user_in.username)
    if existing:
        raise AlreadyExistsException(message=f"User {user_in.username} already exists")

    try:
        to_create = User(
            username=user_in.username,
            full_name=user_in.full_name,
            is_admin=user_in.is_admin,
            is_active=user_in.is_active,
            avatar_url=user_in.avatar_url,
            source=user_in.source,
        )
        # User row + USER-principal go in one transactional helper —
        # ``users.principal_id`` is NOT NULL so the principal must
        # exist first. Admin additionally joins the platform Org as
        # OWNER; regular users do NOT auto-join — admin can add them
        # later if shared workspace access is needed.
        user = await create_user_with_principal(session, to_create)
        if user_in.password:
            await set_password(
                session,
                user.principal_id,
                user_in.password,
                auto_commit=False,
            )
        if user.is_admin:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(
                PrincipalMembership(
                    parent_principal_id=PLATFORM_PRINCIPAL_ID,
                    member_principal_id=user.principal_id,
                    role=OrgRole.OWNER,
                    created_at=now,
                    updated_at=now,
                )
            )

        await session.commit()
        await session.refresh(user)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create user: {e}")

    return user


@router.put("/{id}", response_model=UserPublic)
async def update_user(session: SessionDep, id: int, user_in: UserUpdate):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    if (
        user.is_active
        and user_in.is_active is False
        and await is_only_admin_user(session, user)
    ):
        raise ConflictException(message="Cannot deactivate the only admin user")

    try:
        update_data = user_in.model_dump()
        # ``password`` lives on a PASSWORD ``credentials`` row keyed
        # by principal — ``users.update`` no longer knows about it.
        password: Optional[str] = update_data.pop("password", None)
        # Original code already excluded ``source`` from user updates.
        update_data.pop("source", None)
        await UserService(session).update(user, update_data)
        if password:
            await set_password(
                session,
                user.principal_id,
                password,
                auto_commit=False,
            )
        await session.commit()
        await session.refresh(user)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user


@router.patch("/{id}/activation", response_model=UserPublic)
async def update_user_activation(
    session: SessionDep, id: int, activation_data: UserActivationUpdate
):
    """
    Activate or deactivate a user account.
    Only administrators can perform this action.
    """
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    changed = user.is_active != activation_data.is_active
    if not changed:
        return user

    if (
        user.is_active
        and activation_data.is_active is False
        and await is_only_admin_user(session, user)
    ):
        raise ConflictException(message="Cannot deactivate the only admin user")

    try:
        await UserService(session).update(
            user, {"is_active": activation_data.is_active}
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update user activation: {e}"
        )

    return user


@router.delete("/{id}")
async def delete_user(session: SessionDep, id: int):
    user_service = UserService(session)
    user = await user_service.get_by_id(id)
    if not user:
        raise NotFoundException(message="User not found")

    if await is_only_admin_user(session, user):
        raise ConflictException(message="Cannot delete the only admin user")

    # The user's USER-principal is the canonical owner of any
    # personal-scope resources (models, routes, clusters, api keys).
    # FK cascades on ``owner_principal_id == user.principal_id`` will
    # take those rows with the principal when it's deleted; the
    # principal in turn is RESTRICT-FK'd to ``users.principal_id``, so
    # the user must be deleted first and the principal cleaned up
    # afterward.
    principal = await Principal.one_by_id(session, user.principal_id)

    try:
        await user_service.delete(user)
        if principal is not None:
            await principal.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete user: {e}")


async def is_only_admin_user(session: SessionDep, user: User) -> bool:
    if not user.is_admin:
        return False
    admin_count = await User.count_by_fields(
        session, {"is_admin": True, "is_active": True}
    )
    return admin_count == 1


me_router = APIRouter()


async def _build_user_me(session: SessionDep, user: User) -> UserMePublic:
    credential = (
        await get_password_credential(session, user.principal_id)
        if user.principal_id is not None
        else None
    )
    me = UserMePublic.model_validate(user, from_attributes=True)
    me.require_password_change = credential_requires_password_change(credential)
    return me


@me_router.get("/me", response_model=UserMePublic)
async def get_user_me(session: SessionDep, user: CurrentUserDep):
    return await _build_user_me(session, user)


@me_router.put("/me", response_model=UserMePublic)
async def update_user_me(
    session: SessionDep, user: CurrentUserDep, user_in: UserSelfUpdate
):
    try:
        update_data = user_in.model_dump(exclude_none=True)
        password: Optional[str] = update_data.pop("password", None)
        if update_data:
            await UserService(session).update(user, update_data)
        if password:
            await set_password(
                session,
                user.principal_id,
                password,
                auto_commit=False,
            )
        await session.commit()
        await session.refresh(user)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return await _build_user_me(session, user)


# User-search endpoint accessible to org owners (any) and platform
# admins, so the Add Member picker works without the admin-gated full
# /users endpoint. Returns the standard UsersPublic page.
directory_router = APIRouter()


@directory_router.get("/user-directory", response_model=UsersPublic)
async def list_user_directory(
    ctx: TenantContextDep,
    page: int = 1,
    perPage: int = 30,
    search: str = None,
):
    if not ctx.is_platform_admin and ctx.org_role != OrgRole.OWNER:
        raise ForbiddenException(message="Insufficient permission")
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"username": search, "full_name": search}
    async with async_session() as session:
        return await User.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=page,
            per_page=perPage,
            fields={
                "deleted_at": None,
                "is_system": False,
            },
        )
