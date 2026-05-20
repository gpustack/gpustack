from datetime import datetime, timezone
from typing import List

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
    Principal,
    PrincipalMembership,
    PrincipalType,
    get_platform_principal_id,
)
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep
from gpustack.schemas.users import (
    User,
    UserActivationUpdate,
    UserCreate,
    UserListParams,
    UserUpdate,
    UserPublic,
    UsersPublic,
    UserSelfUpdate,
)
from gpustack.server.passwords import set_password
from gpustack.server.services import UserService

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
        fuzzy_fields = {"slug": search, "name": search}

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
                "kind": PrincipalType.USER,
                "deleted_at": None,
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
            PrincipalMembership.member_principal_id == user.id,
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
    existing = await User.one_by_field(session, "slug", user_in.username)
    if existing:
        raise AlreadyExistsException(message=f"User {user_in.username} already exists")

    try:
        to_create = User(
            slug=user_in.username,
            name=user_in.full_name,
            is_admin=user_in.is_admin,
            is_active=user_in.is_active,
        )
        # Admin additionally joins the platform Org as OWNER; regular
        # users do NOT auto-join — admin can add them later if shared
        # workspace access is needed.
        user = await User.create(session, to_create, auto_commit=False)
        if user_in.password:
            await set_password(
                session,
                user.id,
                user_in.password,
                require_password_change=user_in.require_password_change,
                auto_commit=False,
            )
        if user.is_admin:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            platform_id = await get_platform_principal_id(session)
            session.add(
                PrincipalMembership(
                    parent_principal_id=platform_id,
                    member_principal_id=user.id,
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
        # ``by_alias=True`` maps the wire-level ``username`` / ``full_name``
        # back onto the Principal columns ``slug`` / ``name`` for the
        # storage write — see :class:`UserBase`. ``exclude_none=True``
        # so omitting an optional field (e.g. ``full_name``) on the
        # PUT request leaves the stored value alone rather than
        # clobbering it with NULL; matches :func:`update_user_me`.
        # Password + require-change flag live on the ``user_passwords``
        # row, handled separately.
        update_data = user_in.model_dump(by_alias=True, exclude_none=True)
        update_data.pop("password", None)
        update_data.pop("source", None)
        require_change = update_data.pop("require_password_change", False)
        await UserService(session).update(user, update_data)
        if user_in.password:
            await set_password(
                session,
                user.id,
                user_in.password,
                require_password_change=require_change,
            )
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

    # After identity consolidation the user IS its own principal, so
    # deleting the user row also tears down every resource owned by
    # ``owner_principal_id == user.id`` via FK CASCADE. No separate
    # principal row to clean up.
    try:
        await user_service.delete(user)
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


@me_router.get("/me", response_model=UserPublic)
async def get_user_me(user: CurrentUserDep):
    return user


@me_router.put("/me", response_model=UserPublic)
async def update_user_me(
    session: SessionDep, user: CurrentUserDep, user_in: UserSelfUpdate
):
    try:
        update_data = user_in.model_dump(by_alias=True, exclude_none=True)
        plain = update_data.pop("password", None)
        await UserService(session).update(user, update_data)
        if plain:
            await set_password(session, user.id, plain)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user


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
        fuzzy_fields = {"slug": search, "name": search}
    async with async_session() as session:
        return await User.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=page,
            per_page=perPage,
            fields={
                "kind": PrincipalType.USER,
                "deleted_at": None,
            },
        )
