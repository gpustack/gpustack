from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.security import get_secret_hash
from gpustack.server.deps import CurrentUserDep, ListParamsDep, SessionDep
from gpustack.schemas.users import User, UserCreate, UserUpdate, UserPublic, UsersPublic

router = APIRouter()


@router.get("", response_model=UsersPublic)
async def get_users(session: SessionDep, params: ListParamsDep, search: str = None):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"username": search, "full_name": search}

    if params.watch:
        return StreamingResponse(
            User.streaming(session, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await User.paginated_by_query(
        session=session,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")
    return user


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    existing = await User.one_by_field(session, "username", user_in.username)
    if existing:
        raise AlreadyExistsException(message=f"User f{user_in.username} already exists")

    try:
        to_create = User(
            username=user_in.username,
            full_name=user_in.full_name,
            is_admin=user_in.is_admin,
        )
        if user_in.password:
            to_create.hashed_password = get_secret_hash(user_in.password)
        user = await User.create(session, to_create)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create user: {e}")

    return user


@router.put("/{id}", response_model=UserPublic)
async def update_user(session: SessionDep, id: int, user_in: UserUpdate):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    try:
        update_data = user_in.model_dump()
        if user_in.password:
            hashed_password = get_secret_hash(user_in.password)
            update_data["hashed_password"] = hashed_password
        del update_data["password"]
        await user.update(session, update_data)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user


@router.delete("/{id}")
async def delete_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    try:
        await user.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete user: {e}")


me_router = APIRouter()


@me_router.get("/me", response_model=UserPublic)
async def get_user_me(user: CurrentUserDep):
    return user


@me_router.put("/me", response_model=UserPublic)
async def update_user_me(
    session: SessionDep, user: CurrentUserDep, user_in: UserUpdate
):
    try:
        update_data = user_in.model_dump()
        if "password" in update_data:
            hashed_password = get_secret_hash(update_data["password"])
            update_data["hashed_password"] = hashed_password
            del update_data["password"]
        await user.update(session, update_data)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user
