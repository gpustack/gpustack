from fastapi import APIRouter

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.security import get_password_hash
from gpustack.server.deps import CurrentUserDep, ListParamsDep, SessionDep
from gpustack.schemas.users import User, UserCreate, UserUpdate, UserPublic, UsersPublic

router = APIRouter()


@router.get("", response_model=UsersPublic)
async def get_users(session: SessionDep, params: ListParamsDep):
    fields = {}
    if params.query:
        fields = {"name": params.query}
    return await User.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/me", response_model=UserPublic)
async def get_user_me(user: CurrentUserDep):
    return user


@router.put("/me", response_model=UserPublic)
async def update_user_me(
    session: SessionDep, user: CurrentUserDep, user_in: UserUpdate
):
    try:
        update_data = user_in.model_dump()
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            update_data["hashed_password"] = hashed_password
            del update_data["password"]
        await user.update(session, update_data)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")
    return user


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    try:
        user_data = user_in.model_dump()
        if "password" in user_data:
            hashed_password = get_password_hash(user_in.password)
            user_data["hashed_password"] = hashed_password
            del user_data["password"]
        user = User(**user_data)
        await User.create(session, user)
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
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
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
