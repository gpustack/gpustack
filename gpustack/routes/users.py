from fastapi import APIRouter

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.users import User, UserCreate, UserUpdate, UserPublic, UsersPublic

router = APIRouter()


@router.get("/me")
async def get_user_me():
    return {}


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


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")
    return user


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    try:
        user = await User.create(session, user_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create user: {e}")

    return user


@router.put("/{id}", response_model=UserPublic)
async def update_user(session: SessionDep, id: int, user_in: UserUpdate):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    try:
        await user.update(session, user_in)
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
