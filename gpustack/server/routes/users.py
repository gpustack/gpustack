from fastapi import APIRouter, HTTPException

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
    return User.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = User.one_by_id(session, id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    user = User.model_validate(user_in)

    return user.save(session)


@router.put("/{id}", response_model=UserPublic)
async def update_user(session: SessionDep, user_in: UserUpdate):
    user = User.model_validate(user_in)
    return user.save(session)


@router.delete("/{id}")
async def delete_user(session: SessionDep, id: int):
    user = User.one_by_id(session, id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user.delete(session)
