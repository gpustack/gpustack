from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.users import User
from gpustack.api.auth import get_admin_user, get_current_user
from gpustack.server.db import get_session, get_engine
from gpustack.schemas.common import ListParams

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
ListParamsDep = Annotated[ListParams, Depends(ListParams)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentAdminUserDep = Annotated[User, Depends(get_admin_user)]
