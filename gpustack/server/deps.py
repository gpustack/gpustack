from typing import Annotated
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.schemas.users import User
from gpustack.api.auth import get_admin_user, get_current_user
from gpustack.api.tenant import (
    TenantContext,
    get_tenant_context,
    require_platform_admin,
)
from gpustack.server.db import get_session
from gpustack.schemas.common import ListParams

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ListParamsDep = Annotated[ListParams, Depends(ListParams)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentAdminUserDep = Annotated[User, Depends(get_admin_user)]
TenantContextDep = Annotated[TenantContext, Depends(get_tenant_context)]
PlatformAdminDep = Annotated[TenantContext, Depends(require_platform_admin)]
