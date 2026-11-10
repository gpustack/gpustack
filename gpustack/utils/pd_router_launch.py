"""Resolve Router backend definitions and command ownership."""

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import BackendEnum, find_role, role_backend_field
from gpustack.utils.inference_backend import merge_backend_definitions


@dataclass
class RouterLaunch:
    source: str = 'recipe'
    command: Optional[str] = None
    entrypoint: Optional[List[str]] = None
    version: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def uses_backend_command(self) -> bool:
        return self.source in ('backend_version', 'backend_default')


def resolve_router_launch(
    model, backend: Optional[InferenceBackend] = None
) -> RouterLaunch:
    """Role command > selected version > selected backend default > recipe.

    Only an explicitly selected Router backend contributes a command. A parent
    backend supplies image/version defaults, never its engine invocation.
    Backend commands retain their entrypoint/args contract; role commands are
    complete invocations and replace that contract.
    """
    role = find_role(model, 'router')
    launch = RouterLaunch()
    if role is None:
        return launch
    if backend is not None and getattr(role, 'backend', None) == backend.backend_name:
        version = role_backend_field(role, 'backend_version', None)
        launch.version = backend.resolve_target_version(version) or version
        config = backend.version_configs.root.get(launch.version)
        launch.env = backend.get_backend_env(launch.version)
        version_command = config.run_command if config else None
        if version_command and version_command.strip():
            launch.source, launch.command = 'backend_version', version_command
        elif backend.default_run_command and backend.default_run_command.strip():
            launch.source, launch.command = (
                'backend_default',
                backend.default_run_command,
            )
        if launch.command:
            launch.entrypoint = backend.get_container_entrypoint(launch.version)
            if launch.version is None and backend.default_entrypoint:
                launch.entrypoint = shlex.split(backend.default_entrypoint)
    command = getattr(role, 'run_command', None)
    if command and command.strip():
        launch.source, launch.command, launch.entrypoint = 'role', command, None
    return launch


async def router_backend_definition(
    session: AsyncSession, model, *, cached: bool = False
) -> Optional[InferenceBackend]:
    """Load the Router's selected backend within the deployment owner's scope.

    Reconciliation uses the existing backend catalog cache, invalidated by
    backend writes and synchronized across servers. Admission reads its current
    session so an uncommitted definition can never be replaced by a cache entry.
    """
    role = find_role(model, 'router')
    if (
        role is None
        or not getattr(role, 'backend', None)
        or role.backend == BackendEnum.CUSTOM
    ):
        return None
    owner_id = getattr(model, 'owner_principal_id', None)
    if cached:
        scopes = {
            backend.owner_principal_id: backend
            for backend in await InferenceBackend.cached_all()
            if backend.backend_name == role.backend
            and backend.owner_principal_id in (None, owner_id)
        }
        platform = scopes.get(None)
        owner = scopes.get(owner_id) if owner_id is not None else None
    else:
        platform = await InferenceBackend.one_by_fields(
            session, {'backend_name': role.backend, 'owner_principal_id': None}
        )
        owner = (
            await InferenceBackend.one_by_fields(
                session, {'backend_name': role.backend, 'owner_principal_id': owner_id}
            )
            if owner_id is not None
            else None
        )
    return merge_backend_definitions(platform, owner)
