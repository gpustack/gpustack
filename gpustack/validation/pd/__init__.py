"""PD deployment validation: roles, pairing, runtime and placement constraints.

``validate_pd_config`` is the admission entry point. Parameter and environment
interpretation is also available to the scheduler, controller and worker from
the corresponding modules without importing the admission dependencies.
"""

from typing import Optional, Union

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.models import Model, ModelCreate, ModelSpecBase, ModelUpdate


async def validate_pd_config(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
    owner_principal_id: Optional[int] = None,
) -> None:
    """Validate the effective PD configuration before accepting a deployment.

    Args:
        session: The database session for cluster eligibility checks.
        model_in: The submitted deployment specification.
        cluster_id: The cluster selected by the caller.
        stored: The stored deployment supplying omitted update fields.
    """
    from gpustack.validation.pd.pairing import validate_role_pairing
    from gpustack.validation.pd.placement import validate_gather_layer
    from gpustack.validation.pd.roles import validate_roles
    from gpustack.validation.pd.runtime import (
        resolve_pairing_model_config,
        validate_pd_mode_runtime,
    )

    from gpustack.validation.pd.common import effective_model_spec
    from gpustack.utils.pd_router_launch import (
        resolve_router_launch,
        router_backend_definition,
    )
    from gpustack.schemas.models import find_role
    from gpustack.server.pd_mode_catalog import get_pd_mode
    from gpustack.schemas.pd_modes import PDRouterProtocolEnum
    from gpustack.api.exceptions import BadRequestException

    effective = effective_model_spec(model_in, stored)
    scope = (
        getattr(stored, 'owner_principal_id', None) if stored else owner_principal_id
    )
    # Evaluation specs already carry a stamped owner; keep it when the caller
    # supplies neither a stored model nor an explicit admission scope.
    if scope is not None:
        effective = effective.model_copy(update={'owner_principal_id': scope})
    backend = await router_backend_definition(session, effective)
    validate_roles(model_in, stored=stored, router_backend=backend)
    if effective.disaggregation and find_role(effective, 'router'):
        mode = get_pd_mode(effective.disaggregation.mode.value)
        has_default = bool(
            mode
            and mode.router
            and mode.router.protocol != PDRouterProtocolEnum.USER_PROVIDED
        )
        if not has_default and not resolve_router_launch(effective, backend).command:
            raise BadRequestException(
                message="Router requires a command from its role or selected backend."
            )
    config = await resolve_pairing_model_config(model_in, stored=stored)
    validate_role_pairing(model_in, stored=stored, pretrained_config=config)
    await validate_pd_mode_runtime(
        session, model_in, cluster_id=cluster_id, stored=stored
    )
    await validate_gather_layer(session, model_in, cluster_id=cluster_id, stored=stored)
