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

    validate_roles(model_in, stored=stored)
    config = await resolve_pairing_model_config(model_in, stored=stored)
    validate_role_pairing(model_in, stored=stored, pretrained_config=config)
    await validate_pd_mode_runtime(
        session, model_in, cluster_id=cluster_id, stored=stored
    )
    await validate_gather_layer(session, model_in, cluster_id=cluster_id, stored=stored)
