from fastapi import APIRouter, Request

from gpustack.api.tenant import assert_cluster_visible
from gpustack.config.config import Config
from gpustack.scheduler.evaluator import evaluate_models
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.model_evaluations import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
)

from gpustack.api.exceptions import (
    InternalServerErrorException,
)
from gpustack.server.deps import SessionDep, TenantContextDep


router = APIRouter()


@router.post("", response_model=ModelEvaluationResponse)
async def create_model_evaluation(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    model_evaluation_in: ModelEvaluationRequest,
):
    config: Config = request.app.state.server_config
    model_specs = model_evaluation_in.model_specs

    # If a specific cluster was named, gate access through cluster visibility.
    cluster = None
    if model_evaluation_in.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model_evaluation_in.cluster_id)
        assert_cluster_visible(ctx, cluster, not_found_message="Cluster not found")

    # Stamp the owner the model would be created under (mirrors
    # create_model's resolution: current Org context, else the target
    # cluster's owner Org). Server-authoritative — a client-supplied
    # value is overwritten, so a caller can't probe another Org's
    # backend versions through evaluation results. With the owner set,
    # compatibility checks resolve the same Org-scoped backend versions
    # an actual deploy would.
    owner_principal_id = ctx.current_principal_id
    if owner_principal_id is None and cluster is not None:
        owner_principal_id = cluster.owner_principal_id
    for spec in model_specs:
        spec.owner_principal_id = owner_principal_id

    try:
        results = await evaluate_models(
            cluster_id=model_evaluation_in.cluster_id,
            config=config,
            session=session,
            model_specs=model_specs,
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to evaluate model compatibility: {e}"
        )

    return ModelEvaluationResponse(results=results)
