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
    if model_evaluation_in.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model_evaluation_in.cluster_id)
        assert_cluster_visible(ctx, cluster, not_found_message="Cluster not found")

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
