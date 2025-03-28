from fastapi import APIRouter, Request

from gpustack.config.config import Config
from gpustack.scheduler.evaluator import evaluate_model
from gpustack.schemas.model_evaluations import (
    ModelEvaluationRequest,
    ModelEvaluationResponse,
)

from gpustack.api.exceptions import (
    InternalServerErrorException,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep


router = APIRouter()


@router.post("", response_model=ModelEvaluationResponse)
async def create_model_evaluation(
    request: Request, session: SessionDep, model_evaluation_in: ModelEvaluationRequest
):
    config: Config = request.app.state.server_config
    model_specs = model_evaluation_in.model_specs

    try:
        results = []
        workers = await Worker.all(session)
        for model in model_specs:
            result = await evaluate_model(config, model, workers)
            results.append(result)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to evaluate model compatibility: {e}"
        )

    return ModelEvaluationResponse(results=results)
