from typing import Optional, Set
from fastapi import APIRouter, Request, Response
from gpustack.api.exceptions import InvalidException, ForbiddenException
from gpustack.schemas.api_keys import ApiKey
from gpustack.server.deps import CurrentUserDep

router = APIRouter()


@router.post("")
async def token_auth(
    request: Request,
    user: CurrentUserDep,
):
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is None:
        raise InvalidException(message="Missing X-Higress-Llm-Model header")
    api_key: Optional[ApiKey] = getattr(request.state, "api_key", None)
    access_key = None if api_key is None else api_key.access_key
    allowed_model_names: Set[str] = getattr(
        request.state, "user_allow_model_names", set()
    )
    if model_name not in allowed_model_names:
        raise ForbiddenException(
            message=f"Api key not allowed to access model {model_name}"
        )
    headers = {
        "X-Mse-Consumer": f"gpustack-{user.id}",
    }
    if access_key is not None:
        headers["X-GPUStack-Api-Key"] = access_key
    return Response(
        status_code=200,
        headers=headers,
    )
