from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import HTTPException, Request
import httpx

from ..core.deps import SessionDep
from ..logging import logger
from ..utils import normalize_route_path
from ..schemas.models import Model

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completion(session: SessionDep, request: Request):
    body = await request.json()
    # 获取model字段
    model_name = body.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' field")

    model = Model.one_by_field(session=session, field="name", value=model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    stream = body.get("stream", False)

    url = f"http://localhost:8000{normalize_route_path(model_name)}"

    logger.info(f"proxying to {url}")
    # 使用 httpx 代理请求
    async with httpx.AsyncClient() as client:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() != "content-length" and key.lower() != "host"
        }

        timeout = 120

        if stream:
            # 流式代理
            resp = await client.stream(
                method=request.method,
                url=url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            return StreamingResponse(resp.aiter_raw(), headers=dict(resp.headers))
        else:
            # 常规代理
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            return JSONResponse(content=resp.json(), headers=dict(resp.headers))
