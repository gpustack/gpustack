from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx

from gpustack.api import exceptions, middlewares
from gpustack.routes import ui
from gpustack.routes.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="GPUStack", lifespan=lifespan, response_model_exclude_unset=True)
app.add_middleware(middlewares.ModelUsageMiddleware)
app.add_middleware(middlewares.RefreshTokenMiddleware)
app.include_router(api_router)
ui.register(app)
exceptions.register_handlers(app)
