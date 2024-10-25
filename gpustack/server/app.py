from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_cdn_host import monkey_patch_for_docs_ui
import httpx

from gpustack import __version__
from gpustack.api import exceptions, middlewares
from gpustack.routes import ui
from gpustack.routes.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="GPUStack",
    lifespan=lifespan,
    response_model_exclude_unset=True,
    version=__version__,
)
monkey_patch_for_docs_ui(app)
app.add_middleware(middlewares.RequestTimeMiddleware)
app.add_middleware(middlewares.ModelUsageMiddleware)
app.add_middleware(middlewares.RefreshTokenMiddleware)
app.include_router(api_router)
ui.register(app)
exceptions.register_handlers(app)
