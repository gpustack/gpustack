from contextlib import asynccontextmanager
from importlib.metadata import entry_points
import logging
from pathlib import Path
import aiohttp
from fastapi import FastAPI
from fastapi_cdn_host import patch_docs

from gpustack import __version__
from gpustack.api import exceptions, middlewares
from gpustack.config.config import Config
from gpustack import envs
from gpustack.extension import Plugin
from gpustack.routes import ui
from gpustack.routes.routes import api_router
from gpustack.utils.forwarded import ForwardedHostPortMiddleware
from gpustack.gateway.plugins import register as register_gateway_plugins

logger = logging.getLogger(__name__)


def create_app(cfg: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        connector = aiohttp.TCPConnector(
            limit=envs.TCP_CONNECTOR_LIMIT,
            force_close=True,
        )
        app.state.http_client = aiohttp.ClientSession(
            connector=connector, trust_env=True
        )
        app.state.http_client_no_proxy = aiohttp.ClientSession(connector=connector)
        yield
        await app.state.http_client.close()
        await app.state.http_client_no_proxy.close()

    app = FastAPI(
        title="GPUStack",
        lifespan=lifespan,
        response_model_exclude_unset=True,
        version=__version__,
        docs_url=None if (cfg and cfg.disable_openapi_docs) else "/docs",
        redoc_url=None if (cfg and cfg.disable_openapi_docs) else "/redoc",
        openapi_url=None if (cfg and cfg.disable_openapi_docs) else "/openapi.json",
    )
    patch_docs(app, Path(__file__).parents[1] / "ui" / "static")
    app.add_middleware(ForwardedHostPortMiddleware)
    app.add_middleware(middlewares.RequestTimeMiddleware)
    app.add_middleware(middlewares.ModelUsageMiddleware)
    app.add_middleware(middlewares.RefreshTokenMiddleware)
    app.include_router(api_router)
    ui.register(app)
    register_gateway_plugins(cfg=cfg, app=app)
    _load_extension_plugins(app, cfg)
    exceptions.register_handlers(app)

    return app


def _load_extension_plugins(app: FastAPI, cfg: Config):
    """Load extension plugins registered via entry points."""
    app.state.extension_plugins = []
    eps = entry_points(group="gpustack.plugins")
    for ep in eps:
        try:
            plugin_factory = ep.load()
            plugin = plugin_factory()
            if not isinstance(plugin, Plugin):
                logger.warning(
                    f"Extension plugin {ep.name} does not implement "
                    "the Plugin interface."
                )
                continue

            plugin.register(app, cfg)
            app.state.extension_plugins.append(plugin)
            logger.info(f"Loaded extension plugin: {ep.name}")
        except Exception:
            logger.warning(
                f"Failed to load extension plugin: {ep.name}",
                exc_info=True,
            )
