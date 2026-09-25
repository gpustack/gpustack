from functools import partial
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import aiohttp
from fastapi import FastAPI
from fastapi_cdn_host import patch_docs

from gpustack import __version__
from fastapi.middleware.cors import CORSMiddleware
from gpustack.api import exceptions, middlewares
from gpustack.api.auth import BearerTokenAuthenticator
from gpustack.config.config import Config
from gpustack import envs
from gpustack.routes import ui
from gpustack.routes.routes import api_router
from gpustack.utils.base_path import BasePathMiddleware
from gpustack.utils.forwarded import ForwardedHostPortMiddleware
from gpustack.security import JWTManager
from gpustack.gateway.utils import worker_websocket_connect_callback
from gpustack.websocket_proxy.message_server import MessageServerHandler
from gpustack.extension import Plugin, iter_plugin_classes

logger = logging.getLogger(__name__)


def create_app(cfg: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.server_config = cfg
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
        # Tells FastAPI where it is mounted, so the URLs it generates for the
        # browser carry the prefix. Most visibly /docs, whose page asks for
        # /openapi.json: without this it asks at the origin root, which under a
        # subpath mount is the customer's own application rather than us. That is
        # two of the five extra `location` blocks #5270's reporter had to add.
        #
        # It also makes routing tolerate a proxy that does *not* strip the prefix
        # (AWS ALB cannot), because Starlette matches against the path with
        # root_path removed. Empty string is FastAPI's own default, so the root
        # deployment is unaffected.
        #
        # Paired with BasePathMiddleware below, which is what makes it safe for a
        # proxy that *does* strip.
        root_path=cfg.get_base_path() if cfg else "",
    )
    # Before patch_docs: it auto-mounts its own plain StaticFiles at /static
    # unless that path is already taken, and whichever mount lands first wins
    # every request under it. Registering ours first means the docs assets
    # (swagger-ui, redoc — the largest files in there) get served from the
    # build's precompressed .gz like the rest of the UI, instead of shadowing
    # our mount into dead code.
    ui.register(app)
    patch_docs(app, Path(__file__).parents[1] / "ui" / "static")
    app.add_middleware(
        ForwardedHostPortMiddleware, trusted_hosts=cfg.get_trusted_hosts()
    )
    app.add_middleware(middlewares.RequestTimeMiddleware)
    app.add_middleware(middlewares.ModelUsageMiddleware)
    app.add_middleware(middlewares.RefreshTokenMiddleware)
    if cfg.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.allow_origins,
            allow_credentials=cfg.allow_credentials,
            allow_methods=cfg.allow_methods,
            allow_headers=cfg.allow_headers,
        )
    app.include_router(api_router)
    _load_extension_plugins(app, cfg)
    exceptions.register_handlers(app)

    # Added last, so it runs first: add_middleware prepends, and this one has to
    # settle what `path` means before anything else reads it. Skipped entirely at
    # the root, where there is no prefix to restore.
    if base_path := app.root_path:
        app.add_middleware(BasePathMiddleware, prefix=base_path)

    app.state.jwt_manager = JWTManager(cfg.jwt_secret_key)
    app.state.websocket_authenticator = BearerTokenAuthenticator()
    app.state.message_server_handler = MessageServerHandler(
        listen_address=cfg.get_proxy_listen_address(cfg.get_advertise_address()),
        listen_port=cfg.api_port,
        proxy_port=cfg.get_proxy_port(),
        authenticator=app.state.websocket_authenticator,
        callback_on_connect=partial(
            worker_websocket_connect_callback,
            proxy_address=cfg.get_proxy_url(),
        ),
        callback_on_disconnect=worker_websocket_connect_callback,
    )

    return app


def _load_extension_plugins(app: FastAPI, cfg: Config):
    """Load extension plugins registered via entry points.

    Each entry point is expected to resolve to a ``Plugin`` subclass
    whose ``__init__(app, cfg)`` performs the full registration.
    """
    app.state.extension_plugins = []
    for name, plugin_class in iter_plugin_classes():
        try:
            if not (
                isinstance(plugin_class, type) and issubclass(plugin_class, Plugin)
            ):
                logger.warning(
                    f"Extension plugin {name} does not implement the Plugin interface."
                )
                continue
            plugin = plugin_class(app, cfg)
            app.state.extension_plugins.append(plugin)
            logger.info(f"Loaded extension plugin: {name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load extension plugin '{name}': {e}") from e
