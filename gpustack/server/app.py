from functools import partial
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import aiohttp
from fastapi import FastAPI
from fastapi_cdn_host import patch_docs

from fastapi.middleware.cors import CORSMiddleware
from gpustack.api import exceptions, middlewares
from gpustack.api.auth import BearerTokenAuthenticator
from gpustack.config.config import Config
from gpustack import envs
from gpustack.routes import ui
from gpustack.routes.routes import api_router
from gpustack.utils.forwarded import ForwardedHostPortMiddleware
from gpustack.security import JWTManager
from gpustack.gateway.utils import worker_websocket_connect_callback
from gpustack.websocket_proxy.message_server import MessageServerHandler


from gpustack.extension import (
    Plugin,
    PluginContext,
    iter_plugin_classes,
    plugin_needs_instantiation,
    resolve_version_info,
)

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
        version=resolve_version_info()[0],
        docs_url=None if (cfg and cfg.disable_openapi_docs) else "/docs",
        redoc_url=None if (cfg and cfg.disable_openapi_docs) else "/redoc",
        openapi_url=None if (cfg and cfg.disable_openapi_docs) else "/openapi.json",
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

    app.state.jwt_manager = JWTManager(cfg.jwt_secret_key)
    app.state.websocket_authenticator = BearerTokenAuthenticator()

    # The product deployment has no federation peers: nothing ever
    # configures one, and the federation branch admits any
    # caller-supplied x-server-* identity. Keep the production endpoint
    # on worker client connections only.
    app.state.message_server_handler = MessageServerHandler(
        listen_address=cfg.get_proxy_listen_address(cfg.get_advertise_address()),
        listen_port=cfg.api_port,
        proxy_port=cfg.get_proxy_port(),
        authenticator=app.state.websocket_authenticator,
        server_info_getter=lambda websocket: None,
        callback_on_connect=partial(
            worker_websocket_connect_callback,
            proxy_address=cfg.get_proxy_url(),
        ),
        callback_on_disconnect=worker_websocket_connect_callback,
    )

    return app


def _load_extension_plugins(app: FastAPI, cfg: Config):
    """Load extension plugins registered via entry points.

    Every registered plugin gets a :class:`PluginContext` on
    ``app.state.plugin_contexts`` regardless of shape — the server
    reuses the contexts at startup for coordinator discovery and task
    collection.

    Two shapes are supported:

    * *static* (preferred): the entry point resolves to a
      ``Plugin`` subclass that does **not** override ``__init__``. It
      is never constructed; core calls its classmethod hooks with its
      context — ``migrate`` then ``setup`` here, the rest at server
      startup.
    * *instance* (legacy): the subclass overrides
      ``__init__(app, cfg)`` and does all of its wiring from the
      constructor. It is appended to ``app.state.extension_plugins``
      exactly as before and no classmethod hook is collected for it.

    Any exception from either shape refuses startup.
    """
    app.state.extension_plugins = []
    app.state.plugin_contexts = {}
    for name, plugin_class in iter_plugin_classes():
        try:
            if not (
                isinstance(plugin_class, type) and issubclass(plugin_class, Plugin)
            ):
                logger.warning(
                    f"Extension plugin {name} does not implement the Plugin interface."
                )
                continue

            # Two installed distributions can register the same
            # entry-point name; letting one overwrite the other would
            # silently drop its hooks — and even its coordinator,
            # falling the process back to LocalCoordinator.
            if name in app.state.plugin_contexts:
                raise RuntimeError(
                    f"Duplicate plugin entry-point name '{name}': another "
                    f"distribution already registered it. Remove one of the "
                    f"conflicting packages."
                )

            ctx = PluginContext(name=name, app=app, cfg=cfg, plugin_cls=plugin_class)
            app.state.plugin_contexts[name] = ctx

            if plugin_needs_instantiation(plugin_class):
                plugin = plugin_class(app, cfg)
                app.state.extension_plugins.append(plugin)
                # Record the instance on its context so the server's
                # dispatch loops stay in entry-point order across both
                # plugin shapes.
                ctx.instance = plugin
                logger.info(f"Loaded extension plugin: {name}")
            else:
                plugin_class.migrate(ctx)
                plugin_class.setup(ctx)
                logger.info(f"Loaded extension plugin: {name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load extension plugin '{name}': {e}") from e
