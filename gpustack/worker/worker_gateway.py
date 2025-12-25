import asyncio
import logging
from typing import Optional
from aiocache import Cache

from gpustack.schemas.models import (
    ModelInstancePublic,
    ModelPublic,
)
from gpustack.schemas.config import (
    GatewayModeEnum,
    ModelInstanceProxyModeEnum,
)
from gpustack.config.config import Config
from gpustack.client import ClientSet
from gpustack.server.bus import Event, EventType
from gpustack.gateway.client.networking_higress_io_v1_api import (
    NetworkingHigressIoV1Api,
)
from gpustack.gateway import utils as mcp_handler
from gpustack.gateway import get_async_k8s_config
from gpustack.api.exceptions import NotFoundException

from kubernetes_asyncio.client import ApiException
from kubernetes_asyncio import client as k8s_client

logger = logging.getLogger(__name__)


class WorkerGatewayController:
    def __init__(
        self,
        worker_id: int,
        cluster_id: int,
        clientset: ClientSet,
        cfg: Config,
    ):
        self._worker_id = worker_id
        self._clientset = clientset
        self._config = cfg
        self._namespace = cfg.get_gateway_namespace()
        self._cluster_id = cluster_id
        self._async_k8s_config = get_async_k8s_config(cfg=cfg)
        self._lock = asyncio.Lock()
        self._model_cache = Cache(Cache.MEMORY)
        self._cache_synced = False
        self._disabled_gateway = cfg.gateway_mode == GatewayModeEnum.disabled

    def _disable_controller(self) -> bool:
        return (
            self._disabled_gateway
            or self._config.server_role() != Config.ServerRole.WORKER
            or self._config.proxy_mode != ModelInstanceProxyModeEnum.WORKER
        )

    async def wait_for_cache_sync(self):
        while not self._cache_synced:
            await asyncio.sleep(1)

    async def sync_model_cache(self):
        if self._disable_controller():
            return

        async def set_cache(event: Event):
            model = ModelPublic.model_validate(event.data)
            if event.type == EventType.DELETED:
                await self._model_cache.delete(model.id)
            else:
                await self._model_cache.set(model.id, model)

        current_models = self._clientset.models.list(
            params={"cluster_id": self._cluster_id}
        )
        for model in current_models.items:
            await self._model_cache.set(model.id, model)
        self._cache_synced = True
        while True:
            try:
                logger.debug("Started watching model instances.")
                await self._clientset.models.awatch(
                    callback=set_cache,
                    params={"cluster_id": self._cluster_id},
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to watch model instances: {e}")
                await asyncio.sleep(5)

    async def _get_model_with_cache(self, model_id: int) -> Optional[ModelPublic]:
        """Get model from cache or fetch from clientset."""
        if await self._model_cache.exists(model_id):
            return await self._model_cache.get(model_id)
        try:
            return self._clientset.models.get(model_id)
        except NotFoundException:
            return None
        except Exception as e:
            raise e

    async def _prerun(self):
        async with self._lock:
            if hasattr(self, "_networking_hisgress_api") and hasattr(
                self, "_networking_api"
            ):
                return
            api_client = k8s_client.ApiClient(configuration=self._async_k8s_config)
            self._networking_hisgress_api = NetworkingHigressIoV1Api(
                api_client=api_client
            )
            self._networking_api = k8s_client.NetworkingV1Api(api_client=api_client)
        await self.wait_for_cache_sync()
        if self._config.server_role() == Config.ServerRole.WORKER:
            model_ids = list(self._model_cache._cache.keys())
            await mcp_handler.cleanup_orphaned_model_ingresses(
                self._namespace,
                model_ids,
                self._async_k8s_config,
            )

    async def _handle_model_instance_event(self, event: Event):
        model_instance = ModelInstancePublic.model_validate(event.data)
        if model_instance.worker_id != self._worker_id:
            return
        desired_registry = await mcp_handler.ensure_model_instance_mcp_bridge(
            event.type,
            model_instance,
            self._networking_hisgress_api,
            self._namespace,
            self._cluster_id,
        )
        if self._config.server_role() == Config.ServerRole.WORKER:
            ingress_name = mcp_handler.model_ingress_name(model_instance.model_id)
            model = await self._get_model_with_cache(model_instance.model_id)
            # if model is None, it means the model has been deleted, wo need to delete the model ingress
            if model is None:
                try:
                    await self._networking_api.delete_namespaced_ingress(
                        name=ingress_name,
                        namespace=self._namespace,
                    )
                except ApiException as e:
                    if e.status != 404:
                        raise
            else:
                destinations = [(1, registry) for registry in desired_registry]
                mcp_handler.replace_registry_weight(destinations)
                await mcp_handler.ensure_model_ingress(
                    namespace=self._namespace,
                    destinations=destinations,
                    model=model,
                    event_type=event.type,
                    networking_api=self._networking_api,
                    hostname=None,
                    tls_secret_name=None,
                    included_generic_route=model.generic_proxy,
                    included_proxy_route=False,
                )

    async def start_model_instance_controller(self):
        if self._disable_controller():
            return
        await self._prerun()
        # Implementation of start method
        while True:
            try:
                logger.debug("Started watching model instances.")
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                logger.error(tb)
                logger.error(f"Failed to watch model instances: {e}")
                await asyncio.sleep(5)
