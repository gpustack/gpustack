import asyncio
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Set, Tuple
from aiocache import Cache, BaseCache
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload

from gpustack.api.exceptions import InternalServerErrorException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
    ModelRouteTarget,
    TargetStateEnum,
    AccessPolicyEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import Worker
from gpustack.server.usage_buffer import usage_flush_buffer

logger = logging.getLogger(__name__)
cache = Cache(Cache.MEMORY)


def build_cache_key(func: Callable, *args, **kwargs):
    if kwargs is None:
        kwargs = {}
    ordered_kwargs = sorted(kwargs.items())
    return func.__qualname__ + str(args) + str(ordered_kwargs)


async def delete_cache_by_key(func, *args, **kwargs):
    key = build_cache_key(func, *args, **kwargs)
    logger.trace(f"Deleting cache for key: {key}")
    await cache.delete(key)


async def set_cache_by_key(key: str, value: Any):
    logger.trace(f"Set cache for key: {key}")
    await cache.set(key, value)


_cache_locks: Dict[str, asyncio.Lock] = {}


class locked_cached:
    def __init__(self, ttl: int = 30, cache: BaseCache = cache):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            return await self.decorator(f, *args, **kwargs)

        wrapper.cache = self.cache
        return wrapper

    async def get_from_cache(self, key: str):
        return await self.cache.get(key)

    async def set_in_cache(self, key: str, value: Any):
        await self.cache.set(key, value, ttl=self.ttl)

    async def decorator(self, f, *args, **kwargs):
        # no self arg
        key = build_cache_key(f, *args[1:], **kwargs)
        value = await self.get_from_cache(key)
        if value is not None:
            return value

        lock = _cache_locks.setdefault(key, asyncio.Lock())

        async with lock:
            value = await self.get_from_cache(key)
            if value is not None:
                return value

            logger.trace(f"cache miss for key: {key}")
            result = await f(*args, **kwargs)

            await self.set_in_cache(key, result)

        return result


class UserService:

    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await User.one_by_id(
            self.session,
            user_id,
            options=[selectinload(User.worker), selectinload(User.cluster)],
        )
        if result is None:
            return None
        if result.worker is not None:
            # detach worker to avoid lazy loading
            self.session.expunge(result.worker)
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_username(self, username: str) -> Optional[User]:
        result = await User.one_by_field(self.session, "username", username)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, user: User):
        return await User.create(self.session, user)

    async def update(self, user: User, source: Union[dict, SQLModel, None] = None):
        result = await user.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        return result

    async def delete(self, user: User):
        apikeys = await APIKeyService(self.session).get_by_user_id(user.id)
        result = await user.delete(self.session)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        for apikey in apikeys:
            await delete_cache_by_key(
                APIKeyService.get_by_access_key, apikey.access_key
            )
        return result

    async def model_allowed_for_user(
        self, model_name: str, user_id: int, api_key: Optional[ApiKey]
    ) -> bool:
        limited_model_names: Optional[Set[str]] = (
            set(api_key.allowed_model_names)
            if api_key is not None
            and api_key.allowed_model_names is not None
            and len(api_key.allowed_model_names) > 0
            else None
        )
        accessible_model_names: Set[str] = await self.get_user_accessible_model_names(
            user_id
        )
        return model_name in intersection_nullable_set(
            accessible_model_names, limited_model_names
        )

    @locked_cached(ttl=60)
    async def get_user_accessible_model_names(self, user_id: int) -> Set[str]:
        # Get all accessible model names for the user
        user: User = await self.get_by_id(user_id)
        if user is None:
            return set()
        if user.is_admin or user.is_system:
            all_models = await ModelRoute.all_by_field(self.session, "deleted_at", None)
            model_names = {model.name for model in all_models}
        else:
            allowed_models = await MyModel.all_by_fields(
                self.session, {"user_id": user.id, "deleted_at": None}
            )
            model_names = {model.name for model in allowed_models}
        return model_names


class APIKeyService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_access_key(self, access_key: str) -> Optional[ApiKey]:
        result = await ApiKey.one_by_field(self.session, "access_key", access_key)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def get_by_user_id(self, user_id: int) -> List[ApiKey]:
        results = await ApiKey.all_by_field(self.session, "user_id", user_id)
        if results is None:
            return []
        for result in results:
            self.session.expunge(result)
        return results

    async def delete(self, api_key: ApiKey):
        result = await api_key.delete(self.session)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result


class WorkerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, worker_id: int) -> Optional[Worker]:
        result = await Worker.one_by_id(self.session, worker_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_name(self, name: str) -> Optional[Worker]:
        result = await Worker.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(
        self, worker: Worker, source: Union[dict, SQLModel, None] = None, **kwargs
    ):
        result = await worker.update(self.session, source, **kwargs)
        await delete_cache_by_key(self.get_by_id, worker.id)
        await delete_cache_by_key(self.get_by_name, worker.name)
        return result

    async def delete(self, worker: Worker, **kwargs):
        worker_id = worker.id
        worker_name = worker.name
        result = await worker.delete(self.session, **kwargs)
        await delete_cache_by_key(self.get_by_id, worker_id)
        await delete_cache_by_key(self.get_by_name, worker_name)
        return result


class ModelRouteService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_name(self, name: str) -> Optional[ModelRoute]:
        result = await ModelRoute.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_model_auth_info_by_name(
        self, name: str
    ) -> Optional[Tuple[AccessPolicyEnum, str]]:
        route = await ModelRoute.one_by_field(self.session, "name", name)
        if route is None:
            return None
        route_targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields={"route_id": route.id},
        )
        if len(route_targets) == 0:
            return None
        models = await Model.all_by_fields(
            session=self.session,
            fields={},
            extra_conditions=[
                Model.id.in_(
                    [e.model_id for e in route_targets if e.model_id is not None]
                )
            ],
        )
        # set a default static token to avoid empty token response for public maas model route
        registration_token = "static_token_not_found"
        for model in models:
            cluster = await Cluster.one_by_id(self.session, model.cluster_id)
            if cluster.registration_token is not None:
                registration_token = cluster.registration_token
                break

        return route.access_policy, registration_token

    @locked_cached(ttl=60)
    async def get_model_ids_by_model_route_name(self, name: str) -> List[Model]:
        route_targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields={
                "route_name": name,
                "state": TargetStateEnum.ACTIVE,
                "deleted_at": None,
            },
            options=[selectinload(ModelRouteTarget.model)],
        )
        return [target.model for target in route_targets if target.model is not None]

    async def update(
        self,
        model_route: ModelRoute,
        source: Union[dict, SQLModel, None] = None,
        auto_commit: bool = True,
    ):
        result = await model_route.update(self.session, source, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_model_auth_info_by_name, model_route.name)
        await delete_cache_by_key(
            self.get_model_ids_by_model_route_name, model_route.name
        )
        return result

    async def delete(self, model_route: ModelRoute, auto_commit: bool = True):
        result = await model_route.delete(self.session, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_model_auth_info_by_name, model_route.name)
        await delete_cache_by_key(
            self.get_model_ids_by_model_route_name, model_route.name
        )
        return result


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, model_id: int) -> Optional[Model]:
        result = await Model.one_by_id(self.session, model_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_name(self, name: str) -> Optional[Model]:
        result = await Model.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(self, model: Model, source: Union[dict, SQLModel, None] = None):
        result = await model.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result

    async def delete(self, model: Model):
        result = await model.delete(self.session)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_running_instances(self, model_id: int) -> List[ModelInstance]:
        results = await ModelInstance.all_by_fields(
            self.session,
            fields={"model_id": model_id, "state": ModelInstanceStateEnum.RUNNING},
        )
        if results is None:
            return []

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_instance):
        result = await ModelInstance.create(self.session, model_instance)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        result = await model_instance.update(self.session, source)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def delete(self, model_instance: ModelInstance):
        result = await model_instance.delete(self.session)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def batch_delete(self, model_instances: List[ModelInstance]):
        if not model_instances:
            return []

        names = [mi.name for mi in model_instances]
        ids = set()
        try:
            for m in model_instances:
                await m.delete(self.session, auto_commit=False)
                ids.add(m.model_id)
            await self.session.commit()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to delete model instances {names}: {e}"
            )

    async def batch_update(
        self,
        model_instances: List[ModelInstance],
        source: Union[dict, SQLModel, None] = None,
    ):
        names = [mi.name for mi in model_instances]
        ids = set()
        try:
            for m in model_instances:
                await m.update(self.session, source, auto_commit=False)
                ids.add(m.model_id)
            await self.session.commit()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to update model instances {names}: {e}"
            )


class ModelUsageService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_fields(self, fields: dict) -> ModelUsage:
        result = await ModelUsage.one_by_fields(
            self.session,
            fields=fields,
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, model_usage: ModelUsage):
        return await ModelUsage.create(self.session, model_usage)

    async def update(
        self,
        model_usage: ModelUsage,
        completion_token_count: int,
        prompt_token_count: int,
    ):
        model_usage.completion_token_count += completion_token_count
        model_usage.prompt_token_count += prompt_token_count
        model_usage.request_count += 1

        key = build_cache_key(
            self.get_by_fields,
            model_usage.user_id,
            model_usage.model_id,
            model_usage.operation,
            model_usage.date,
        )
        await set_cache_by_key(key, model_usage)
        usage_flush_buffer[key] = model_usage
        return model_usage


class ModelFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_resolved_path(self, path: str) -> List[ModelFile]:
        results = await ModelFile.all_by_fields(
            self.session,
        )
        filtered_results = []
        for result in results:
            self.session.expunge(result)
            if path in result.resolved_paths:
                filtered_results.append(result)

        return filtered_results

    async def get_by_source_index(self, source_index: str) -> List[ModelFile]:
        results = await ModelFile.all_by_field(
            self.session, "source_index", source_index
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_file: ModelFile):
        return await ModelFile.create(self.session, model_file)


def intersection_nullable_set(set1: Set[str], set2: Optional[Set[str]]) -> Set[str]:
    if set2 is None:
        return set1
    return set1.intersection(set2)


async def delete_accessible_model_cache(
    session: AsyncSession,
    *user_ids: int,
):
    for user_id in user_ids:
        await delete_cache_by_key(
            UserService(session).get_user_accessible_model_names, user_id
        )
