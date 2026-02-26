import logging
import math
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from urllib.parse import urlencode
from gpustack_runtime.detector import ManufacturerEnum
from sqlalchemy.orm import selectinload
from sqlmodel import and_, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from enum import Enum

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.inference_backend import is_custom_backend
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstancesPublic,
    BackendEnum,
    ModelListParams,
)
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import GPUDeviceStatus, Worker
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
)
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.server.services import (
    ModelService,
    WorkerService,
    revoke_model_access_cache,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id
from gpustack.routes.model_common import (
    build_category_conditions,
    categories_filter,
)
from gpustack.config.config import get_global_config
from gpustack.utils.grafana import resolve_grafana_base_url

router = APIRouter()

logger = logging.getLogger(__name__)


class ModelStateFilterEnum(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    STOPPED = "stopped"


@router.get("", response_model=ModelsPublic)
async def get_models(
    params: ModelListParams = Depends(),
    state: Optional[ModelStateFilterEnum] = Query(
        default=None,
        description="Filter by model state.",
    ),
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    cluster_id: int = None,
    backend: Optional[str] = Query(None, description="Filter by backend."),
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id

    if backend:
        fields["backend"] = backend

    if params.watch:
        return StreamingResponse(
            Model.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: categories_filter(data, categories),
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = []
        if categories:
            conditions = build_category_conditions(session, Model, categories)
            extra_conditions.append(or_(*conditions))

        if state is None:
            pass
        elif state == ModelStateFilterEnum.READY:
            extra_conditions.append(Model.ready_replicas > 0)
        elif state == ModelStateFilterEnum.NOT_READY:
            extra_conditions.append(and_(Model.ready_replicas == 0, Model.replicas > 0))
        elif state == ModelStateFilterEnum.STOPPED:
            extra_conditions.append(Model.replicas == 0)

        order_by = params.order_by
        if order_by:
            # When sorting by "source", add additional sorting fields for deterministic ordering
            new_order_by = []
            for field, direction in order_by:
                new_order_by.append((field, direction))
                if field == "source":
                    new_order_by.append(("huggingface_repo_id", direction))
                    new_order_by.append(("huggingface_filename", direction))
                    new_order_by.append(("model_scope_model_id", direction))
                    new_order_by.append(("model_scope_file_path", direction))
                    new_order_by.append(("local_path", direction))
            order_by = new_order_by

        return await Model.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            fields=fields,
            order_by=order_by,
        )


@router.get("/{id}", response_model=ModelPublic)
async def get_model(
    session: SessionDep,
    id: int,
):
    return await _get_model(session=session, id=id)


@router.get("/{id}/dashboard")
async def get_model_dashboard(
    session: SessionDep,
    id: int,
    request: Request,
):
    model = await _get_model(session=session, id=id)

    cfg = get_global_config()
    if not cfg.get_grafana_url() or not cfg.grafana_model_dashboard_uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if model.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model.cluster_id)

    query_params = {}
    if cluster is not None:
        query_params["var-cluster_name"] = cluster.name
    query_params["var-model_name"] = model.name

    grafana_base = resolve_grafana_base_url(cfg, request)
    slug = "gpustack-model"
    dashboard_url = f"{grafana_base}/d/{cfg.grafana_model_dashboard_uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


async def _get_model(
    session: SessionDep,
    id: int,
):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return model


@router.get("/{id}/instances", response_model=ModelInstancesPublic)
async def get_model_instances(id: int, params: ListParamsDep):
    if params.watch:
        fields = {"model_id": id}
        return StreamingResponse(
            ModelInstance.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        model = await Model.one_by_id(
            session, id, options=[selectinload(Model.instances)]
        )
        if not model:
            raise NotFoundException(message="Model not found")

        instances = model.instances
        count = len(instances)
        total_page = math.ceil(count / params.perPage)
        pagination = Pagination(
            page=params.page,
            perPage=params.perPage,
            total=count,
            totalPage=total_page,
        )

        return ModelInstancesPublic(items=instances, pagination=pagination)


async def validate_model_in(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    if model_in.gpu_selector is not None and model_in.replicas > 0:
        await validate_gpu_ids(session, model_in, cluster_id=cluster_id)

    if model_in.backend_parameters:
        param_gpu_layers = find_parameter(
            model_in.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
        )

        if param_gpu_layers:
            int_param_gpu_layers = safe_int(param_gpu_layers, None)
            if (
                not param_gpu_layers.isdigit()
                or int_param_gpu_layers < 0
                or int_param_gpu_layers > 999
            ):
                raise BadRequestException(
                    message="Invalid backend parameter --gpu-layers. Please provide an integer in the range 0-999 (inclusive)."
                )

            if (
                int_param_gpu_layers == 0
                and model_in.gpu_selector is not None
                and len(model_in.gpu_selector.gpu_ids) > 0
            ):
                raise BadRequestException(
                    message="Cannot set --gpu-layers to 0 and manually select GPUs at the same time. Setting --gpu-layers to 0 means running on CPU only."
                )

        param_port = find_parameter(model_in.backend_parameters, ["port"])

        if param_port:
            raise BadRequestException(
                message="Setting the port using --port is not supported."
            )


async def validate_gpu_ids(  # noqa: C901
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    effective_cluster_id = (
        cluster_id if cluster_id is not None else getattr(model_in, "cluster_id", None)
    )

    if (
        model_in.gpu_selector
        and model_in.gpu_selector.gpu_ids
        and model_in.gpu_selector.gpus_per_replica
    ):
        if len(model_in.gpu_selector.gpu_ids) < model_in.gpu_selector.gpus_per_replica:
            raise BadRequestException(
                message="The number of selected GPUs must be greater than or equal to gpus_per_replica."
            )

    model_backend = model_in.backend

    if model_backend == BackendEnum.VOX_BOX and (
        len(model_in.gpu_selector.gpu_ids) > 1
        or (
            model_in.gpu_selector.gpus_per_replica is not None
            and model_in.gpu_selector.gpus_per_replica > 1
        )
    ):
        raise BadRequestException(
            message="The vox-box backend is restricted to execution on a single NVIDIA GPU."
        )

    worker_name_set = set()
    for gpu_id in model_in.gpu_selector.gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise BadRequestException(message=f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        gpu_index = safe_int(matched.get("gpu_index"), -1)
        worker_name_set.add(worker_name)

        if effective_cluster_id is None:
            raise BadRequestException(
                message=f"A cluster context is required for manual GPU selection, but was not provided. Cannot validate worker '{worker_name}'."
            )

        worker = await WorkerService(session).get_by_cluster_id_name(
            effective_cluster_id, worker_name
        )
        if not worker:
            raise BadRequestException(message=f"Worker {worker_name} not found")

        gpu = (
            next(
                (gpu for gpu in worker.status.gpu_devices if gpu.index == gpu_index),
                None,
            )
            if worker.status and worker.status.gpu_devices
            else None
        )
        if gpu:
            validate_gpu(gpu, model_backend=model_backend)

        if model_backend == BackendEnum.VLLM and len(worker_name_set) > 1:
            await validate_distributed_vllm_limit_per_worker(session, model_in, worker)

    if (
        is_custom_backend(model_backend)
        and len(worker_name_set) > 1
        and model_in.replicas == 1
    ):
        raise BadRequestException(
            message="Distributed inference across multiple workers is not supported for custom backends."
        )


def validate_gpu(gpu_device: GPUDeviceStatus, model_backend: str = ""):
    if (
        model_backend == BackendEnum.VOX_BOX
        and gpu_device.vendor != ManufacturerEnum.NVIDIA.value
    ):
        raise BadRequestException(
            "The vox-box backend is supported only on NVIDIA GPUs."
        )

    if (
        model_backend == BackendEnum.ASCEND_MINDIE
        and gpu_device.vendor != ManufacturerEnum.ASCEND.value
    ):
        raise BadRequestException(
            f"Ascend MindIE backend requires Ascend NPUs. Selected {gpu_device.vendor} GPU is not supported."
        )


async def validate_distributed_vllm_limit_per_worker(
    session: AsyncSession, model: Union[ModelCreate, ModelUpdate], worker: Worker
):
    """
    Validate that there is no more than one distributed vLLM instance per worker.
    """
    instances = await ModelInstance.all_by_field(session, "worker_id", worker.id)
    for instance in instances:
        if (
            instance.distributed_servers
            and instance.distributed_servers.subordinate_workers
            and instance.model_name != model.name
        ):
            raise BadRequestException(
                message=f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
            )


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    existing = await Model.one_by_field(session, "name", model_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Model '{model_in.name}' already exists. "
            "Please choose a different name or check the existing model."
        )
    should_create_route = (
        model_in.enable_model_route is not None and model_in.enable_model_route
    )
    if should_create_route:
        existing_route = await ModelRoute.one_by_field(session, "name", model_in.name)
        if existing_route:
            raise AlreadyExistsException(
                message=f"Model route '{model_in.name}' already exists. "
                "Please choose a different name or check the existing model route."
            )
    await validate_model_in(session, model_in)
    model_in_dict = model_in.model_dump(exclude={"enable_model_route"})

    try:
        model: Model = await Model.create(
            session, source=model_in_dict, auto_commit=(not should_create_route)
        )
        if should_create_route:
            model_route = ModelRoute(
                name=model.name,
                description=model.description,
                categories=model.categories,
                generic_proxy=model.generic_proxy,
                created_by_model=True,
                access_policy=model.access_policy,
            )
            model_route: ModelRoute = await ModelRoute.create(
                session, source=model_route, auto_commit=False
            )
            model_route_target = ModelRouteTarget(
                name=f"{model.name}-deployment",
                route_name=model_route.name,
                generic_proxy=model.generic_proxy,
                model_route=model_route,
                model=model,
                weight=100,
                state=TargetStateEnum.UNAVAILABLE,
            )
            await ModelRouteTarget.create(
                session,
                source=model_route_target,
                auto_commit=False,
            )
            await session.commit()
            await revoke_model_access_cache(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


@router.put("/{id}", response_model=ModelPublic)
async def update_model(session: SessionDep, id: int, model_in: ModelUpdate):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    await validate_model_in(session, model_in)

    if model_in.backend != BackendEnum.CUSTOM.value and (
        model.run_command or model.image_name
    ):
        patch = model_in.model_dump(exclude_unset=True)
        patch["run_command"] = None
        patch["image_name"] = None
        model_in = patch

    try:
        await ModelService(session).update(model, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return model


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = await Model.one_by_id(
        session,
        id,
        options=[
            selectinload(Model.instances),
            selectinload(Model.model_route_targets),
        ],
    )
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        await ModelService(session).delete(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
