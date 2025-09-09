import logging
from typing import List, Tuple, Optional

import yaml
from fastapi import APIRouter, HTTPException, Body
from sqlalchemy import or_, func
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas import Worker
from gpustack.schemas.common import Pagination
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    InferenceBackendCreate,
    InferenceBackendListItem,
    InferenceBackendResponse,
    InferenceBackendUpdate,
    InferenceBackendsPublic,
    VersionConfig,
    VersionConfigDict,
    get_build_in_backend,
    get_build_in_backend_show_name,
    InferenceBackendPublic,
)
from gpustack.schemas.models import BackendEnum
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack_runner import list_service_runners

logger = logging.getLogger(__name__)
router = APIRouter()


def is_build_in_backend(backend_name: str) -> bool:
    """
    Check if a backend is a build-in backend.

    Args:
        backend_name: The name of the backend to check

    Returns:
        True if the backend is build-in, False otherwise
    """
    build_in_backends = get_build_in_backend()
    build_in_backend_names = {backend.backend_name for backend in build_in_backends}
    return backend_name in build_in_backend_names


def get_runner_versions_and_configs(
    backend_name: str,
) -> Tuple[List[str], VersionConfigDict, Optional[str]]:
    """
    Get runner versions and version configs for a given backend.

    Args:
        backend_name: The name of the backend service

    Returns:
        A tuple containing:
        - List of version strings
        - VersionConfigDict with version configurations
        - Default version (first available version or None)
    """
    runners_list = list_service_runners(service=backend_name)
    versions = []
    version_configs = VersionConfigDict()
    default_version = None

    if runners_list and len(runners_list) > 0:
        for version in runners_list[0].versions:
            if version.version:
                versions.append(version.version)
                backend_list = [
                    f"{backend_runner.backend}" for backend_runner in version.backends
                ]
                version_configs.root[version.version] = VersionConfig(
                    image_name=backend_name,
                    build_in_frameworks=backend_list,
                )
                if default_version is None:
                    default_version = version.version

    return versions, version_configs, default_version


@router.get("/list", response_model=InferenceBackendResponse)
async def list_backend_configs(session: SessionDep):
    """
    Get list of available backend configurations with version information.

    Returns both build-in backends and custom backends from database.
    Build-in backends are identified and enhanced with runner versions.
    Each backend item includes available versions.
    """
    items = []

    workers = await Worker.all(session)
    framework_list = set()
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                framework_list.add(gpu.runtime_framework)

    # Process all backends from database (includes both build-in and custom backends)
    try:
        inference_backends = await InferenceBackend.all(session)
        for backend in inference_backends:
            # Get versions from version_config
            versions = []
            if backend.version_configs and backend.version_configs.root:
                versions = list(backend.version_configs.root.keys())

            if backend.is_build_in:
                # For build-in backends, add runner versions and use special show name
                _, version_configs, default_version = get_runner_versions_and_configs(
                    backend.backend_name
                )
                # Merge runner versions with existing versions
                for version, config in version_configs.root.items():
                    filtered_frameworks = [
                        framework
                        for framework in config.build_in_frameworks
                        if framework in framework_list
                    ]
                    if filtered_frameworks:
                        versions.append(version)

                # Remove duplicates while preserving order
                versions = list(dict.fromkeys(versions))

                # Update default version if found from runner
                if default_version and not backend.default_version:
                    backend.default_version = default_version

                backend_item = InferenceBackendListItem(
                    backend_name=backend.backend_name,
                    backend_show_name=get_build_in_backend_show_name(
                        backend.backend_name
                    ),
                    default_version=backend.default_version,
                    default_backend_param=backend.default_backend_param,
                    versions=versions,
                    is_build_in=backend.is_build_in,
                )
            else:
                # For custom backends, use backend_name as show_name
                backend_item = InferenceBackendListItem(
                    backend_name=backend.backend_name,
                    backend_show_name=backend.backend_name,
                    default_version=backend.default_version,
                    default_backend_param=backend.default_backend_param,
                    versions=versions,
                    is_build_in=False,
                )

            items.append(backend_item)

        # Ensure Custom backend is always included even if not in database
        custom_backend_item = InferenceBackendListItem(
            backend_name=BackendEnum.CUSTOM,
            backend_show_name=get_build_in_backend_show_name(BackendEnum.CUSTOM),
            default_version=None,
            default_backend_param=None,
            versions=[],
            is_build_in=False,
        )
        items.append(custom_backend_item)

    except Exception as e:
        # Log error but don't fail the entire request
        logger.error(f"Failed to load backends from database: {e}")

    return InferenceBackendResponse(items=items)


async def merge_runner_versions_to_db(
    session: SessionDep,
) -> List[InferenceBackendPublic]:
    # Get database backends first
    db_result = await InferenceBackend.all(session)

    # Create a map of database backends by name for easy lookup
    db_backends_map = {
        backend.backend_name: InferenceBackendPublic(**backend.model_dump())
        for backend in db_result
    }

    # Ensure all BUILD_IN_BACKENDS are included
    merged_backends = []
    build_in_backend_names = set()

    for build_in_backend in get_build_in_backend():
        if build_in_backend.backend_name == BackendEnum.CUSTOM:
            continue
        build_in_backend_names.add(build_in_backend.backend_name)

        # Get versions from list_service_runners using the common function
        _, runner_versions, default_version = get_runner_versions_and_configs(
            build_in_backend.backend_name
        )

        if default_version and not build_in_backend.default_version:
            build_in_backend.default_version = default_version

        db_backend = db_backends_map[build_in_backend.backend_name]
        if not db_backend:
            logger.warning(
                f"No database backend found for {build_in_backend.backend_name}"
            )
            continue
        # Merge versions from database backend.
        for runner_version, version_config in runner_versions.root.items():
            db_backend.build_in_version_configs[runner_version] = version_config

        if default_version and not db_backend.default_version:
            db_backend.default_version = default_version

        merged_backends.append(db_backend)

    # Add remaining database backends that are not in BUILD_IN_BACKENDS
    for backend_name, db_backend in db_backends_map.items():
        if backend_name not in build_in_backend_names:
            merged_backends.append(db_backend)

    return merged_backends


@router.get("", response_model=InferenceBackendsPublic)
async def get_inference_backends(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
):
    """
    Get paginated list of inference backends with optional search.
    """
    fields = {}

    extra_conditions = []
    if search:
        lower_search = search.lower()
        extra_conditions.append(
            or_(
                func.lower(InferenceBackend.backend_name).like(f"%{lower_search}%"),
                func.lower(InferenceBackend.description).like(f"%{lower_search}%"),
            )
        )

    if params.watch:
        return StreamingResponse(
            InferenceBackend.streaming(
                engine,
                fields=fields,
            ),
            media_type="text/event-stream",
        )

    merged_backends = await merge_runner_versions_to_db(session)
    filter_backends = []
    workers = await Worker.all(session)
    framework_list = set()
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                framework_list.add(gpu.runtime_framework)

    for backend in merged_backends:
        filtered_version_configs = {}
        for version, config in backend.build_in_version_configs.items():
            filtered_frameworks = [
                framework
                for framework in config.build_in_frameworks
                if framework in framework_list
            ]
            if filtered_frameworks:
                config.build_in_frameworks = filtered_frameworks
                filtered_version_configs[version] = config

        backend.build_in_version_configs = filtered_version_configs
        filter_backends.append(backend)

    # Apply search filter to merged backends if search is provided
    if search:
        lower_search = search.lower()
        search_filter_backends = []
        for backend in filter_backends:
            if lower_search in backend.backend_name.lower() or (
                backend.description and lower_search in backend.description.lower()
            ):
                search_filter_backends.append(backend)
        filter_backends = search_filter_backends

    # Apply pagination to merged results
    total = len(filter_backends)
    start_idx = (params.page - 1) * params.perPage
    end_idx = start_idx + params.perPage
    paginated_backends = filter_backends[start_idx:end_idx]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=total,
        totalPage=max(total / params.perPage, 1),
    )

    # Create the response with the same structure as the original
    return InferenceBackendsPublic(
        items=paginated_backends,
        pagination=pagination,
    )


@router.get("/all", response_model=List[InferenceBackend])
async def get_all_inference_backends(
    session: SessionDep,
):
    backends = await merge_runner_versions_to_db(session)
    ret = []
    for backend in backends:
        if not backend.is_build_in:
            ret.append(backend)
            continue
        for build_in_version, config in backend.build_in_version_configs.items():
            # if version in same, db version first
            if build_in_version not in backend.version_configs.root:
                backend.version_configs.root[build_in_version] = config

        ret.append(backend)

    return ret


@router.get("/{id}", response_model=InferenceBackend)
async def get_inference_backend(session: SessionDep, id: int):
    """
    Get a specific inference backend by ID.
    """
    backend = await InferenceBackend.one_by_id(session, id)
    if not backend:
        raise HTTPException(status_code=400, detail=f"Inference backend {id} not found")
    return backend


@router.get("/backend_name/{backend_name}", response_model=InferenceBackend)
async def get_inference_backend_by_name(session: SessionDep, backend_name: str):
    """
    Get a specific inference backend by backend name.
    """
    backend = await InferenceBackend.one_by_field(session, "backend_name", backend_name)
    if not backend:
        raise HTTPException(status_code=400, detail=f"Inference backend {id} not found")
    return backend


@router.post("", response_model=InferenceBackend)
async def create_inference_backend(
    session: SessionDep, backend_in: InferenceBackendCreate
):
    """
    Create a new inference backend.
    """
    # Check if backend with same name already exists
    existing = await InferenceBackend.one_by_field(
        session, "backend_name", backend_in.backend_name
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Inference backend with name '{backend_in.backend_name}' already exists",
        )

    # Validate that build-in backends cannot have default_version set
    if is_build_in_backend(backend_in.backend_name) and backend_in.default_version:
        raise HTTPException(
            status_code=400,
            detail=f"Build-in backend '{backend_in.backend_name}' cannot have default_version set. Default version is managed automatically.",
        )

    try:
        backend = InferenceBackend(**backend_in.model_dump())
        backend = await InferenceBackend.create(session, backend)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create inference backend: {e}"
        )

    return backend


@router.put("/{id}", response_model=InferenceBackend)
async def update_inference_backend(
    session: SessionDep, id: int, backend_in: InferenceBackendUpdate
):
    """
    Update an existing inference backend.
    """
    backend = await InferenceBackend.one_by_id(session, id)
    if not backend:
        raise NotFoundException(message=f"Inference backend {id} not found")

    # Check if updating to a name that already exists (excluding current backend)
    if backend_in.backend_name != backend.backend_name:
        raise HTTPException(
            status_code=400,
            detail="The name of inference-backend can not be modified",
        )

    # Validate that build-in backends cannot have default_version set
    if is_build_in_backend(backend.backend_name) and backend_in.default_version:
        raise HTTPException(
            status_code=400,
            detail=f"Build-in backend '{backend.backend_name}' cannot have default_version set. Default version is managed automatically.",
        )

    try:
        await backend.update(session, backend_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update inference backend: {e}"
        )

    return backend


@router.delete("/{id}")
async def delete_inference_backend(session: SessionDep, id: int):
    """
    Delete an inference backend.
    """
    backend = await InferenceBackend.one_by_id(session, id)
    if not backend:
        raise NotFoundException(message=f"Inference backend {id} not found")

    try:
        await backend.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete inference backend: {e}"
        )


@router.post("/from-yaml", response_model=InferenceBackend)
async def create_inference_backend_from_yaml(
    session: SessionDep, payload: dict = Body(...)
):
    """
    Create an inference backend from YAML configuration.

    Expected YAML format:
    ```yaml
    backend_name: "my-custom-backend"
    version_configs:
      "v1.0.0":
        image_name: "my-backend:v1.0.0"
        run_command: "python server.py --port {{port}} --model {{model_path}}"
      "v1.1.0":
        image_name: "my-backend:v1.1.0"
        run_command: "python server.py --port {{port}} --model {{model_path}}"
    default_version: "v1.1.0"
    default_backend_param: ["--max-tokens", "2048"]
    default_run_command: "python server.py"
    description: "My custom inference backend"
    health_check_path: "/health"
    allowed_proxy_uris: ["/v1/chat/completions", "/v1/completions"]
    ```
    """
    try:
        # Extract YAML content from JSON payload
        yaml_content = payload.get("content")
        if not yaml_content:
            raise HTTPException(
                status_code=400, detail="Missing 'content' field in request body"
            )

        # Parse YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not yaml_data.get("backend_name"):
            raise HTTPException(
                status_code=400, detail="backend_name is required in YAML"
            )

        # Check if backend with same name already exists
        existing = await InferenceBackend.one_by_field(
            session, "backend_name", yaml_data["backend_name"]
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Inference backend with name '{yaml_data['backend_name']}' already exists",
            )

        # Validate that build-in backends cannot have default_version set
        if is_build_in_backend(yaml_data["backend_name"]) and yaml_data.get(
            "default_version"
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Build-in backend '{yaml_data['backend_name']}' cannot have default_version set. Default version is managed automatically.",
            )

        # Convert version_configs to VersionConfigDict if present
        if 'version_configs' in yaml_data and yaml_data['version_configs']:
            version_configs_dict = {}
            for version, config in yaml_data['version_configs'].items():
                version_configs_dict[version] = VersionConfig(**config)
            yaml_data['version_configs'] = VersionConfigDict(root=version_configs_dict)

        # Create the backend
        backend = InferenceBackend(**yaml_data)
        backend = await InferenceBackend.create(session, backend)

        return backend

    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {e}")
    except HTTPException:
        raise  # Re-raise HTTPException without wrapping
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create inference backend from YAML: {e.__str__()}"
        )


@router.put("/{id}/from-yaml", response_model=InferenceBackend)
async def update_inference_backend_from_yaml(
    session: SessionDep, id: int, payload: dict = Body(...)
):
    """
    Update an existing inference backend from YAML configuration.

    Expected JSON format:
    ```json
    {
      "content": "backend_name: \"my-custom-backend\"\nversion_configs:\n  \"v1.0.0\":\n    image_name: \"my-backend:v1.0.0\"\n    run_command: \"python server.py --port {{port}} --model {{model_path}}\"\n  \"v1.1.0\":\n    image_name: \"my-backend:v1.1.0\"\n    run_command: \"python server.py --port {{port}} --model {{model_path}}\"\ndefault_version: \"v1.1.0\"\ndefault_backend_param: [\"--max-tokens\", \"2048\"]\ndefault_run_command: \"python server.py\"\ndescription: \"My custom inference backend\"\nhealth_check_path: \"/health\"\nallowed_proxy_uris: [\"/v1/chat/completions\", \"/v1/completions\"]"
    }
    """
    backend = await InferenceBackend.one_by_id(session, id)
    if not backend:
        raise NotFoundException(message=f"Inference backend {id} not found")

    try:
        # Extract YAML content from JSON payload
        yaml_content = payload.get("content")
        if not yaml_content:
            raise HTTPException(
                status_code=400, detail="Missing 'content' field in request body"
            )

        # Parse YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not yaml_data.get("backend_name"):
            raise HTTPException(
                status_code=400, detail="backend_name is required in YAML"
            )

        # Check if updating to a name that already exists (excluding current backend)
        if yaml_data["backend_name"] != backend.backend_name:
            raise HTTPException(
                status_code=400,
                detail="The name of inference-backend can not be modified",
            )

        # Validate that build-in backends cannot have default_version set
        if is_build_in_backend(backend.backend_name) and yaml_data.get(
            "default_version"
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Build-in backend '{backend.backend_name}' cannot have default_version set. Default version is managed automatically.",
            )

        # Convert version_configs to VersionConfigDict if present
        if 'version_configs' in yaml_data and yaml_data['version_configs']:
            version_configs_dict = {}
            for version, config in yaml_data['version_configs'].items():
                version_configs_dict[version] = VersionConfig(**config)
            yaml_data['version_configs'] = VersionConfigDict(root=version_configs_dict)

        # Create InferenceBackendUpdate object from YAML data
        backend_data = InferenceBackendUpdate(**yaml_data)

        # Update the backend
        await backend.update(session, backend_data)

        return backend

    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {e}")
    except HTTPException:
        raise  # Re-raise HTTPException without wrapping
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update inference backend from YAML: {e}"
        )
