import logging
from typing import List, Tuple, Optional, Dict

import yaml
from fastapi import APIRouter, Body
from gpustack_runner.runner import ServiceVersionedRunner
from sqlalchemy import or_, func
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
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
    get_built_in_backend,
    InferenceBackendPublic,
    VersionListItem,
)
from gpustack.schemas.models import BackendEnum, Model
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack_runner import list_service_runners

logger = logging.getLogger(__name__)
router = APIRouter()


def is_built_in_backend(backend_name: str) -> bool:
    """
    Check if a backend is a built-in backend.

    Args:
        backend_name: The name of the backend to check

    Returns:
        True if the backend is built-in, False otherwise
    """
    built_in_backends = get_built_in_backend()
    built_in_backend_names = {backend.backend_name for backend in built_in_backends}
    return backend_name in built_in_backend_names


def filter_yaml_fields(yaml_data: Dict, filter_keys: List[str]) -> Dict:
    """
    Recursively remove specified keys from a nested YAML dict.

    Args:
        yaml_data: Dictionary parsed from YAML content.
        filter_keys: List of keys to remove wherever they appear.

    Returns:
        The same dict instance after filtering.
    """

    if not isinstance(yaml_data, dict):
        return yaml_data

    def _filter_in_place(obj: Dict):
        # Delete keys that should be filtered
        for key in list(obj.keys()):
            if key in filter_keys:
                try:
                    del obj[key]
                except Exception:
                    # Silently ignore any deletion issues
                    pass
                continue

            # Recurse into nested dicts
            val = obj.get(key)
            if isinstance(val, dict):
                _filter_in_place(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _filter_in_place(item)

    _filter_in_place(yaml_data)
    return yaml_data


async def check_backend_in_use(
    session: SessionDep, backend_name: str, backend_version: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Check if a backend or specific backend version is being used by any models.

    Args:
        session: Database session
        backend_name: The name of the backend to check
        backend_version: Optional specific version to check. If None, checks all versions.

    Returns:
        A tuple containing:
        - Boolean indicating if the backend/version is in use
        - List of model names that are using the backend/version
    """
    try:
        # Query models that use the specified backend
        if backend_version:
            # Check for specific backend and version combination
            models = await Model.all_by_fields(
                session, {"backend": backend_name, "backend_version": backend_version}
            )
        else:
            # Check for any models using this backend (any version)
            models = await Model.all_by_field(session, "backend", backend_name)
        models = [model for model in models if model.replicas > 0]
        model_names = [model.name for model in models]
        is_in_use = len(models) > 0

        return is_in_use, model_names
    except Exception as e:
        logger.error(f"Error checking backend usage: {e}")
        return False, []


def get_runner_versions_and_configs(
    backend_name: str,
) -> Tuple[Dict[str, ServiceVersionedRunner], VersionConfigDict, Optional[str]]:
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
    runners_list = list_service_runners(service=backend_name.lower())
    runner_versions: Dict[str, ServiceVersionedRunner] = {}
    version_configs = VersionConfigDict()
    default_version = None

    if runners_list and len(runners_list) > 0:
        for version in runners_list[0].versions:
            if version.version:
                runner_versions[version.version] = version
                backend_list = [
                    f"{backend_runner.backend}" for backend_runner in version.backends
                ]
                version_configs.root[version.version] = VersionConfig(
                    image_name=backend_name,
                    built_in_frameworks=backend_list,
                )
                if default_version is None:
                    default_version = version.version

    return runner_versions, version_configs, default_version


def deduplicate_versions(versions: List[VersionListItem]) -> List[VersionListItem]:
    seen = set()
    result = []

    for item in versions:
        key = (item.version, item.is_deprecated)
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


@router.get("/list", response_model=InferenceBackendResponse)
async def list_backend_configs(session: SessionDep, cluster_id: Optional[int] = None):
    """
    Get list of available backend configurations with version information.

    Returns both built-in backends and custom backends from database.
    Built-in backends are identified and enhanced with runner versions.
    Each backend item includes available versions.
    """
    items = []

    if cluster_id and cluster_id > 0:
        workers = await Worker.all_by_field(session, "cluster_id", cluster_id)
    else:
        workers = await Worker.all(session)
    framework_list = set()
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                framework_list.add(gpu.runtime_framework)

    # Process all backends from database (includes both built-in and custom backends)
    try:
        inference_backends = await InferenceBackend.all(session)
        for backend in inference_backends:
            # Get versions from version_config
            versions: List[VersionListItem] = []
            if backend.version_configs and backend.version_configs.root:
                versions = [
                    VersionListItem(version=version)
                    for version in backend.version_configs.root.keys()
                ]

            if backend.is_built_in:
                # For built-in backends, add runner versions and use special show name
                runner_versions, version_configs, default_version = (
                    get_runner_versions_and_configs(backend.backend_name)
                )
                # Merge runner versions with existing versions
                for version, config in version_configs.root.items():
                    filtered_frameworks = [
                        framework
                        for framework in config.built_in_frameworks
                        if framework in framework_list
                    ]
                    if filtered_frameworks:
                        is_deprecated = False
                        if runner_versions.get(version):
                            is_deprecated = runner_versions[version].deprecated
                        versions.append(
                            VersionListItem(
                                version=version, is_deprecated=is_deprecated
                            )
                        )

                # Remove duplicates while preserving order
                versions = deduplicate_versions(versions)

                # Update default version if found from runner
                if default_version and not backend.default_version:
                    backend.default_version = default_version

                backend_item = InferenceBackendListItem(
                    backend_name=backend.backend_name,
                    default_version=backend.default_version,
                    default_backend_param=backend.default_backend_param,
                    versions=versions,
                    is_built_in=backend.is_built_in,
                )
            else:
                # For custom backends, use backend_name as show_name
                backend_item = InferenceBackendListItem(
                    backend_name=backend.backend_name,
                    default_version=backend.default_version,
                    default_backend_param=backend.default_backend_param,
                    versions=versions,
                    is_built_in=False,
                )

            items.append(backend_item)

        # Ensure Custom backend is always included even if not in database
        custom_backend_item = InferenceBackendListItem(
            backend_name=BackendEnum.CUSTOM,
            default_version=None,
            default_backend_param=None,
            versions=[],
            is_built_in=False,
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

    # Ensure all BUILT_IN_BACKENDS are included
    merged_backends = []
    built_in_backend_names = set()

    for built_in_backend in get_built_in_backend():
        if built_in_backend.backend_name == BackendEnum.CUSTOM:
            continue
        built_in_backend_names.add(built_in_backend.backend_name)

        # Get versions from list_service_runners using the common function
        _, runner_versions, default_version = get_runner_versions_and_configs(
            built_in_backend.backend_name
        )

        if default_version and not built_in_backend.default_version:
            built_in_backend.default_version = default_version

        db_backend = db_backends_map[built_in_backend.backend_name]
        if not db_backend:
            logger.warning(
                f"No database backend found for {built_in_backend.backend_name}"
            )
            continue
        # Merge versions from database backend.
        for runner_version, version_config in runner_versions.root.items():
            db_backend.built_in_version_configs[runner_version] = version_config

        if default_version and not db_backend.default_version:
            db_backend.default_version = default_version

        merged_backends.append(db_backend)

    # Add remaining database backends that are not in BUILT_IN_BACKENDS
    for backend_name, db_backend in db_backends_map.items():
        if backend_name not in built_in_backend_names:
            merged_backends.append(db_backend)

    return merged_backends


def _generate_framework_index_map(
    version_config_dicts: List[Dict[str, VersionConfig]]
) -> Dict[str, List[str]]:
    """
    Generate framework index map from a list of version config dictionaries.

    Args:
        version_config_dicts: List of dictionaries mapping version names to VersionConfig objects

    Returns:
        Dictionary mapping framework names to sorted lists of supported versions
    """
    framework_map = {}

    for version_configs in version_config_dicts:
        if not version_configs:
            continue

        for version, config in version_configs.items():
            if config.built_in_frameworks:
                for framework in config.built_in_frameworks:
                    if framework not in framework_map:
                        framework_map[framework] = []
                    if version not in framework_map[framework]:
                        framework_map[framework].append(version)
            if config.custom_framework:
                if config.custom_framework not in framework_map:
                    framework_map[config.custom_framework] = []
                framework_map[config.custom_framework].append(version)

    # Sort versions for each framework
    for framework in framework_map:
        framework_map[framework].sort()

    return framework_map


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
        for version, config in backend.built_in_version_configs.items():
            filtered_frameworks = [
                framework
                for framework in config.built_in_frameworks
                if framework in framework_list
            ]
            if filtered_frameworks:
                config.built_in_frameworks = filtered_frameworks
                filtered_version_configs[version] = config

        backend.built_in_version_configs = filtered_version_configs
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

    # Generate framework_index_map for each backend
    for backend in filter_backends:
        version_config_dicts = []
        if backend.built_in_version_configs:
            version_config_dicts.append(backend.built_in_version_configs)
        if backend.version_configs and backend.version_configs.root:
            version_config_dicts.append(backend.version_configs.root)

        backend.framework_index_map = _generate_framework_index_map(
            version_config_dicts
        )

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
        if not backend.is_built_in:
            ret.append(backend)
            continue
        for built_in_version, config in backend.built_in_version_configs.items():
            # if version in same, db version first
            if built_in_version not in backend.version_configs.root:
                backend.version_configs.root[built_in_version] = config

        ret.append(backend)

    return ret


@router.get("/{id}", response_model=InferenceBackend)
async def get_inference_backend(session: SessionDep, id: int):
    """
    Get a specific inference backend by ID.
    """
    backend = await InferenceBackend.one_by_id(session, id)
    if not backend:
        raise BadRequestException(message=f"Inference backend {id} not found")
    return backend


@router.get("/backend_name/{backend_name}", response_model=InferenceBackend)
async def get_inference_backend_by_name(session: SessionDep, backend_name: str):
    """
    Get a specific inference backend by backend name.
    """
    backend = await InferenceBackend.one_by_field(session, "backend_name", backend_name)
    if not backend:
        raise BadRequestException(message=f"Inference backend {backend_name} not found")
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
        raise BadRequestException(
            message=f"Inference backend with name '{backend_in.backend_name}' already exists",
        )

    # Validate that built-in backends cannot have default_version set
    if is_built_in_backend(backend_in.backend_name) and backend_in.default_version:
        raise BadRequestException(
            message=f"Built-in backend '{backend_in.backend_name}' cannot have default_version set. Default version is managed automatically.",
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
        raise BadRequestException(
            message="The name of inference-backend can not be modified",
        )

    # Validate that built-in backends cannot have default_version set
    if is_built_in_backend(backend.backend_name) and backend_in.default_version:
        raise BadRequestException(
            message=f"Built-in backend '{backend.backend_name}' cannot have default_version set. Default version is managed automatically.",
        )

    # Check if any versions are being removed or modified that are currently in use
    if backend_in.version_configs is not None:
        current_versions = {}
        if backend.version_configs and backend.version_configs.root:
            current_versions = backend.version_configs.root

        new_versions = {}
        if backend_in.version_configs and backend_in.version_configs.root:
            new_versions = backend_in.version_configs.root

        # Find versions that are being removed
        removed_versions = set(current_versions.keys()) - set(new_versions.keys())

        # Find versions that are being modified (same version name but different config)
        modified_versions = []
        for version_name in set(current_versions.keys()) & set(new_versions.keys()):
            current_config = current_versions[version_name]
            new_config = new_versions[version_name]

            # Compare the configurations by converting to dict and comparing
            current_dict = (
                current_config.model_dump()
                if hasattr(current_config, 'model_dump')
                else current_config.__dict__
            )
            new_dict = (
                new_config.model_dump()
                if hasattr(new_config, 'model_dump')
                else new_config.__dict__
            )

            if current_dict != new_dict:
                modified_versions.append(version_name)

        # Collect all versions that need to be checked (removed + modified)
        versions_to_check = list(removed_versions) + modified_versions

        # Check if any of these versions are in use
        for version in versions_to_check:
            is_in_use, model_names = await check_backend_in_use(
                session, backend.backend_name, version
            )
            if is_in_use:
                action = "remove" if version in removed_versions else "modify"
                raise BadRequestException(
                    message=f"Cannot {action} version '{version}' of backend '{backend.backend_name}' because it is currently being used by the following models: {', '.join(model_names)}",
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

    # Check if the backend is being used by any models
    is_in_use, model_names = await check_backend_in_use(session, backend.backend_name)
    if is_in_use:
        raise BadRequestException(
            message=f"Cannot delete backend '{backend.backend_name}' because it is currently being used by the following models: {', '.join(model_names)}",
        )

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
            raise BadRequestException(message="Missing 'content' field in request body")

        # Parse YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not yaml_data.get("backend_name"):
            raise BadRequestException(message="backend_name is required in YAML")

        # Check if backend with same name already exists
        existing = await InferenceBackend.one_by_field(
            session, "backend_name", yaml_data["backend_name"]
        )
        if existing:
            raise BadRequestException(
                message=f"Inference backend with name '{yaml_data['backend_name']}' already exists",
            )

        # Filter out fields users should not set
        common_filter_keys = [
            "id",
            "created_at",
            "updated_at",
            "framework_index_map",
            "built_in_version_configs",
        ]
        if is_built_in_backend(yaml_data["backend_name"]):
            # Built-in backend default_version is managed automatically
            common_filter_keys.append("default_version")

        yaml_data = filter_yaml_fields(yaml_data, common_filter_keys)

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
        raise BadRequestException(message=f"Invalid YAML format: {e}")
    except BadRequestException:
        raise  # Re-raise BadRequestException without wrapping
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create inference backend from YAML: {e.__str__()}"
        )


@router.put("/{id}/from-yaml", response_model=InferenceBackend)
async def update_inference_backend_from_yaml(  # noqa: C901
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
            raise BadRequestException(message="Missing 'content' field in request body")

        # Parse YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not yaml_data.get("backend_name"):
            raise BadRequestException(message="backend_name is required in YAML")

        # Check if updating to a name that already exists (excluding current backend)
        if yaml_data["backend_name"] != backend.backend_name:
            raise BadRequestException(
                message="The name of inference-backend can not be modified",
            )

        # Filter out fields users should not set
        common_filter_keys = [
            "id",
            "created_at",
            "updated_at",
            "framework_index_map",
            "built_in_version_configs",
        ]
        if is_built_in_backend(backend.backend_name):
            # Built-in backend default_version is managed automatically
            common_filter_keys.append("default_version")

        yaml_data = filter_yaml_fields(yaml_data, common_filter_keys)

        # Convert version_configs to VersionConfigDict if present
        if 'version_configs' in yaml_data and yaml_data['version_configs']:
            version_configs_dict = {}
            for version, config in yaml_data['version_configs'].items():
                version_configs_dict[version] = VersionConfig(**config)
            yaml_data['version_configs'] = VersionConfigDict(root=version_configs_dict)

        # Create InferenceBackendUpdate object from YAML data
        backend_data = InferenceBackendUpdate(**yaml_data)

        # Check if any versions are being removed or modified that are currently in use
        if 'version_configs' in yaml_data:
            current_versions = {}
            if backend.version_configs and backend.version_configs.root:
                current_versions = backend.version_configs.root

            new_versions = {}
            if yaml_data['version_configs'] and hasattr(
                yaml_data['version_configs'], 'root'
            ):
                new_versions = yaml_data['version_configs'].root

            # Find versions that are being removed
            removed_versions = set(current_versions.keys()) - set(new_versions.keys())

            # Find versions that are being modified (same version name but different config)
            modified_versions = []
            for version_name in set(current_versions.keys()) & set(new_versions.keys()):
                current_config = current_versions[version_name]
                new_config = new_versions[version_name]

                # Compare the configurations by converting to dict and comparing
                current_dict = (
                    current_config.model_dump()
                    if hasattr(current_config, 'model_dump')
                    else current_config.__dict__
                )
                new_dict = (
                    new_config.model_dump()
                    if hasattr(new_config, 'model_dump')
                    else new_config.__dict__
                )

                if current_dict != new_dict:
                    modified_versions.append(version_name)

            # Collect all versions that need to be checked (removed + modified)
            versions_to_check = list(removed_versions) + modified_versions

            # Check if any of these versions are in use
            for version in versions_to_check:
                is_in_use, model_names = await check_backend_in_use(
                    session, backend.backend_name, version
                )
                if is_in_use:
                    action = "remove" if version in removed_versions else "modify"
                    raise BadRequestException(
                        message=f"Cannot {action} version '{version}' of backend '{backend.backend_name}' because it is currently being used by the following models: {', '.join(model_names)}",
                    )

        # Update the backend
        await backend.update(session, backend_data)

        return backend

    except yaml.YAMLError as e:
        raise BadRequestException(message=f"Invalid YAML format: {e}")
    except BadRequestException:
        raise  # Re-raise BadRequestException without wrapping
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update inference backend from YAML: {e}"
        )
