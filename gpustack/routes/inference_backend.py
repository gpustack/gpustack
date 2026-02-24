import logging
import math
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

import yaml
from fastapi import APIRouter, Body
from gpustack_runner.runner import ServiceVersionedRunner, ServiceRunner
from gpustack_runtime.deployer.__utils__ import compare_versions
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
    is_built_in_backend,
)
from gpustack.schemas.models import BackendEnum, Model, BackendSourceEnum
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack_runner import list_service_runners
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack_runtime.detector import ManufacturerEnum

logger = logging.getLogger(__name__)
router = APIRouter()


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


def get_lower_version_runners(
    runners: list[ServiceRunner], backend_version: str
) -> list[ServiceRunner]:
    """
    Filter runners whose version is less than or equal to the given backend_version.
    Rebuilds the list[ServiceRunner] structure with only the matching elements.

    Args:
        runners: List of ServiceRunner objects to filter
        backend_version: The version to compare against (only runners with versions <= this will be kept)

    Returns:
        List of ServiceRunner objects with filtered versions/backends
    """
    filtered_runners = []
    for runner in runners:
        # Create a new runner with filtered structure
        new_runner = deepcopy(runner)

        # Filter versions in backends
        for version in new_runner.versions:
            for backend in version.backends:
                # Filter backend versions that are <= backend_version
                backend.versions = [
                    bv
                    for bv in backend.versions
                    if compare_versions(bv.version, backend_version) <= 0
                ]

        # Remove backends with no matching versions
        for version in new_runner.versions:
            version.backends = [
                backend for backend in version.backends if backend.versions
            ]

        # Remove versions with no matching backends
        new_runner.versions = [
            version for version in new_runner.versions if version.backends
        ]

        # Only add runner if it has matching versions
        if new_runner.versions:
            filtered_runners.append(new_runner)

    return filtered_runners


def get_runner_versions_and_configs(
    backend_name: str, backend_version: Optional[str], **kwargs
) -> Tuple[Dict[str, ServiceVersionedRunner], VersionConfigDict, Optional[str]]:
    """
    Get runner versions and version configs for a given backend.

    Args:
        backend_name: The name of the backend service
        kwargs: Others keyword arguments to pass to list_service_runners()

    Returns:
        A tuple containing:
        - List of version strings
        - VersionConfigDict with version configurations
        - Default version (first available version or None)
    """
    runners_list = list_service_runners(
        service=backend_name.lower(),
        **kwargs,
    )
    if backend_version:
        runners_list = get_lower_version_runners(runners_list, backend_version)
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


def get_runner_deprecate(runners: List[ServiceVersionedRunner]) -> bool:
    """
    Check if all runners are deprecated.

    Args:
        runners: List of ServiceVersionedRunner objects

    Returns:
        True if all runners are deprecated, False otherwise.
        Returns False if the list is empty.
    """
    if not runners:
        return False
    return all(
        runner.backends[0].versions[0].variants[0].deprecated for runner in runners
    )


def merge_list_runners(
    backend_name: str, workers: List[Worker]
) -> Tuple[Dict[str, List[ServiceVersionedRunner]], VersionConfigDict, Optional[str]]:
    """
    Merge runner versions and configs from multiple workers.

    Extracts gpu.type and gpu.runtime_version from each worker's GPU devices
    and uses them as query conditions for list_service_runners.

    Args:
        backend_name: The name of the backend service
        workers: List of workers to extract GPU information from

    Returns:
        A tuple containing:
        - Dict[str, List[ServiceVersionedRunner]]: Merged runner versions, grouped by version
        - VersionConfigDict: Merged version configurations
        - Optional[str]: Default version (from first query)
    """
    # Collect unique query conditions from all workers
    query_conditions = set()
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                # Extract variant for Ascend GPUs
                variant = None
                if gpu.vendor == ManufacturerEnum.ASCEND and gpu.arch_family:
                    variant = get_ascend_cann_variant(gpu.arch_family).lower()

                # Add (type, runtime_version, variant) tuple to set
                # Use None for runtime_version if not available
                query_conditions.add((gpu.type, gpu.runtime_version, variant))

    merged_runner_versions: Dict[str, List[ServiceVersionedRunner]] = {}
    merged_version_configs = VersionConfigDict()
    merged_default_version = None

    # Loop through each unique query condition
    for idx, (gpu_type, runtime_version, variant) in enumerate(query_conditions):
        # Build kwargs for get_runner_versions_and_configs
        kwargs = {"backend": gpu_type}
        if variant:
            kwargs["backend_variant"] = variant

        # Get runner versions and configs for this condition
        runner_versions, version_configs, default_version = (
            get_runner_versions_and_configs(backend_name, runtime_version, **kwargs)
        )

        # For the first condition, use its results as base
        if idx == 0:
            # Convert Dict[str, ServiceVersionedRunner] to Dict[str, List[ServiceVersionedRunner]]
            merged_runner_versions = {
                version: [runner] for version, runner in runner_versions.items()
            }
            merged_version_configs = version_configs
            merged_default_version = default_version
        else:
            # Merge runner versions (append to list if exists)
            for version, runner in runner_versions.items():
                if version in merged_runner_versions:
                    merged_runner_versions[version].append(runner)
                else:
                    merged_runner_versions[version] = [runner]

            # Merge version configs
            for version, config in version_configs.root.items():
                if version not in merged_version_configs.root:
                    # Add new version
                    merged_version_configs.root[version] = config
                else:
                    # Merge built_in_frameworks (deduplicate)
                    existing_frameworks = (
                        merged_version_configs.root[version].built_in_frameworks or []
                    )
                    new_frameworks = config.built_in_frameworks or []
                    merged_frameworks = list(set(existing_frameworks + new_frameworks))
                    merged_version_configs.root[version].built_in_frameworks = (
                        merged_frameworks
                    )

    return merged_runner_versions, merged_version_configs, merged_default_version


@router.get("/list", response_model=InferenceBackendResponse)
async def list_backend_configs(  # noqa: C901
    session: SessionDep, cluster_id: Optional[int] = None
):
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

    # Process all backends from database (includes both built-in and custom backends)
    try:
        inference_backends = await InferenceBackend.all(session)
        for backend in inference_backends:
            # Get versions from version_config
            versions: List[VersionListItem] = []
            if backend.version_configs and backend.version_configs.root:
                versions = [
                    VersionListItem(
                        version=version, env=backend.get_backend_env(version)
                    )
                    for version in backend.version_configs.root.keys()
                ]

            if backend.is_built_in:
                # For built-in backends, add runner versions and use special show name
                runner_versions, version_configs, default_version = merge_list_runners(
                    backend.backend_name,
                    workers,
                )
                # Merge runner versions with existing versions
                for version, config in version_configs.root.items():
                    # Check if this version has any built-in frameworks
                    if config.built_in_frameworks:
                        # Versions are only marked deprecated when no worker is compatible with them.
                        is_deprecated = get_runner_deprecate(
                            runner_versions.get(version, [])
                        )
                        # Get environment for this specific version
                        version_env = backend.get_backend_env(version)
                        versions.append(
                            VersionListItem(
                                version=version,
                                is_deprecated=is_deprecated,
                                env=version_env,
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
                    enabled=True,
                    backend_source=BackendSourceEnum.BUILT_IN,
                    default_env=backend.default_env,
                )
            else:
                if (
                    backend.backend_source == BackendSourceEnum.COMMUNITY
                    and not backend.enabled
                ):
                    continue
                # For custom backends, use backend_name as show_name
                backend_item = InferenceBackendListItem(
                    backend_name=backend.backend_name,
                    default_version=backend.default_version,
                    default_backend_param=backend.default_backend_param,
                    versions=versions,
                    is_built_in=False,
                    enabled=backend.enabled,
                    backend_source=backend.backend_source,
                    default_env=backend.default_env,
                )

            items.append(backend_item)

        # Ensure Custom backend is always included even if not in database
        custom_backend_item = InferenceBackendListItem(
            backend_name=BackendEnum.CUSTOM,
            default_version=None,
            default_backend_param=None,
            versions=[],
            is_built_in=False,
            enabled=True,
            backend_source=BackendSourceEnum.BUILT_IN,
            default_env=None,
        )
        items.append(custom_backend_item)

    except Exception as e:
        # Log error but don't fail the entire request
        logger.error(f"Failed to load backends from database: {e}")

    return InferenceBackendResponse(items=items)


async def merge_runner_versions_to_db(
    session: SessionDep, with_deprecated: bool = True
) -> List[InferenceBackendPublic]:
    # Get database backends first
    db_result = await InferenceBackend.all(session)

    # Sort by id ascending to ensure consistent ordering
    db_result_sorted = sorted(db_result, key=lambda x: x.id if x.id else 0)

    # Create a map of database backends by name for easy lookup
    db_backends_map = {
        backend.backend_name: InferenceBackendPublic(**backend.model_dump())
        for backend in db_result_sorted
    }

    # Ensure all BUILT_IN_BACKENDS are included
    merged_backends = []
    built_in_backend_names = set()

    for built_in_backend in get_built_in_backend():
        if built_in_backend.backend_name == BackendEnum.CUSTOM.value:
            continue
        built_in_backend_names.add(built_in_backend.backend_name)

        # Get versions from list_service_runners using the common function
        _, runner_versions, default_version = get_runner_versions_and_configs(
            built_in_backend.backend_name,
            backend_version=None,
            with_deprecated=with_deprecated,
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
            # Migrate versions with built_in_frameworks to built_in_version_configs
            if (
                db_backend.backend_source == BackendSourceEnum.COMMUNITY
                and db_backend.version_configs
                and db_backend.version_configs.root
            ):
                versions_to_move = {}
                for version, config in db_backend.version_configs.root.items():
                    if config.built_in_frameworks:
                        versions_to_move[version] = config

                if versions_to_move:
                    if not db_backend.built_in_version_configs:
                        db_backend.built_in_version_configs = {}
                    db_backend.built_in_version_configs.update(versions_to_move)

                    for version in versions_to_move:
                        del db_backend.version_configs.root[version]

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


def _filter_community_backends(
    backends: List[InferenceBackendPublic],
    is_only_community: Optional[bool] = None,
) -> List[InferenceBackendPublic]:
    """
    Filter backends to only include community backends without custom frameworks.

    This function filters the backend list to only include backends with
    backend_source=COMMUNITY, and removes any versions that have custom_framework set.

    Args:
        backends: List of inference backends to filter

    Returns:
        List of community backends with non-custom framework versions only
    """
    filter_backends = []

    for backend in backends:
        if is_only_community:
            # using in community_backends catalog
            if backend.backend_source != BackendSourceEnum.COMMUNITY:
                continue
            backend.version_configs.root = {}
        else:
            # using in common inference_backends view
            if (
                backend.backend_source == BackendSourceEnum.COMMUNITY
                and not backend.enabled
            ):
                continue

        filter_backends.append(backend)

    return filter_backends


@router.get("", response_model=InferenceBackendsPublic)
async def get_inference_backends(  # noqa: C901
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    include_deprecated: bool = False,
    community: Optional[bool] = None,
    backend_source: Optional[str] = None,
):
    """
    Get paginated list of inference backends with optional search and filters.

    Args:
        session: Database session
        params: List parameters (page, perPage, watch, sort_by)
        search: Search keyword for backend_name and description
        include_deprecated: Include deprecated versions
        community: Filter community backends (True=community only with non-custom versions, False/None=all backends)
        backend_source: Filter by backend source (built-in, custom, or community)

    Returns:
        InferenceBackendsPublic: Paginated list of inference backends
    """
    fields = {}

    if params.watch:
        return StreamingResponse(
            InferenceBackend.streaming(fields=fields),
            media_type="text/event-stream",
        )

    merged_backends = await merge_runner_versions_to_db(
        session, with_deprecated=include_deprecated
    )

    # Get worker GPU information for framework sorting
    workers = await Worker.all(session)
    framework_list = set()
    for worker in workers:
        if worker.status and worker.status.gpu_devices:
            for gpu in worker.status.gpu_devices:
                framework_list.add(gpu.type)

    # Single-pass filtering and transformation pipeline:
    # 1. Framework sorting (data transformation)
    # 2. Search filter (early rejection)
    # 3. Community filter (early rejection)
    # 4. Backend source filter (early rejection)
    # 5. Framework index map generation (final transformation)
    filter_backends = []
    for backend in merged_backends:
        # 1. Sort frameworks by support status (must be first as it modifies data structure)
        sorted_version_configs = {}
        for version, config in backend.built_in_version_configs.items():
            if config.built_in_frameworks:
                supported = [
                    framework
                    for framework in config.built_in_frameworks
                    if framework in framework_list
                ]
                unsupported = [
                    framework
                    for framework in config.built_in_frameworks
                    if framework not in framework_list
                ]
                config.built_in_frameworks = supported + unsupported

            sorted_version_configs[version] = config

        backend.built_in_version_configs = sorted_version_configs

        # 2. Apply search filter (early rejection to reduce subsequent processing)
        if search:
            lower_search = search.lower()
            if not (
                lower_search in backend.backend_name.lower()
                or (backend.description and lower_search in backend.description.lower())
            ):
                continue  # Skip backends that don't match search criteria

        # 3. Apply community filter (early rejection)
        if community is True:
            # Using in community_backends catalog
            if backend.backend_source != BackendSourceEnum.COMMUNITY:
                continue
            # Clear custom versions for community backends
            if backend.version_configs:
                backend.version_configs.root = {}
        else:
            # Using in common inference_backends view
            if (
                backend.backend_source == BackendSourceEnum.COMMUNITY
                and not backend.enabled
            ):
                continue

        # 4. Apply backend_source filter (early rejection)
        if backend_source:
            try:
                source_enum = BackendSourceEnum(backend_source)
                if backend.backend_source != source_enum:
                    continue
            except ValueError:
                # Invalid backend_source value, log warning but don't filter
                logger.warning(f"Invalid backend_source value: {backend_source}")

        # 5. Generate framework_index_map (must be last as it depends on processed data)
        version_config_dicts = []
        if backend.built_in_version_configs:
            version_config_dicts.append(backend.built_in_version_configs)
        if backend.version_configs and backend.version_configs.root:
            version_config_dicts.append(backend.version_configs.root)

        backend.framework_index_map = _generate_framework_index_map(
            version_config_dicts
        )

        # Backend passed all filters, add to result list
        filter_backends.append(backend)

    # Apply pagination to merged results
    total = len(filter_backends)
    start_idx = (params.page - 1) * params.perPage
    end_idx = start_idx + params.perPage
    paginated_backends = filter_backends[start_idx:end_idx]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=total,
        totalPage=max(math.ceil(total / params.perPage), 1),
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
    # Duplicate names with built-in backends are prohibited (case-insensitive).
    if is_built_in_backend(backend_in.backend_name):
        raise BadRequestException(
            message=(
                f"Backend name {backend_in.backend_name} duplicates with built-in backends (case-insensitive). Please use another name."
            ),
        )
    backend_in.backend_source = BackendSourceEnum.CUSTOM
    backend_in.enabled = True
    # Check if backend with same name already exists
    existing = await InferenceBackend.one_by_field(
        session, "backend_name", backend_in.backend_name
    )
    if existing:
        raise BadRequestException(
            message=f"Inference backend with name '{backend_in.backend_name}' already exists",
        )

    # Validate version names for custom backends before creating
    validate_custom_suffix(backend_in.backend_name, None)

    for version in backend_in.version_configs.root.keys():
        backend_in.version_configs.root[version].built_in_frameworks = None

    try:
        backend = InferenceBackend(
            backend_name=backend_in.backend_name,
            version_configs=backend_in.version_configs,
            default_version=backend_in.default_version,
            default_backend_param=backend_in.default_backend_param,
            default_run_command=backend_in.default_run_command,
            default_entrypoint=backend_in.default_entrypoint,
            health_check_path=backend_in.health_check_path,
            description=backend_in.description,
            default_env=backend_in.default_env,
            enabled=backend_in.enabled,
            backend_source=backend_in.backend_source,
        )
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

    if backend_in.version_configs is not None:
        await _validate_version_removal(session, backend, backend_in.version_configs)

    # Validate version names for custom backends before updating
    if backend.backend_source == BackendSourceEnum.CUSTOM or (
        backend.backend_source is None and not backend.is_built_in
    ):
        validate_custom_suffix(backend_in.backend_name, None)
    else:
        validate_custom_suffix(None, backend_in.version_configs)

    for version in backend_in.version_configs.root.keys():
        backend_in.version_configs.root[version].built_in_frameworks = None

    try:
        # Use a dict for changes to prevent version_config serialization errors and None field overrides issues.
        update_data = {
            "backend_name": backend_in.backend_name,
            "version_configs": backend_in.version_configs,
            "default_version": backend_in.default_version,
            "default_backend_param": backend_in.default_backend_param,
            "default_run_command": backend_in.default_run_command,
            "default_entrypoint": backend_in.default_entrypoint,
            "health_check_path": backend_in.health_check_path,
            "description": backend_in.description,
            "default_env": backend_in.default_env,
            "backend_source": backend_in.backend_source,
        }
        if backend_in.backend_source == BackendSourceEnum.COMMUNITY:
            if backend_in.enabled is not None:
                update_data["enabled"] = backend_in.enabled
            built_in_version = {
                k: v
                for k, v in backend.version_configs.root.items()
                if v.built_in_frameworks
            }
            # merge built-in versions with custom versions for update
            built_in_version.update(update_data['version_configs'].root)
            update_data['version_configs'].root = built_in_version

        await backend.update(session, update_data)
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

    if (
        backend.backend_source != BackendSourceEnum.CUSTOM
        and backend.backend_source is not None
    ):
        raise BadRequestException(message="Cannot delete built-in or community backend")

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
        req_yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not req_yaml_data.get("backend_name"):
            raise BadRequestException(message="backend_name is required in YAML")

        # Check if backend name duplicates with built-in backends (case-insensitive)
        if is_built_in_backend(req_yaml_data["backend_name"]):
            raise BadRequestException(
                message=(
                    f"Backend name {req_yaml_data['backend_name']} duplicates with built-in backends (case-insensitive). Please use another name."
                ),
            )
        req_yaml_data["backend_source"] = BackendSourceEnum.CUSTOM
        req_yaml_data["enabled"] = True

        # Check if backend with same name already exists
        existing = await InferenceBackend.one_by_field(
            session, "backend_name", req_yaml_data["backend_name"]
        )
        if existing:
            raise BadRequestException(
                message=f"Inference backend with name '{req_yaml_data['backend_name']}' already exists",
            )

        allowed_keys = [
            "backend_name",
            "version_configs",
            "default_version",
            "default_backend_param",
            "default_run_command",
            "health_check_path",
            "description",
            "default_env",
            "enabled",
            "backend_source",
        ]
        yaml_data = {k: req_yaml_data[k] for k in allowed_keys if k in req_yaml_data}

        # Convert version_configs to VersionConfigDict if present
        if 'version_configs' in yaml_data and yaml_data['version_configs']:
            version_configs_dict = {}
            for version, config in yaml_data['version_configs'].items():
                if config.get('built_in_frameworks'):
                    config['built_in_frameworks'] = None
                version_configs_dict[version] = VersionConfig(**config)
            yaml_data['version_configs'] = VersionConfigDict(root=version_configs_dict)

        # Validate version names for custom backends
        validate_custom_suffix(yaml_data['backend_name'], None)

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
        req_yaml_data = yaml.safe_load(yaml_content)

        # Validate required fields
        if not req_yaml_data.get("backend_name"):
            raise BadRequestException(message="backend_name is required in YAML")

        # Check if updating to a name that already exists (excluding current backend)
        if req_yaml_data["backend_name"] != backend.backend_name:
            raise BadRequestException(
                message="The name of inference-backend can not be modified",
            )

        allowed_keys = [
            "backend_name",
            "version_configs",
            "default_backend_param",
            "default_run_command",
            "health_check_path",
            "description",
            "default_env",
            "enabled",
            "backend_source",
        ]
        if not is_built_in_backend(backend.backend_name):
            allowed_keys.append("default_version")

        yaml_data = {k: req_yaml_data[k] for k in allowed_keys if k in req_yaml_data}

        # Process version_configs if present
        yaml_data['version_configs'] = _process_version_configs(
            yaml_data.get('version_configs')
        )

        # Check if any versions are being removed and validate they're not in use
        await _validate_version_removal(
            session, backend, yaml_data.get('version_configs')
        )

        # Validate version names based on backend source
        if backend.backend_source == BackendSourceEnum.CUSTOM or (
            backend.backend_source is None and not backend.is_built_in
        ):
            validate_custom_suffix(yaml_data['backend_name'], None)
        else:
            validate_custom_suffix(None, yaml_data.get('version_configs'))

        # Clear built_in_frameworks for all versions in yaml_data
        _clear_built_in_frameworks(yaml_data.get('version_configs'))

        # Merge built-in versions for COMMUNITY backends
        if backend.backend_source == BackendSourceEnum.COMMUNITY:
            yaml_data['version_configs'] = _merge_community_versions(
                backend, yaml_data.get('version_configs')
            )

        # Update the backend from YAML data (after normalization)
        await backend.update(session, yaml_data)

        return backend

    except yaml.YAMLError as e:
        raise BadRequestException(message=f"Invalid YAML format: {e}")
    except BadRequestException:
        raise  # Re-raise BadRequestException without wrapping
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update inference backend from YAML: {e}"
        )


def _process_version_configs(
    version_configs_data: Optional[dict],
) -> VersionConfigDict:
    """
    Convert raw version_configs dict to VersionConfigDict.

    Returns None if version_configs_data is None or empty.
    """
    version_configs_dict = {}
    for version, config in version_configs_data.items() if version_configs_data else []:
        # Clear built_in_frameworks during initial processing
        if config.get('built_in_frameworks'):
            config['built_in_frameworks'] = None
        version_configs_dict[version] = VersionConfig(**config)

    return VersionConfigDict(root=version_configs_dict)


async def _validate_version_removal(
    session,
    backend: InferenceBackend,
    new_version_configs: Optional[VersionConfigDict],
):
    """
    Check if any versions are being removed and validate they're not in use.
    """
    # Get current versions (empty dict if none)
    current_versions = {}
    if backend.version_configs and backend.version_configs.root:
        current_versions = {
            v: config
            for v, config in backend.version_configs.root.items()
            if not config.built_in_frameworks
        }

    # Get new versions (empty dict if none)
    new_versions = {}
    if new_version_configs and new_version_configs.root:
        new_versions = new_version_configs.root

    # Find removed versions
    removed_versions = set(current_versions.keys()) - set(new_versions.keys())

    # Check if removed versions are in use
    for version in removed_versions:
        is_in_use, model_names = await check_backend_in_use(
            session, backend.backend_name, version
        )
        if is_in_use:
            raise BadRequestException(
                message=f"Cannot remove version name '{version}' of backend '{backend.backend_name}' because it is currently being used by the following models: {', '.join(model_names)}",
            )


def _clear_built_in_frameworks(version_configs: Optional[VersionConfigDict]):
    """
    Clear built_in_frameworks for all versions in version_configs.
    """
    if not version_configs or not version_configs.root:
        return

    for version_config in version_configs.root.values():
        version_config.built_in_frameworks = None


def _merge_community_versions(
    backend: InferenceBackend,
    new_version_configs: Optional[VersionConfigDict],
) -> VersionConfigDict:
    """
    Merge built-in versions with new versions for COMMUNITY backends.

    Returns:
        VersionConfigDict: Merged version configurations with built-in versions preserved
    """
    # Extract built-in versions from current backend
    built_in_versions = {}
    if backend.version_configs and backend.version_configs.root:
        built_in_versions = {
            k: v
            for k, v in backend.version_configs.root.items()
            if v.built_in_frameworks
        }

    if not new_version_configs or not new_version_configs.root:
        return VersionConfigDict(root=built_in_versions)

    # Merge: built-in versions + new versions (new versions take precedence)
    built_in_versions.update(new_version_configs.root or {})
    new_version_configs.root = built_in_versions
    return new_version_configs


def validate_custom_suffix(
    backend_name: Optional[str],
    version_configs: Optional[VersionConfigDict],
):
    """
    Validate custom suffix for backend names and version names.

    Rules:
    - Backend name: Must end with '-custom' if provided
    - Version name: Must end with '-custom' ONLY if it's a user-defined version
      (i.e., built_in_frameworks is None and custom_framework has value)
    """
    # Validate backend name
    if backend_name and not backend_name.endswith("-custom"):
        raise BadRequestException(
            message=f"Custom backend name '{backend_name}' must end with '-custom'",
        )

    # Validate version names
    if version_configs and version_configs.root:
        for version, config in version_configs.root.items():
            # Skip predefined versions (built_in_frameworks has value)
            if config.built_in_frameworks:
                continue

            # User-defined versions must have -custom suffix
            if not isinstance(version, str) or not version.endswith("-custom"):
                raise BadRequestException(
                    message=f"Custom backend version '{version}' must end with '-custom'",
                )
