from datetime import datetime
from typing import Dict, List, Optional

from gpustack_runtime.deployer.__utils__ import compare_versions
from pydantic import BaseModel, Field, RootModel
from sqlalchemy import JSON, Column, Text
from sqlmodel import SQLModel, Field as SQLField

from gpustack.mixins import BaseModelMixin
from .common import pydantic_column_type, PaginatedList
from .models import BackendEnum


class VersionConfig(BaseModel):
    """
    Configuration for a specific version of an inference backend.

    Attributes:
        image_name: Docker image name for this version
        run_command: Command to run the inference server (Optional, uses default if not specified)
        built_in_frameworks: Only built-in backend will return this field, sourced from gpustack-runner configuration. (Optional)
        custom_framework: User-provided value (upon backend creation) used for deployment and compatibility checks. (Optional)
    """

    image_name: Optional[str] = Field(None)
    run_command: Optional[str] = Field(None)
    built_in_frameworks: Optional[List[str]] = Field(None)
    custom_framework: Optional[str] = Field(None)


class VersionConfigDict(RootModel[Dict[str, VersionConfig]]):
    """
    Wrapper model for version configs dictionary to enable proper JSON serialization.
    """

    root: Dict[str, VersionConfig] = Field(default_factory=dict)


# Database Models
class InferenceBackendBase(SQLModel):
    """
    Base model for inference backends.

    Attributes:
        backend_name: Name of the backend (e.g., 'SGLang')
        version_configs: Dictionary mapping version strings to their configurations
        default_version: Default version to use if not specified
        default_backend_param: Default parameters to pass to the backend
        default_run_command: Default command to run the inference server
        description: Backend description
        health_check_path: Path for health check endpoint

    """

    backend_name: str = SQLField(index=True, unique=True)
    version_configs: VersionConfigDict = SQLField(
        sa_column=Column(pydantic_column_type(VersionConfigDict)()),
        default_factory=lambda: VersionConfigDict(root={}),
    )
    default_version: Optional[str] = SQLField(default=None)
    default_backend_param: Optional[List[str]] = SQLField(
        sa_column=Column(JSON), default=[]
    )
    default_run_command: Optional[str] = SQLField(default="")
    is_built_in: bool = SQLField(default=False)
    description: Optional[str] = SQLField(
        default=None, sa_column=Column(Text, nullable=True)
    )
    health_check_path: Optional[str] = SQLField(default=None)

    def get_version_config(self, version: Optional[str] = None) -> VersionConfig:
        """
        Get configuration for a specific version.

        Args:
            version: Version string, uses default_version if None

        Returns:
            VersionConfig for the specified version

        Raises:
            KeyError: If the version is not found in version_configs
        """
        target_version = version or self.default_version

        version_configs_dict = self.version_configs.root

        if target_version not in version_configs_dict:
            # if no version or default_version is specified,
            # automatically select the latest version from the available versions
            if version_configs_dict and not self.is_built_in:
                # Get the latest version from the available versions
                # Sort using the same logic as compare_versions function from gpustack_runtime
                try:
                    # Find the highest version by comparing all versions
                    version_list = list(version_configs_dict.keys())
                    latest_version = version_list[0]
                    for ver in version_list[1:]:
                        if (
                            compare_versions(ver, latest_version) > 0
                        ):  # ver > latest_version
                            latest_version = ver
                    target_version = latest_version
                except Exception:
                    # Fallback to simple string sorting if compare_versions is not available
                    sorted_versions = sorted(version_configs_dict.keys())
                    target_version = sorted_versions[-1]

                return version_configs_dict[target_version]

            raise KeyError(
                f"Version '{target_version}' not found in backend '{self.backend_name}'"
            )
        return version_configs_dict[target_version]

    def get_run_command(self, version: Optional[str] = None) -> str:
        if not version:
            version = self.default_version
        version_config = self.get_version_config(version)
        return version_config.run_command or self.default_run_command

    def replace_command_param(
        self,
        version: Optional[str],
        model_path: Optional[str],
        port: Optional[int],
        model_name: Optional[str] = None,
        command: Optional[str] = None,
    ) -> str:
        if not command:
            command = self.get_run_command(version)
            if not command:
                return ""

        command = command.replace("{{model_path}}", model_path)
        command = command.replace("{{port}}", str(port))
        command = command.replace("{{model_name}}", model_name)
        return command

    def get_image_name(self, version: Optional[str] = None) -> str:
        if self.backend_name == BackendEnum.CUSTOM.value:
            return ""
        try:
            version_config = self.get_version_config(version)
        except KeyError:
            return ""
        return version_config.image_name


class InferenceBackend(InferenceBackendBase, BaseModelMixin, table=True):
    __tablename__ = 'inference_backends'
    id: Optional[int] = SQLField(default=None, primary_key=True)


class VersionListItem(BaseModel):
    version: str = Field(...)
    is_deprecated: bool = Field(default=False)


class InferenceBackendListItem(BaseModel):
    """Backend configuration item."""

    backend_name: str = Field(...)
    is_built_in: Optional[bool] = Field(None)
    default_version: Optional[str] = Field(None)
    default_backend_param: Optional[List[str]] = Field(None)
    versions: Optional[List[VersionListItem]] = Field(
        None, description="Available versions for this backend"
    )


class InferenceBackendResponse(BaseModel):
    """Response for backend configs list."""

    items: List[InferenceBackendListItem] = Field(...)


# CRUD API Models
class InferenceBackendCreate(InferenceBackendBase):
    pass


class InferenceBackendUpdate(InferenceBackendBase):
    pass


class InferenceBackendPublic(InferenceBackendBase):
    id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    built_in_version_configs: Optional[Dict[str, VersionConfig]] = {}
    framework_index_map: Optional[Dict[str, List[str]]] = {}


InferenceBackendsPublic = PaginatedList[InferenceBackendPublic]


# built-in backend configurations
def get_built_in_backend() -> List[InferenceBackend]:
    return [
        InferenceBackend(backend_name=BackendEnum.VLLM.value, is_built_in=True),
        InferenceBackend(backend_name=BackendEnum.SGLANG.value, is_built_in=True),
        InferenceBackend(
            backend_name=BackendEnum.ASCEND_MINDIE.value, is_built_in=True
        ),
        InferenceBackend(backend_name=BackendEnum.VOX_BOX.value, is_built_in=True),
        InferenceBackend(backend_name=BackendEnum.CUSTOM.value, is_built_in=True),
    ]


def is_built_in_backend(backend_name: str) -> bool:
    """
    Check if a backend is a built-in backend.

    Args:
        backend_name: The name of the backend to check

    Returns:
        True if the backend is built-in, False otherwise
    """
    built_in_backends = get_built_in_backend()
    built_in_backend_names = {
        backend.backend_name.lower() for backend in built_in_backends
    }
    return backend_name.lower() in built_in_backend_names
