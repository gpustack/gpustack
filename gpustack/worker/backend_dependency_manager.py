import logging
from typing import Dict, List
from dataclasses import dataclass

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from gpustack.schemas.models import BackendEnum

logger = logging.getLogger(__name__)


@dataclass
class BackendDependencySpec:
    """
    Represents a backend dependency specification.

    Attributes:
        backend: The backend name (e.g., 'vox-box', 'vllm')
        dependencies: List of dependency specifications (e.g., ['transformers==4.51.3', 'torch>=2.0.0'])
    """

    backend: str
    dependencies: List[str]

    def to_pip_args(self) -> str:
        """
        Convert dependencies to pip arguments format.

        Returns:
            String in format "--pip-args='dep1 dep2 dep3'"
        """
        if not self.dependencies:
            return ""

        deps_str = " ".join(self.dependencies)
        return f"--pip-args='{deps_str}'"


class BackendDependencyManager:
    """
    Manages backend dependencies for different inference backends.

    Examples:
        - model_env: {"GPUSTACK_BACKEND_DEPS"="transformers==4.53.3,torch>=2.0.0"}
    """

    def __init__(self, backend: str, version: str, model_env: Dict[str, str] = None):
        self.backend = backend
        self.version = version
        self._custom_specs: BackendDependencySpec = None

        # Initialize default dependencies for each backend using version specifiers
        # Format: {backend: {version_specifier: [dependencies]}}
        self.default_dependencies_specs: Dict[str, Dict[str, List[str]]] = {
            BackendEnum.VOX_BOX: {
                "<=0.0.20": ["transformers==4.51.3"],
            },
            BackendEnum.VLLM: {
                "<=0.10.0": ["transformers==4.53.3"],
            },
        }

        self._load_from_environment(model_env)

    def _load_from_environment(self, model_env: Dict[str, str] = None):
        """
        Load custom dependency specifications from model environment variables.

        Environment variable format:
        GPUSTACK_BACKEND_DEPS="dep1,dep2"
        """
        if not model_env:
            return
        # First try to get from model_env, then fallback to system environment
        env_deps = model_env.get("GPUSTACK_BACKEND_DEPS")

        if not env_deps:
            return

        try:
            dependencies = [dep.strip() for dep in env_deps.split(",") if dep.strip()]
            self._custom_specs = BackendDependencySpec(
                backend=self.backend, dependencies=dependencies
            )
            logger.info(f"Loaded custom dependency spec: {dependencies}")
        except Exception as e:
            logger.warning(f"Failed to parse GPUSTACK_BACKEND_DEPS: {e}")

    def get_dependency_spec(self) -> BackendDependencySpec:
        """
        Get dependency specification for a backend and version.

        Returns:
            BackendDependencySpec with custom or default dependencies
        """
        # First check for legacy format (backend:version)
        if self._custom_specs:
            return self._custom_specs

        # Fall back to default dependencies using version specifiers
        default_version_deps = self.default_dependencies_specs.get(self.backend, {})
        if not default_version_deps:
            return None

        # Normalize version by removing 'v' prefix if present
        normalized_version = self.version.lstrip('v')

        try:
            version_obj = Version(normalized_version)
        except Exception as e:
            logger.warning(
                f"Invalid version format '{self.version}' for backend {self.backend}: {e}"
            )
            return None

        # Check each version specifier to find a match
        for version_spec, dependencies in default_version_deps.items():
            specifier_set = SpecifierSet(version_spec)
            if version_obj in specifier_set:
                logger.debug(
                    f"Found matching dependency spec for {self.backend} {self.version}: {version_spec}"
                )
                return BackendDependencySpec(
                    backend=self.backend, dependencies=dependencies
                )

        return None

    def get_pipx_install_args(self) -> List[str]:
        """
        Get pipx installation arguments for a backend.

        Args:
            backend: Backend name
            version: Backend version

        Returns:
            List of additional arguments for pipx install command
        """
        spec = self.get_dependency_spec()
        if not spec or not spec.dependencies:
            return []

        pip_args = spec.to_pip_args()
        return [pip_args] if pip_args else []
