from typing import Optional, Union
from gpustack_runtime.envs import (
    GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE,
    GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE,
)
from gpustack_runtime.deployer.docker import DockerWorkloadPlan
from gpustack_runtime.deployer import (
    DockerDeployer,
    KubernetesDeployer,
    KubernetesResourceInjectionPolicyEnum,
    KubernetesWorkloadPlan,
    WorkloadPlan,
    WorkloadStatus,
)

from gpustack.config.config import Config
from gpustack.schemas.cache_services import CACHE_SERVICE_WORKLOAD_TYPE
from gpustack.utils.config import apply_registry_override_to_image


def transform_workload_plan(
    config: Config,
    workload: WorkloadPlan,
    fallback_registry: Optional[str] = None,
) -> Union[DockerWorkloadPlan, KubernetesWorkloadPlan, WorkloadPlan]:
    """
    If the deployer is docker, transform the generic WorkloadPlan to DockerWorkloadPlan,
    and fill the pause image and restart image with registry override.
    Kubernetes cache services use Env injection to access devices without
    consuming device-plugin allocations reserved for model workloads.
    """
    if not DockerDeployer().is_supported():
        if is_cache_service_workload(workload) and KubernetesDeployer.is_supported():
            kubernetes_workload = KubernetesWorkloadPlan(**workload.__dict__)
            kubernetes_workload.resource_injection_policy = (
                KubernetesResourceInjectionPolicyEnum.ENV
            )
            return kubernetes_workload
        return workload
    pause_image = apply_registry_override_to_image(
        config, GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE, fallback_registry
    )
    restart_image = apply_registry_override_to_image(
        config, GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE, fallback_registry
    )
    docker_workload = DockerWorkloadPlan(
        pause_image=pause_image,
        unhealthy_restart_image=restart_image,
        **workload.__dict__,
    )
    return docker_workload


def is_benchmark_workload(status: WorkloadStatus) -> bool:
    """
    Check if a workload is a benchmark workload.

    A workload is considered a benchmark workload if it has the 'type' label
    set to 'benchmark'.

    Args:
        status: The workload status to check.

    Returns:
        True if the workload is a benchmark workload, False otherwise.
    """
    if not status.labels:
        return False
    return status.labels.get("type") == "benchmark"


def is_cache_service_workload(status: Union[WorkloadPlan, WorkloadStatus]) -> bool:
    """
    Check if a workload is a cache service workload.

    A workload is considered a cache service workload if it has the 'type'
    label set to 'cache-service'.

    Args:
        status: The workload status to check.

    Returns:
        True if the workload is a cache service workload, False otherwise.
    """
    if not status.labels:
        return False
    return status.labels.get("type") == CACHE_SERVICE_WORKLOAD_TYPE
