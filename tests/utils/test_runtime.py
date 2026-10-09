import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gpustack_runtime.deployer import (
    Container,
    KubernetesResourceInjectionPolicyEnum,
    KubernetesWorkloadPlan,
    WorkloadPlan,
)
from gpustack_runtime.deployer.docker import DockerWorkloadPlan

from gpustack.utils.runtime import transform_workload_plan


@pytest.mark.parametrize("kubernetes", [True, False])
@pytest.mark.parametrize("workload_type", ["model-instance", "benchmark", None])
def test_other_workloads_keep_their_plan(kubernetes, workload_type):
    plan = WorkloadPlan(labels={"type": workload_type})
    with (
        patch("gpustack.utils.runtime.DockerDeployer.is_supported", return_value=False),
        patch(
            "gpustack.utils.runtime.KubernetesDeployer.is_supported",
            return_value=kubernetes,
        ),
    ):
        assert transform_workload_plan(None, plan) is plan


@pytest.mark.parametrize("specialized", [True, False])
def test_kubernetes_cache_plan_sets_policy_and_survives_process_transfer(specialized):
    plan_type = KubernetesWorkloadPlan if specialized else WorkloadPlan
    plan = plan_type(
        name="cache-svc-1-i1",
        labels={"type": "cache-service"},
        host_ipc=True,
        host_network=True,
        containers=[Container(name="default", image="example.com/cache:test")],
    )
    with (
        patch("gpustack.utils.runtime.DockerDeployer.is_supported", return_value=False),
        patch(
            "gpustack.utils.runtime.KubernetesDeployer.is_supported", return_value=True
        ),
    ):
        result = transform_workload_plan(None, plan)
    assert isinstance(result, KubernetesWorkloadPlan)
    assert result.resource_injection_policy is KubernetesResourceInjectionPolicyEnum.ENV
    assert result.containers is plan.containers
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert (
        restored.resource_injection_policy is KubernetesResourceInjectionPolicyEnum.ENV
    )


def test_cache_plan_without_docker_or_kubernetes_is_unchanged():
    plan = WorkloadPlan(labels={"type": "cache-service"})
    with (
        patch("gpustack.utils.runtime.DockerDeployer.is_supported", return_value=False),
        patch(
            "gpustack.utils.runtime.KubernetesDeployer.is_supported", return_value=False
        ),
    ):
        assert transform_workload_plan(None, plan) is plan


def test_docker_cache_plan_keeps_registry_overrides():
    plan = WorkloadPlan(labels={"type": "cache-service"}, host_ipc=True)
    config = SimpleNamespace(system_default_container_registry="registry.example.com")
    with (
        patch("gpustack.utils.runtime.DockerDeployer.is_supported", return_value=True),
        patch("gpustack.utils.runtime.KubernetesDeployer.is_supported") as kubernetes,
        patch(
            "gpustack.utils.runtime.GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE", "pause:3.9"
        ),
        patch(
            "gpustack.utils.runtime.GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE",
            "restart:latest",
        ),
    ):
        result = transform_workload_plan(config, plan)
    assert isinstance(result, DockerWorkloadPlan)
    assert result.host_ipc is True
    assert result.pause_image == "registry.example.com/pause:3.9"
    assert result.unhealthy_restart_image == "registry.example.com/restart:latest"
    kubernetes.assert_not_called()
