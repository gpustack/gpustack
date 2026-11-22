from types import SimpleNamespace

import pytest

from gpustack.utils.model_resources import (
    non_model_workload_enabled,
    non_model_workload_resources,
)


@pytest.mark.parametrize("value", ["true", " TRUE ", "1", "yes", "on"])
def test_non_model_workload_is_an_explicit_deployment_opt_in(value):
    assert non_model_workload_enabled(
        SimpleNamespace(env={"GPUSTACK_NON_MODEL_WORKLOAD": value})
    )


@pytest.mark.parametrize("value", [None, "", "false", "0", "no", "off", "invalid"])
def test_disabled_non_model_workload_does_not_follow_worker_environment(
    monkeypatch, value
):
    monkeypatch.setenv("GPUSTACK_NON_MODEL_WORKLOAD", "true")
    env = {} if value is None else {"GPUSTACK_NON_MODEL_WORKLOAD": value}
    assert not non_model_workload_enabled(SimpleNamespace(env=env))


@pytest.mark.parametrize(
    "cpu,ram",
    [(None, None), ("0.25", None), (None, 64 * 1024**2), ("1.5", 4 * 1024**3)],
)
def test_non_model_container_resources_require_explicit_declarations(cpu, ram):
    env = {}
    if cpu is not None:
        env["GPUSTACK_MODEL_CPU_CLAIM"] = cpu
    if ram is not None:
        env["GPUSTACK_MODEL_RAM_CLAIM"] = str(ram)
    resources = non_model_workload_resources(SimpleNamespace(env=env))
    assert resources.cpu == (float(cpu) if cpu is not None else None)
    assert resources.memory == ram
