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
    "ram,expected",
    [(None, 2), ("0", 0), (str(4 * 1024**3), 4), ("invalid", 2), ("-1", 2)],
)
def test_non_model_resources_use_explicit_ram_or_router_defaults(ram, expected):
    env = {} if ram is None else {"GPUSTACK_MODEL_RAM_CLAIM": ram}
    resources = non_model_workload_resources(SimpleNamespace(env=env))
    assert resources.cpu == 2
    assert resources.memory == expected * 1024**3
