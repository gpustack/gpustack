import pytest

from gpustack.utils.envs import filter_env_vars, sanitize_env


@pytest.mark.parametrize(
    "name, env, expected",
    [
        ("Empty", {}, {}),
        (
            "Insensitive",
            {
                "SOME_OTHER_ENV": "value",
                "ANOTHER_ENV": "another_value",
            },
            {
                "SOME_OTHER_ENV": "value",
                "ANOTHER_ENV": "another_value",
            },
        ),
        (
            "Prefixes",
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "GPUSTACK_WORKER_ID": "worker-1",
                "GPUSTACK_WORKER_NAME": "worker-name",
                "GPUSTACK_WORKER_TYPE": "worker-type",
            },
            {
                "CUDA_VISIBLE_DEVICES": "0",
            },
        ),
        (
            "Suffixes",
            {
                "HF_HOME": "/path/to/hf_home",
                "HF_KEY": "",
                "hf_key": "",
                "HF_TOKEN": "",
                "hf_token": "",
                "ABC_SECRET": "",
                "abc_secret": "",
                "XYZ_PASSWORD": "",
                "xyz_password": "",
                "XYZ_PASS": "",
                "xyz_pass": "",
            },
            {
                "HF_HOME": "/path/to/hf_home",
            },
        ),
    ],
)
def test_sanitize_env(name, env, expected):
    actual = sanitize_env(env)
    assert actual == expected, f"Case {name} expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    "name, env, expected",
    [
        ("Empty", {}, {}),
        (
            "MigCapabilityDeclarationsDropped",
            {
                # Declared by the worker DaemonSet; a model pod inheriting them
                # is rejected by the NVIDIA runtime's CDI modifier as
                # non-privileged.
                "NVIDIA_MIG_CONFIG_DEVICES": "all",
                "NVIDIA_MIG_MONITOR_DEVICES": "all",
                "SOME_OTHER_ENV": "value",
            },
            {"SOME_OTHER_ENV": "value"},
        ),
        (
            "OtherRuntimeVarsDropped",
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "NVIDIA_DISABLE_REQUIRE": "1",
            },
            {},
        ),
    ],
)
def test_filter_env_vars(name, env, expected):
    actual = filter_env_vars(env)
    assert actual == expected, f"Case {name} expected {expected}, but got {actual}"
