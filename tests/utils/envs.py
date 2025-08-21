import pytest

from gpustack.utils.envs import sanitize_env


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
