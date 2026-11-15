import argparse

import pytest

from gpustack.cmd.start import set_common_options, setup_start_cmd
from gpustack.config.config import Config


def _parse(argv):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    return parser.parse_args(["start", *argv])


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    monkeypatch.delenv("GPUSTACK_CONTAINER_NAMESPACE", raising=False)


def test_flag_default_comes_from_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_CONTAINER_NAMESPACE", "gpustack-ai")
    assert _parse([]).container_namespace == "gpustack-ai"


def test_flag_reaches_the_config(tmp_path):
    config_data = {"data_dir": str(tmp_path / "data")}
    set_common_options(_parse(["--container-namespace", "gpustack-ai"]), config_data)
    assert Config(**config_data).container_namespace == "gpustack-ai"


def test_unset_flag_does_not_overwrite_the_config_file_value(tmp_path):
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "container_namespace": "from-yaml",
    }
    set_common_options(_parse([]), config_data)
    assert Config(**config_data).container_namespace == "from-yaml"
