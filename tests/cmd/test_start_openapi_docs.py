import argparse

import pytest

from gpustack.cmd.start import set_server_options, setup_start_cmd
from gpustack.config.config import Config
from gpustack.schemas.config import PredefinedConfigNoDefaults


@pytest.mark.parametrize(
    "argv, environment, file_value, expected",
    [
        ([], None, None, False),
        (["--disable-openapi-docs"], None, None, True),
        ([], "true", None, True),
        ([], "false", None, False),
        ([], None, True, True),
        ([], None, False, False),
        (["--disable-openapi-docs"], None, False, True),
        ([], "false", True, False),
        (["--disable-openapi-docs"], "false", False, True),
    ],
)
def test_openapi_setting_reaches_standalone_worker_config(
    monkeypatch, tmp_path, argv, environment, file_value, expected
):
    monkeypatch.delenv("GPUSTACK_DISABLE_OPENAPI_DOCS", raising=False)
    if environment is not None:
        monkeypatch.setenv("GPUSTACK_DISABLE_OPENAPI_DOCS", environment)
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["start", *argv])
    config_data = {
        "data_dir": str(tmp_path),
        "server_url": "http://server",
        "token": "test",
    }
    if file_value is not None:
        config_data["disable_openapi_docs"] = file_value
    set_server_options(args, config_data)
    cfg = Config(**config_data)

    assert cfg.disable_openapi_docs is expected
    assert cfg.server_role() == Config.ServerRole.WORKER

    cfg.reload_worker_config(PredefinedConfigNoDefaults(worker_metrics_port=10152))

    assert cfg.disable_openapi_docs is expected
