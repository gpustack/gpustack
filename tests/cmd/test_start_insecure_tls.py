import argparse
import os

import pytest

from gpustack import envs
from gpustack.cmd.start import (
    export_insecure_tls_env,
    set_common_options,
    setup_start_cmd,
)
from gpustack.config.config import Config
from gpustack.schemas.config import SensitivePredefinedConfig


def _parse(argv):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    return parser.parse_args(["start", *argv])


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    """The flag default is read from the environment at parser construction
    time, so an ambient GPUSTACK_INSECURE_TLS would silently pass these
    tests."""
    monkeypatch.delenv(envs.INSECURE_TLS_ENV, raising=False)


def test_flag_is_accepted():
    assert _parse(["--insecure-tls"]).insecure_tls is True


def test_flag_defaults_to_none():
    assert _parse([]).insecure_tls is None


def test_flag_default_comes_from_the_env_var(monkeypatch):
    monkeypatch.setenv(envs.INSECURE_TLS_ENV, "true")
    assert _parse([]).insecure_tls is True


def test_flag_reaches_the_config(tmp_path):
    config_data = {"data_dir": str(tmp_path / "data")}
    set_common_options(_parse(["--insecure-tls"]), config_data)
    assert config_data["insecure_tls"] is True
    assert Config(**config_data).insecure_tls is True


def test_unset_flag_does_not_overwrite_the_config_file_value(tmp_path):
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "insecure_tls": True,
    }
    set_common_options(_parse([]), config_data)
    assert Config(**config_data).insecure_tls is True


def test_export_insecure_tls_env_reaches_subprocesses(monkeypatch, tmp_path):
    """The benchmark runner and the embedded worker read the env var at
    import and have no global config, so a config-file / ``--insecure-tls``
    setting has to be exported as env before they are spawned."""
    # Registered with monkeypatch so teardown removes whatever the function
    # writes, instead of leaking it into later tests.
    monkeypatch.setenv(envs.INSECURE_TLS_ENV, "false")
    cfg = Config(data_dir=str(tmp_path / "data"), insecure_tls=True)

    export_insecure_tls_env(cfg)

    assert os.environ[envs.INSECURE_TLS_ENV] == "true"


def test_export_insecure_tls_env_noop_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv(envs.INSECURE_TLS_ENV, "false")
    cfg = Config(data_dir=str(tmp_path / "data"))

    export_insecure_tls_env(cfg)

    assert os.environ[envs.INSECURE_TLS_ENV] == "false"


def test_insecure_tls_rides_the_registration_env_channel():
    """The cluster channel for ``insecure_tls`` is the registration command
    (Cluster.worker_config -> SensitiveRegistrationConfig ->
    GPUSTACK_INSECURE_TLS), the delivery read before the worker's first TLS
    handshake; that membership is what ``SensitivePredefinedConfig``
    expresses."""
    assert "insecure_tls" in SensitivePredefinedConfig.model_fields
