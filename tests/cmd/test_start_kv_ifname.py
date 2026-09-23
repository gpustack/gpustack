import argparse

import pytest

from gpustack.cmd.start import set_worker_options, setup_start_cmd
from gpustack.config.config import Config
from gpustack.schemas.config import PredefinedConfig


def _parse(argv):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    return parser.parse_args(["start", *argv])


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    """The flag defaults are read from the environment at parser construction
    time, so an ambient GPUSTACK_KV_IFNAME would silently pass these tests."""
    monkeypatch.delenv("GPUSTACK_KV_IFNAME", raising=False)


def test_flag_is_accepted():
    assert _parse(["--kv-ifname", "ib0"]).kv_ifname == "ib0"


def test_flag_defaults_to_none():
    assert _parse([]).kv_ifname is None


def test_flag_default_comes_from_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_KV_IFNAME", "ib1")
    assert _parse([]).kv_ifname == "ib1"


def test_flag_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_KV_IFNAME", "ib1")
    assert _parse(["--kv-ifname", "ib0"]).kv_ifname == "ib0"


def test_flag_reaches_the_config(tmp_path):
    """The whole point of the flag: ``derive_net_device`` reads
    ``config.kv_ifname``, so an option the collector does not forward is a
    no-op that looks like it works."""
    config_data = {"data_dir": str(tmp_path / "data")}
    set_worker_options(_parse(["--kv-ifname", "ib0"]), config_data)
    assert config_data["kv_ifname"] == "ib0"
    assert Config(**config_data).kv_ifname == "ib0"


def test_unset_flag_does_not_overwrite_the_config_file_value(tmp_path):
    """``--kv-ifname`` is per-host and often set in the YAML instead; a None
    from argparse must not clobber it."""
    config_data = {"data_dir": str(tmp_path / "data"), "kv_ifname": "from-yaml"}
    set_worker_options(_parse([]), config_data)
    assert Config(**config_data).kv_ifname == "from-yaml"


def test_kv_ifname_stays_out_of_the_cluster_wide_channel():
    """``PredefinedConfig`` is broadcast to every worker in the cluster (via
    ``Cluster.worker_config`` -> the registration env). A NIC name belongs to
    one machine, so promoting this field would hand ``ib0`` to hosts that have
    no ``ib0``."""
    assert "kv_ifname" not in PredefinedConfig.model_fields
    assert "kv_ifname" in Config.model_fields
