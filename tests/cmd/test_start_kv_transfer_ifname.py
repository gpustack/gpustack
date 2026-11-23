import argparse

import pytest

from gpustack.cmd.start import set_worker_options, setup_start_cmd
from gpustack.config.config import Config
from gpustack.schemas.config import PredefinedConfig, PredefinedConfigNoDefaults


def _parse(argv):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    return parser.parse_args(["start", *argv])


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    """The flag defaults are read from the environment at parser construction
    time, so an ambient GPUSTACK_KV_TRANSFER_IFNAME would silently pass these tests."""
    monkeypatch.delenv("GPUSTACK_KV_TRANSFER_IFNAME", raising=False)


def test_flag_is_accepted():
    assert _parse(["--kv-transfer-ifname", "ib0"]).kv_transfer_ifname == "ib0"


def test_flag_help_describes_the_nixl_use_case(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parse(["--help"])
    assert exit_info.value.code == 0
    assert "KV transfer in built-in vLLM NIXL and SGLang NIXL" in " ".join(
        capsys.readouterr().out.split()
    )


def test_flag_defaults_to_none():
    assert _parse([]).kv_transfer_ifname is None


def test_flag_default_comes_from_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_KV_TRANSFER_IFNAME", "ib1")
    assert _parse([]).kv_transfer_ifname == "ib1"


def test_flag_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_KV_TRANSFER_IFNAME", "ib1")
    assert _parse(["--kv-transfer-ifname", "ib0"]).kv_transfer_ifname == "ib0"


def test_flag_reaches_the_config(tmp_path):
    """The whole point of the flag: ``derive_net_device`` reads
    ``config.kv_transfer_ifname``, so an option the collector does not forward is a
    no-op that looks like it works."""
    config_data = {"data_dir": str(tmp_path / "data")}
    set_worker_options(_parse(["--kv-transfer-ifname", "ib0"]), config_data)
    assert config_data["kv_transfer_ifname"] == "ib0"
    assert Config(**config_data).kv_transfer_ifname == "ib0"


def test_unset_flag_does_not_overwrite_the_config_file_value(tmp_path):
    """``--kv-transfer-ifname`` is per-host and often set in the YAML instead; a None
    from argparse must not clobber it."""
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "kv_transfer_ifname": "from-yaml",
    }
    set_worker_options(_parse([]), config_data)
    assert Config(**config_data).kv_transfer_ifname == "from-yaml"


def test_kv_transfer_ifname_stays_out_of_shared_startup_schema():
    assert "kv_transfer_ifname" not in PredefinedConfig.model_fields
    assert "kv_transfer_ifname" in Config.model_fields


def test_worker_reload_keeps_the_cluster_kv_default_out_of_startup_config(tmp_path):
    worker_config = PredefinedConfigNoDefaults.model_validate(
        {"kv_transfer_ifname": "mlx5_2:1", "worker_metrics_port": 10152}
    )
    assert worker_config.model_dump().get("kv_transfer_ifname") == "mlx5_2:1"

    cfg = Config(data_dir=str(tmp_path / "data"))
    cfg.reload_worker_config(worker_config)

    assert cfg.kv_transfer_ifname is None
    assert cfg.worker_metrics_port == 10152
