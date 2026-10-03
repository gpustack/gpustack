import shutil
import subprocess

import pytest

from gpustack.cmd.prerun import prepare_gateway_config, prepare_s6_overlay
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum


def test_prepare_s6_overlay_enables_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)

    stale_migration = user_contents / "gpustack-migration"
    stale_migration.write_text("")

    # prepare_s6_overlay should cleanup the base dir and generate base on the input services
    prepare_s6_overlay(["postgres"], ["gpustack-migration"], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert (user_contents / "gpustack-migration").exists()


def test_prepare_s6_overlay_cleans_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)
    (user_contents / "gpustack-migration").write_text("")

    prepare_s6_overlay(["postgres"], [], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert not (user_contents / "gpustack-migration").exists()


def _source_with_bash(env_file, *names):
    """Read variables back the way the s6 service scripts do: `source`."""
    script = 'source "$1"; shift; for n in "$@"; do printf "%s\\0" "${!n}"; done'
    result = subprocess.run(
        ["bash", "-c", script, "_", str(env_file), *names],
        capture_output=True,
        check=True,
    )
    assert result.stderr == b""
    return dict(zip(names, result.stdout.decode().split("\0")))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
@pytest.mark.parametrize(
    "dir_name",
    ["my data", "x$(echo INJECTED)", "x`echo INJECTED`", "it's; echo INJECTED"],
)
def test_prepare_gateway_config_values_survive_bash_source(
    tmp_path, monkeypatch, dir_name
):
    data_dir = tmp_path / dir_name
    env_file = tmp_path / "run" / "gateway" / ".env"
    monkeypatch.setenv("GPUSTACK_GATEWAY_CONFIG", str(env_file))
    cfg = Config(data_dir=str(data_dir), gateway_mode=GatewayModeEnum.embedded)

    prepare_gateway_config(cfg)

    values = _source_with_bash(
        env_file, "DATA_DIR", "LOG_DIR", "EMBEDDED_KUBECONFIG_PATH"
    )
    assert values == {
        "DATA_DIR": cfg.data_dir,
        "LOG_DIR": cfg.log_dir,
        "EMBEDDED_KUBECONFIG_PATH": str(data_dir / "higress" / "kubeconfig"),
    }
