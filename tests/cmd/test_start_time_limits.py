import argparse

import pytest
import yaml

from gpustack import envs
from gpustack.cmd import start


@pytest.fixture
def parse_config(monkeypatch, tmp_path):
    monkeypatch.delenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", raising=False)
    monkeypatch.delenv("GPUSTACK_LOG_STREAM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(start, "set_global_config", lambda _: None)

    def parse(*flags, **settings):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.safe_dump({"data_dir": str(tmp_path), **settings}), encoding="utf-8"
        )
        parser = argparse.ArgumentParser()
        start.setup_start_cmd(parser.add_subparsers())
        args = parser.parse_args(["start", "--config-file", str(config_file), *flags])
        return start.parse_args(args)

    return parse


def test_default_time_limits_and_legacy_log_fallback(parse_config, monkeypatch):
    monkeypatch.setattr(envs, "PROXY_TIMEOUT", 1800)
    cfg = parse_config()
    assert cfg.jwt_token_expire_minutes == 120
    assert cfg.log_stream_timeout_seconds == 1800

    monkeypatch.setattr(envs, "PROXY_TIMEOUT", 7200)
    assert parse_config().log_stream_timeout_seconds == 7200
    assert parse_config(log_stream_timeout_seconds=0).log_stream_timeout_seconds == 0


def test_time_limit_configuration_sources(parse_config, monkeypatch):
    settings = dict(jwt_token_expire_minutes=30, log_stream_timeout_seconds=3600)
    cfg = parse_config(**settings)
    assert (cfg.jwt_token_expire_minutes, cfg.log_stream_timeout_seconds) == (30, 3600)

    monkeypatch.setenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", "15")
    monkeypatch.setenv("GPUSTACK_LOG_STREAM_TIMEOUT_SECONDS", "600")
    cfg = parse_config(**settings)
    assert (cfg.jwt_token_expire_minutes, cfg.log_stream_timeout_seconds) == (15, 600)

    cfg = parse_config(
        "--jwt-token-expire-minutes",
        "5",
        "--log-stream-timeout-seconds",
        "0",
        **settings,
    )
    assert (cfg.jwt_token_expire_minutes, cfg.log_stream_timeout_seconds) == (5, 0)


@pytest.mark.parametrize(
    "setting, value",
    [("jwt_token_expire_minutes", 0), ("log_stream_timeout_seconds", -1)],
)
def test_invalid_time_limit_is_rejected(parse_config, setting, value):
    with pytest.raises(Exception, match=setting):
        parse_config(**{setting: value})
