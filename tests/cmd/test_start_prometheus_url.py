"""``--prometheus-url``, the flag that was missing next to ``--grafana-url``.

`Config.prometheus_url` has existed and been read by the metric-backed
features all along, but only the config file and the environment could set
it: there was no CLI flag, so `gpustack start --prometheus-url ...` exited
with `unrecognized arguments` and the operator's next guess was usually that
the field did not exist. Its sibling `--grafana-url` did have one, which is
what made the gap look like a typo rather than an omission.
"""

import argparse

import pytest

from gpustack.cmd.start import set_server_options, setup_start_cmd
from gpustack.config.config import Config


def _parse(argv):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest="command"))
    return parser.parse_args(["start", *argv])


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    """The default is read from the environment when the parser is built, so
    an ambient GPUSTACK_PROMETHEUS_URL would silently pass these tests."""
    monkeypatch.delenv("GPUSTACK_PROMETHEUS_URL", raising=False)


def test_flag_is_accepted():
    assert _parse(["--prometheus-url", "http://127.0.0.1:9090"]).prometheus_url == (
        "http://127.0.0.1:9090"
    )


def test_flag_defaults_to_none():
    assert _parse([]).prometheus_url is None


def test_flag_default_comes_from_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_PROMETHEUS_URL", "http://prom:9090")
    assert _parse([]).prometheus_url == "http://prom:9090"


def test_flag_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("GPUSTACK_PROMETHEUS_URL", "http://prom:9090")
    assert _parse(["--prometheus-url", "http://other:9090"]).prometheus_url == (
        "http://other:9090"
    )


def test_flag_reaches_the_config(tmp_path):
    """The whole point of the flag: the server's own metric queries resolve
    through `config.prometheus_url`, so an option the collector does not
    forward is a no-op that looks like it works."""
    config_data = {"data_dir": str(tmp_path / "data")}
    set_server_options(
        _parse(["--prometheus-url", "http://127.0.0.1:9090"]), config_data
    )
    assert config_data["prometheus_url"] == "http://127.0.0.1:9090"
    assert Config(**config_data).prometheus_url == "http://127.0.0.1:9090"


def test_unset_flag_does_not_overwrite_the_config_file_value(tmp_path):
    """An external Prometheus is usually pinned in the YAML; a None from
    argparse must not clobber it."""
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "prometheus_url": "http://from-yaml:9090",
    }
    set_server_options(_parse([]), config_data)
    assert Config(**config_data).prometheus_url == "http://from-yaml:9090"


def test_it_sits_alongside_grafana_url(tmp_path):
    """The two are set together on a deployment that delegates observability,
    and the pairing is the reason the missing flag was a surprise."""
    config_data = {"data_dir": str(tmp_path / "data")}
    set_server_options(
        _parse(
            [
                "--disable-builtin-observability",
                "--prometheus-url",
                "http://127.0.0.1:9090",
                "--grafana-url",
                "http://host:3000",
            ]
        ),
        config_data,
    )
    config = Config(**config_data)
    assert config.prometheus_url == "http://127.0.0.1:9090"
    assert config.grafana_url == "http://host:3000"
    assert config.disable_builtin_observability is True
