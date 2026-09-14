"""``--external-auth-provider-name`` plumbing: flag, environment variable and
config-file key must all reach ``Config``.

The ``/auth/config`` tests build a ``Config`` by hand, so they stay green even
if the option never makes it out of ``argparse`` — a typo in the environment
variable name, or a missing entry in the ``set_server_options`` whitelist,
would silently ignore the documented flag. These tests cover that gap.
"""

import argparse

import pytest

from gpustack.cmd.start import set_server_options, setup_start_cmd

ENV_VAR = "GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME"
OPTION = "external_auth_provider_name"


def _config_data(argv: list, config_file: dict = None) -> dict:
    """Parse ``argv`` the way ``gpustack start`` does and apply the server
    options on top of what a config file would have contributed."""
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers())
    args = parser.parse_args(["start"] + argv)

    config_data = dict(config_file or {})
    set_server_options(args, config_data)
    return config_data


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # The developer shell that runs the suite may export SSO settings.
    monkeypatch.delenv(ENV_VAR, raising=False)


def test_flag_reaches_config_data():
    assert _config_data(["--external-auth-provider-name", "Okta"])[OPTION] == "Okta"


def test_env_var_reaches_config_data(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "Campus SSO")
    assert _config_data([])[OPTION] == "Campus SSO"


def test_flag_beats_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "Campus SSO")
    assert _config_data(["--external-auth-provider-name", "Okta"])[OPTION] == "Okta"


def test_config_file_value_survives_without_a_flag():
    config_data = _config_data([], config_file={OPTION: "Campus SSO"})
    assert config_data[OPTION] == "Campus SSO"


def test_flag_beats_config_file():
    config_data = _config_data(
        ["--external-auth-provider-name", "Okta"],
        config_file={OPTION: "Campus SSO"},
    )
    assert config_data[OPTION] == "Okta"


def test_unset_leaves_the_option_out():
    assert OPTION not in _config_data([])
