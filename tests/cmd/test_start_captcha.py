"""Regression tests for login CAPTCHA server configuration."""

import argparse
import sys

import pytest

if sys.platform == "win32":
    pytest.skip(
        "The server CLI imports Unix-only worker modules.",
        allow_module_level=True,
    )

from gpustack.cmd.start import set_server_options, start_cmd_options


def _parse_server_options(arguments):
    parser = argparse.ArgumentParser()
    start_cmd_options(parser)
    args = parser.parse_args(arguments)
    server_options = {}
    set_server_options(args, server_options)
    return server_options


def test_captcha_cli_options_are_forwarded_to_server_config():
    server_options = _parse_server_options(
        ["--enable-login-captcha", "--login-captcha-length", "6"]
    )

    assert server_options["enable_login_captcha"] is True
    assert server_options["login_captcha_length"] == 6
