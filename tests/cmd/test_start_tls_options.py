import argparse

from gpustack.cmd.start import set_server_options, setup_start_cmd


def _config_data(argv: list) -> dict:
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers())
    args = parser.parse_args(["start"] + argv)
    config_data = {}
    set_server_options(args, config_data)
    return config_data


def test_ssl_ca_certfile_flag_reaches_server_config():
    assert _config_data(["--ssl-ca-certfile", "/certs/ca.pem"]) == {
        "ssl_ca_certfile": "/certs/ca.pem"
    }


def test_ssl_ca_certfile_environment_variable_reaches_server_config(monkeypatch):
    monkeypatch.setenv("GPUSTACK_SSL_CA_CERTFILE", "/certs/ca.pem")

    assert _config_data([]) == {"ssl_ca_certfile": "/certs/ca.pem"}
