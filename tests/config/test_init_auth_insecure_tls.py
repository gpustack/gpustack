"""How ``insecure_tls`` reaches the OIDC discovery fetch at startup.

The fetch runs inside ``Config.__init__`` -- before the global config is
registered -- so ``make_ssl_context`` cannot see the setting there; the
pass-through is what keeps a self-signed IdP bootable and stops the
context cache from being seeded with a verifying context first.
"""

from unittest import mock

from gpustack.config import config as config_module
from gpustack.config.config import Config


def _config(**overrides) -> Config:
    return Config(**overrides)


def test_oidc_discovery_skips_verification_when_insecure_tls(tmp_path):
    with mock.patch.object(config_module, "get_openid_configuration") as fetch:
        _config(
            data_dir=str(tmp_path / "data"),
            oidc_issuer="https://idp.example.com",
            insecure_tls=True,
        )

    assert fetch.call_args.kwargs["insecure_skip_tls_verify"] is True


def test_oidc_discovery_skips_verification_when_external_auth_flag_set(tmp_path):
    with mock.patch.object(config_module, "get_openid_configuration") as fetch:
        _config(
            data_dir=str(tmp_path / "data"),
            oidc_issuer="https://idp.example.com",
            external_auth_insecure_skip_tls_verify=True,
        )

    assert fetch.call_args.kwargs["insecure_skip_tls_verify"] is True


def test_oidc_discovery_verifies_by_default(tmp_path):
    with mock.patch.object(config_module, "get_openid_configuration") as fetch:
        _config(
            data_dir=str(tmp_path / "data"),
            oidc_issuer="https://idp.example.com",
        )

    assert fetch.call_args.kwargs["insecure_skip_tls_verify"] is False


def test_make_ssl_context_is_not_called_during_config_construction(tmp_path):
    """Seeding the factory's cache here would pin a verifying context for
    the whole process, silently neutralizing --insecure-tls."""
    with mock.patch.object(config_module, "get_openid_configuration"):
        with mock.patch.object(
            config_module,
            "make_ssl_context",
            side_effect=AssertionError("must not be reached during __init__"),
        ):
            _config(
                data_dir=str(tmp_path / "data"),
                oidc_issuer="https://idp.example.com",
                insecure_tls=True,
            )


def test_insecure_tls_defaults_to_none(tmp_path):
    assert _config(data_dir=str(tmp_path / "data")).insecure_tls is None
