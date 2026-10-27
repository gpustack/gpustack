import certifi
import pytest

from gpustack.config.config import Config


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not a certificate",
        b"-----BEGIN CERTIFICATE-----\ndGVzdA==\n-----END CERTIFICATE-----\n",
    ],
)
def test_config_rejects_invalid_ca_bundle(monkeypatch, tmp_path, content):
    path = tmp_path / "ca.pem"
    path.write_bytes(content)
    config = Config.model_construct(ssl_ca_certfile=str(path))
    monkeypatch.delenv("PYTEST_CURRENT_TEST")

    with pytest.raises(ValueError, match="contains"):
        config.check_all()


def test_config_accepts_valid_ca_bundle(monkeypatch):
    config = Config.model_construct(ssl_ca_certfile=certifi.where())
    monkeypatch.delenv("PYTEST_CURRENT_TEST")

    assert config.check_all() is config
