import ssl

from gpustack.client import ClientSet
from gpustack.schemas.config import PredefinedConfig, PredefinedConfigNoDefaults


def test_insecure_tls_defaults_to_false():
    assert PredefinedConfig().insecure_tls is False
    assert PredefinedConfigNoDefaults().insecure_tls is None


def test_clientset_insecure_tls_skips_verification():
    client = ClientSet(base_url="https://example.com", insecure_tls=True)
    assert client.http_client._verify_ssl is False


def test_clientset_verifies_by_default():
    client = ClientSet(base_url="https://example.com")
    # Default path uses the process-wide ``make_ssl_context()`` context, not a
    # bare ``False`` (which would silently disable verification).
    assert isinstance(client.http_client._verify_ssl, ssl.SSLContext)
