from gpustack.utils.api_keys import get_masked_api_key_value


def test_get_masked_api_key_value_shows_access_key_prefix():
    assert get_masked_api_key_value("abcd1234") == "gpustack_abcd***"


def test_get_masked_api_key_value_handles_short_access_key():
    assert get_masked_api_key_value("abcd") == "gpustack_***"


def test_get_masked_api_key_value_handles_custom_key():
    assert get_masked_api_key_value("abcd1234", is_custom=True) == "Custom API Key"
