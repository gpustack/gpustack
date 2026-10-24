import pytest

from gpustack.config.config import Config


def _config(tmp_path, external_url):
    return Config(data_dir=str(tmp_path / "data"), server_external_url=external_url)


@pytest.mark.parametrize(
    "external_url, base_path",
    [
        # Nothing to go on: the server is at the root until told otherwise.
        (None, ""),
        # A URL with no path means the root, and so does a bare "/" — a prefix of
        # "/" would only add an empty path segment to everything.
        ("https://example.com", ""),
        ("https://example.com/", ""),
        ("https://example.com:30080", ""),
        ("https://example.com/gpustack", "/gpustack"),
        # Normalised to no trailing slash, because both consumers want it that
        # way: ASGI root_path is concatenated with a path that has its own
        # leading slash, and a cookie ``Path`` compares segment by segment.
        ("https://example.com/gpustack/", "/gpustack"),
        ("https://example.com:30080/gpustack", "/gpustack"),
        # More than one segment deep is legal and has to survive intact.
        ("https://example.com/inner/gpustack", "/inner/gpustack"),
    ],
)
def test_base_path_is_the_path_component_of_the_external_url(
    tmp_path, external_url, base_path
):
    assert _config(tmp_path, external_url).get_base_path() == base_path


@pytest.mark.parametrize("external_url", [None, "https://example.com"])
def test_cookie_path_at_the_root_is_a_slash_not_empty(tmp_path, external_url):
    # "" is the right answer for root_path — it is FastAPI's own default — but an
    # empty ``Path`` is not a valid cookie attribute, so the two diverge here.
    assert _config(tmp_path, external_url).get_base_path() == ""
    assert _config(tmp_path, external_url).get_cookie_path() == "/"


def test_cookie_path_is_scoped_to_the_mount_prefix(tmp_path):
    # Not "/": under a subpath mount the rest of the origin is typically the
    # customer's own application, which has no business being offered a GPUStack
    # session cookie.
    config = _config(tmp_path, "https://example.com/gpustack")

    assert config.get_cookie_path() == "/gpustack"
