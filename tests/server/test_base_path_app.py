import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import gpustack
from gpustack.config.config import Config
from gpustack.server.app import create_app

# Same gate as tests/routes/test_ui_static.py: `hack/install.sh` skips the UI
# download when UI_DOWNLOAD=false, and create_app mounts from this directory.
UI_DIR = Path(gpustack.__file__).parent / "ui"

pytestmark = pytest.mark.skipif(
    not UI_DIR.is_dir(),
    reason="UI assets not downloaded; create_app requires gpustack/ui",
)

PREFIX = "/gpustack"


def _client(tmp_path, external_url):
    cfg = Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path / "data"),
        server_external_url=external_url,
    )
    return TestClient(create_app(cfg))


@pytest.fixture
def mounted(tmp_path):
    return _client(tmp_path, f"https://example.com{PREFIX}")


def _absolute_refs(html: str):
    """Every root-relative URL the docs page tells the browser to fetch."""
    return re.findall(r"""["'](/[^"']*)["']""", html)


@pytest.mark.parametrize(
    "requested",
    [
        # What a prefix-preserving proxy forwards...
        f"{PREFIX}/docs",
        f"{PREFIX}/openapi.json",
        f"{PREFIX}/version",
        # ...and what a stripping one does. Both have to work, because which of
        # the two a customer runs is not something the server can find out.
        "/docs",
        "/openapi.json",
        "/version",
    ],
)
def test_reachable_whichever_way_the_proxy_forwards(mounted, requested):
    assert mounted.get(requested).status_code == 200


def test_docs_page_asks_for_everything_under_the_prefix(mounted):
    """This is the part of #5270 that needed a server-side fix.

    The UI is hash-routed, so it needs no configuration to live under a subpath.
    The docs page is the opposite: it is served HTML that names absolute paths —
    ``/openapi.json`` and the swagger-ui bundle — and at the origin root those
    are the customer's own application, not us. That is why the reporter had to
    add ``location`` blocks for them by hand.
    """
    refs = _absolute_refs(mounted.get("/docs").text)

    assert refs, "expected the docs page to reference assets by absolute path"
    assert any(ref.endswith("/openapi.json") for ref in refs)
    assert all(ref.startswith(f"{PREFIX}/") for ref in refs), refs


def test_static_assets_are_served_under_the_prefix(mounted):
    """The mounts are the reason ``root_path`` alone is not enough.

    ``/static`` here is one of three (``/css`` and ``/js`` carry the UI bundles),
    and a ``Mount`` under a ``root_path`` looks for a file named after its own URL
    prefix unless the path arrives in the shape ASGI describes. Plain routes
    tolerate the mismatch, so the API above would answer while every asset 404s.
    """
    for requested in (f"{PREFIX}/static/swagger-ui.css", "/static/swagger-ui.css"):
        assert mounted.get(requested).status_code == 200, requested


@pytest.mark.parametrize("external_url", [None, "https://example.com"])
def test_root_deployment_keeps_the_prefix_meaningless(tmp_path, external_url):
    # The overwhelmingly common deployment. Nothing above may leak into it: a
    # prefixed path is not a route here, and the docs page must not invent one.
    client = _client(tmp_path, external_url)

    assert client.get("/docs").status_code == 200
    assert client.get(f"{PREFIX}/docs").status_code == 404
    assert all(
        not ref.startswith(f"{PREFIX}/")
        for ref in _absolute_refs(client.get("/docs").text)
    )
