import gzip

import pytest
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.testclient import TestClient

from gpustack.routes.ui import (
    CACHE_FOREVER,
    CACHE_REVALIDATE,
    PrecompressedStaticFiles,
    accepts_gzip,
    cache_control_for,
)

# Large enough that the real build would have emitted a .gz for it.
BUNDLE_JS = b"console.log('bundle');\n" * 500
SMALL_JS = b"console.log('small');\n"


@pytest.fixture
def client(tmp_path):
    """A static mount covering every combination the real build produces.

    ``bundle.js`` has a .gz and a stable name, ``small.js`` has neither, and
    ``hashed.530e136d.js`` has both a .gz and a cache-busted name.
    """
    (tmp_path / "bundle.js").write_bytes(BUNDLE_JS)
    (tmp_path / "bundle.js.gz").write_bytes(gzip.compress(BUNDLE_JS))
    (tmp_path / "small.js").write_bytes(SMALL_JS)
    (tmp_path / "hashed.530e136d.js").write_bytes(BUNDLE_JS)
    (tmp_path / "hashed.530e136d.js.gz").write_bytes(gzip.compress(BUNDLE_JS))

    app = Starlette()
    app.mount("/js", PrecompressedStaticFiles(directory=tmp_path), name="js")
    return TestClient(app)


def test_serves_precompressed_sibling_when_gzip_accepted(client):
    response = client.get("/js/bundle.js", headers={"accept-encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    # The transport decodes it, so this asserts the round trip: the browser
    # ends up with the original bundle, not the .gz wrapper.
    assert response.content == BUNDLE_JS
    assert int(response.headers["content-length"]) == len(gzip.compress(BUNDLE_JS))


def test_precompressed_response_is_typed_as_the_decoded_asset(client):
    gzipped = client.get("/js/bundle.js", headers={"accept-encoding": "gzip"})
    plain = client.get("/js/bundle.js", headers={"accept-encoding": "identity"})

    # Not application/gzip: the browser has to parse this as JS once decoded,
    # and it must not decode it differently than the uncompressed asset.
    assert gzipped.headers["content-type"] == "text/javascript; charset=utf-8"
    assert gzipped.headers["content-type"] == plain.headers["content-type"]


def test_serves_plain_asset_when_gzip_not_accepted(client):
    response = client.get("/js/bundle.js", headers={"accept-encoding": "identity"})

    assert response.status_code == 200
    assert "content-encoding" not in response.headers
    assert response.content == BUNDLE_JS
    assert int(response.headers["content-length"]) == len(BUNDLE_JS)


def test_falls_back_to_plain_asset_when_no_gz_sibling(client):
    response = client.get("/js/small.js", headers={"accept-encoding": "gzip"})

    assert response.status_code == 200
    assert "content-encoding" not in response.headers
    assert response.content == SMALL_JS


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_vary_is_set_on_both_representations(client, accept_encoding):
    # Without Vary a shared cache could replay a gzip body to a client that
    # never asked for one.
    response = client.get("/js/bundle.js", headers={"accept-encoding": accept_encoding})

    assert "accept-encoding" in response.headers["vary"].lower()


def test_conditional_request_against_the_gz_representation(client):
    first = client.get("/js/bundle.js", headers={"accept-encoding": "gzip"})
    etag = first.headers["etag"]

    second = client.get(
        "/js/bundle.js",
        headers={"accept-encoding": "gzip", "if-none-match": etag},
    )

    assert second.status_code == 304
    # A 304 carries no body, so it must not claim an encoding for one.
    assert "content-encoding" not in second.headers
    assert "accept-encoding" in second.headers["vary"].lower()


def test_gz_etag_differs_from_plain_etag(client):
    gzipped = client.get("/js/bundle.js", headers={"accept-encoding": "gzip"})
    plain = client.get("/js/bundle.js", headers={"accept-encoding": "identity"})

    # Distinct representations of the same URL need distinct validators,
    # otherwise a conditional request can be answered with the wrong encoding.
    assert gzipped.headers["etag"] != plain.headers["etag"]


def test_head_request_reports_the_compressed_length(client):
    response = client.head("/js/bundle.js", headers={"accept-encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert int(response.headers["content-length"]) == len(gzip.compress(BUNDLE_JS))


def test_directory_traversal_still_blocked(client):
    response = client.get("/js/../../etc/passwd", headers={"accept-encoding": "gzip"})

    assert response.status_code == 404


def test_cache_busted_asset_is_cacheable_forever(client):
    response = client.get("/js/hashed.530e136d.js", headers={"accept-encoding": "gzip"})

    assert response.headers["cache-control"] == CACHE_FOREVER
    assert "immutable" in response.headers["cache-control"]


def test_stable_name_asset_must_be_revalidated(client):
    # No cache buster in the name, so a new build can replace this body in
    # place — the client has to ask before reusing what it stored.
    response = client.get("/js/bundle.js", headers={"accept-encoding": "gzip"})

    assert response.headers["cache-control"] == CACHE_REVALIDATE


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_both_encodings_expire_together(client, accept_encoding):
    # The gzip and plain bodies are one resource; different lifetimes would let
    # a cache keep one long after the other went stale.
    response = client.get(
        "/js/hashed.530e136d.js", headers={"accept-encoding": accept_encoding}
    )

    assert response.headers["cache-control"] == CACHE_FOREVER


def test_not_modified_response_still_carries_the_policy(client):
    first = client.get("/js/hashed.530e136d.js", headers={"accept-encoding": "gzip"})
    second = client.get(
        "/js/hashed.530e136d.js",
        headers={"accept-encoding": "gzip", "if-none-match": first.headers["etag"]},
    )

    # A 304 is how an expired entry gets refreshed; without the policy here the
    # refreshed copy would have no lifetime and revalidate again immediately.
    assert second.status_code == 304
    assert second.headers["cache-control"] == CACHE_FOREVER


@pytest.mark.parametrize(
    "path,expected",
    [
        # Cache-busted names from the real build.
        ("umi.530e136d.js", CACHE_FOREVER),
        ("p__playground__index.1752132356288.chunk.css", CACHE_FOREVER),
        ("editor.a85ce25e.worker.js", CACHE_FOREVER),
        ("bagel.16fb8279.png", CACHE_FOREVER),
        ("KaTeX_AMS-Regular.1608a09b.woff", CACHE_FOREVER),
        # Stable names: backend-shipped icons and the docs bundles, both of
        # which get replaced in place.
        ("catalog_icons/qwen.png", CACHE_REVALIDATE),
        ("swagger-ui-bundle.js", CACHE_REVALIDATE),
        ("favicon.png", CACHE_REVALIDATE),
        ("index.html", CACHE_REVALIDATE),
        # A directory component that looks hashed must not promote the file.
        ("d41d8cd9.assets/qwen.png", CACHE_REVALIDATE),
        # Too short to be a build hash.
        ("logo.abc.png", CACHE_REVALIDATE),
    ],
)
def test_cache_control_for(path, expected):
    assert cache_control_for(path) == expected


def test_static_mount_is_not_shadowed_by_the_docs_mount(config):
    """``/static`` must resolve to our mount, not fastapi_cdn_host's.

    ``patch_docs`` auto-mounts a plain ``StaticFiles`` at ``/static`` unless
    that path is already taken, and the first matching mount wins every
    request under it. So if ``ui.register`` ever moves back after
    ``patch_docs``, the swagger-ui and redoc bundles — the largest files in
    that directory, ~2.6 MB uncompressed — silently stop being served from
    their precompressed ``.gz``, with nothing failing to show it.
    """
    from starlette.routing import Mount

    from gpustack.server.app import create_app

    app = create_app(config)
    static_mounts = [
        route
        for route in app.routes
        if isinstance(route, Mount) and route.path == "/static"
    ]

    assert len(static_mounts) == 1
    assert isinstance(static_mounts[0].app, PrecompressedStaticFiles)


@pytest.mark.parametrize(
    "header,expected",
    [
        ("gzip", True),
        ("gzip, deflate, br", True),
        ("deflate, gzip;q=0.8", True),
        ("*", True),
        ("", False),
        ("identity", False),
        ("deflate, br", False),
        # An explicit refusal, which some proxies send to opt out.
        ("gzip;q=0", False),
        ("identity;q=1, gzip;q=0", False),
        # Malformed q-values are treated as a refusal rather than crashing.
        ("gzip;q=bogus", False),
    ],
)
def test_accepts_gzip(header, expected):
    assert accepts_gzip(Headers({"accept-encoding": header})) is expected
