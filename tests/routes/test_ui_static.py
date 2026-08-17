import errno
import gzip
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

import gpustack
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

# `hack/install.sh` downloads this, and skips it when UI_DOWNLOAD=false, so a
# checkout can legitimately lack it. Only the test that builds the real app
# needs it.
UI_DIR = Path(gpustack.__file__).parent / "ui"


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
    # An extension mimetypes has no entry for, to exercise the type fallback.
    (tmp_path / "sourcemap.js.map").write_bytes(BUNDLE_JS)
    (tmp_path / "sourcemap.js.map.gz").write_bytes(gzip.compress(BUNDLE_JS))
    # A .gz with no plain sibling, which the build never produces.
    (tmp_path / "orphan.js.gz").write_bytes(gzip.compress(BUNDLE_JS))

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


@pytest.mark.parametrize("name", ["bundle.js", "sourcemap.js.map"])
def test_precompressed_response_is_typed_as_the_decoded_asset(client, name):
    gzipped = client.get(f"/js/{name}", headers={"accept-encoding": "gzip"})
    plain = client.get(f"/js/{name}", headers={"accept-encoding": "identity"})

    # The exact type is left to mimetypes, which reads the OS registry and so
    # differs between a developer's machine and CI. What must hold everywhere
    # is that the two representations agree, and that neither describes the
    # .gz wrapper instead of the asset inside it. `.map` is the case that
    # actually exercises the fallback: mimetypes has no entry for it.
    assert gzipped.headers["content-type"] == plain.headers["content-type"]
    assert "gzip" not in gzipped.headers["content-type"]


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


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("gone"),
        PermissionError("denied"),
        OSError(errno.ENAMETOOLONG, "name too long"),
        # What os.stat raises for a path holding a null byte. Not an OSError.
        ValueError("embedded null byte"),
        HTTPException(status_code=404),
    ],
)
def test_falls_back_to_plain_asset_when_the_gz_probe_raises(
    client, tmp_path, monkeypatch, failure
):
    """A failed probe for the .gz must never cost us the plain asset.

    The installed Starlette reports a miss by returning ``("", None)``, but the
    dependency range spans a major version of it, so this pins the contract
    from the caller's side instead of trusting one implementation: however
    ``lookup_path`` chooses to fail on the .gz, the request still succeeds.
    """
    mount = PrecompressedStaticFiles(directory=tmp_path)
    real_lookup = mount.lookup_path

    def lookup(path: str):
        if path.endswith(".gz"):
            raise failure
        return real_lookup(path)

    monkeypatch.setattr(mount, "lookup_path", lookup)
    app = Starlette()
    app.mount("/js", mount, name="js")

    response = TestClient(app).get("/js/bundle.js", headers={"accept-encoding": "gzip"})

    assert response.status_code == 200
    assert "content-encoding" not in response.headers
    assert response.content == BUNDLE_JS


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_orphan_gz_does_not_make_the_url_exist(client, accept_encoding):
    """A .gz with no plain sibling is not an encoding of anything.

    Serving it would make the URL resolve for clients that accept gzip and 404
    for everyone else, so whether the resource exists would turn on
    Accept-Encoding — which selects among representations, not existence. It
    would also type the body from a file name nothing on disk has, and a
    `page.html.gz` alone would become HTML the browser renders on this origin.
    """
    response = client.get("/js/orphan.js", headers={"accept-encoding": accept_encoding})

    assert response.status_code == 404


def test_the_gz_itself_is_still_reachable_by_its_own_name(client):
    # Refusing the orphan above must not stop a .gz being served as a plain
    # file when that is literally what was asked for.
    response = client.get("/js/orphan.js.gz", headers={"accept-encoding": "identity"})

    assert response.status_code == 200
    assert response.content == gzip.compress(BUNDLE_JS)


@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_vary_is_set_on_both_representations(client, accept_encoding):
    # Without Vary a shared cache could replay a gzip body to a client that
    # never asked for one.
    response = client.get("/js/bundle.js", headers={"accept-encoding": accept_encoding})

    assert "accept-encoding" in response.headers["vary"].lower()


@pytest.mark.parametrize("origin", [None, "https://elsewhere.example"])
def test_vary_origin_is_set_whether_or_not_the_request_carries_one(client, origin):
    """CORSMiddleware varies the response by Origin without saying so.

    With ``allow_origins=["*"]`` it attaches Access-Control-Allow-Origin only
    to requests that carry an Origin, and adds no ``Vary`` for it. Paired with
    a year-long immutable lifetime, an entry stored from a plain same-origin
    load would be replayed to a cross-origin one with the header missing and
    stay blocked until the next release changes the hash in the URL.
    """
    headers = {"accept-encoding": "gzip"}
    if origin is not None:
        headers["origin"] = origin

    response = client.get("/js/hashed.530e136d.js", headers=headers)

    assert "origin" in response.headers["vary"].lower()


def test_not_modified_response_carries_both_vary_keys(client):
    first = client.get("/js/hashed.530e136d.js", headers={"accept-encoding": "gzip"})
    second = client.get(
        "/js/hashed.530e136d.js",
        headers={"accept-encoding": "gzip", "if-none-match": first.headers["etag"]},
    )

    # The 304 is what refreshes the stored entry, so it has to restate the
    # terms under which that entry may be reused.
    assert second.status_code == 304
    vary = second.headers["vary"].lower()
    assert "accept-encoding" in vary
    assert "origin" in vary


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
    # Percent-encoded, because httpx resolves dot segments client-side: sent
    # literally, the request never reaches the mount and the 404 proves
    # nothing. Encoded, it arrives intact and lookup_path is asked for both
    # "../../etc/passwd.gz" and "../../etc/passwd".
    response = client.get(
        "/js/%2e%2e%2f%2e%2e%2fetc%2fpasswd", headers={"accept-encoding": "gzip"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/js/bundle.js%00", "/js/foo%00bar.js"])
@pytest.mark.parametrize("accept_encoding", ["gzip", "identity"])
def test_null_byte_in_path_is_a_404_not_a_500(client, path, accept_encoding):
    """A null byte must not depend on Accept-Encoding to be rejected safely.

    `os.stat` raises ValueError for these, not an OSError, so probing the .gz
    without catching it let an unauthenticated request reach the error handler
    — a 500 and a logged traceback for gzip-accepting clients, where everyone
    else got Starlette's 404. Parametrised over both encodings because the
    asymmetry is the bug.
    """
    response = client.get(path, headers={"accept-encoding": accept_encoding})

    assert response.status_code == 404


def test_missing_asset_404_carries_no_cache_policy(client):
    # StaticFiles raises for a miss rather than returning a response, so the
    # header work never runs — a cache-busted-looking path that does not exist
    # must not come back with a year-long lifetime attached to its 404.
    response = client.get(
        "/js/nonexistent.deadbeef.js", headers={"accept-encoding": "gzip"}
    )

    assert response.status_code == 404
    assert "cache-control" not in response.headers


def test_range_request_is_consistent_with_the_encoding_it_returns(client):
    response = client.get(
        "/js/bundle.js",
        headers={"accept-encoding": "gzip", "range": "bytes=0-49"},
    )

    # A range applies to the representation actually selected, so serving the
    # .gz is correct — as long as every header agrees it is the gzip one, and
    # the total is the compressed length rather than the original's. Silently
    # handing back compressed bytes described as plain JS is the failure mode
    # worth pinning.
    compressed = gzip.compress(BUNDLE_JS)
    assert response.status_code == 206
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["content-range"] == f"bytes 0-49/{len(compressed)}"
    # The transport decodes as it reads, so what lands here is however much of
    # the asset those 50 compressed bytes unpack to — a prefix of the original,
    # which is only true if the range really was taken from this asset's gzip
    # stream and labelled as such.
    assert response.content
    assert BUNDLE_JS.startswith(response.content)


def test_resuming_a_plain_download_does_not_switch_to_gzip(client):
    plain_etag = client.get(
        "/js/bundle.js", headers={"accept-encoding": "identity"}
    ).headers["etag"]

    response = client.get(
        "/js/bundle.js",
        headers={
            "accept-encoding": "gzip",
            "range": "bytes=100-199",
            "if-range": plain_etag,
        },
    )

    # The validator names the plain representation but gzip is on offer, so
    # the range must be refused outright. Splicing these 100 bytes into a
    # half-downloaded plain file would corrupt it silently.
    assert response.status_code == 200


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


@pytest.mark.skipif(
    not UI_DIR.is_dir(),
    reason="UI assets not downloaded; create_app requires gpustack/ui",
)
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
        # Optional whitespace around the delimiters is legal (RFC 9110 §5.6.6)
        # and must not turn acceptance into a silent fallback to plain.
        ("gzip ; q=0.5", True),
        ("gzip , deflate", True),
        (" GZIP ", True),
        # A named coding beats the wildcard, in either order (§12.5.3), so an
        # explicit refusal is not undone by a trailing "*".
        ("gzip;q=0, *", False),
        ("*, gzip;q=0", False),
        ("*, gzip;q=1", True),
        ("*;q=0", False),
        ("*;q=0, gzip", True),
    ],
)
def test_accepts_gzip(header, expected):
    assert accepts_gzip(Headers({"accept-encoding": header})) is expected
