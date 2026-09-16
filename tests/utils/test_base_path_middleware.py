import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from starlette.testclient import TestClient

from gpustack.utils.base_path import BasePathMiddleware

PREFIX = "/gpustack"


async def _noop(*args, **kwargs):
    pass


async def _run(prefix, path, scope_type="http", raw_path=None):
    """Return the scope the wrapped app was called with."""
    seen = {}

    async def app(scope, receive, send):
        seen.update(scope)

    scope = {"type": scope_type, "path": path}
    if raw_path is not None:
        scope["raw_path"] = raw_path

    await BasePathMiddleware(app, prefix=prefix)(scope, _noop, _noop)
    return seen


@pytest.mark.asyncio
async def test_stripped_path_gets_the_prefix_back():
    scope = await _run(PREFIX, "/v1/models")

    assert scope["path"] == "/gpustack/v1/models"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/gpustack/v1/models", "/gpustack/", "/gpustack"])
async def test_path_that_already_carries_the_prefix_is_left_alone(path):
    # A proxy that preserves the prefix (AWS ALB cannot rewrite paths) already
    # sends the shape root_path wants. Prefixing again would 404 everything.
    scope = await _run(PREFIX, path)

    assert scope["path"] == path


@pytest.mark.asyncio
async def test_a_sibling_of_the_prefix_is_not_inside_the_mount():
    # ``/gpustack-internal`` shares the prefix's characters but not its path
    # segment, so under a stripping proxy it is a real path below the mount and
    # has to be restored like any other.
    scope = await _run(PREFIX, "/gpustack-internal")

    assert scope["path"] == "/gpustack/gpustack-internal"


@pytest.mark.asyncio
async def test_websocket_scopes_are_normalised_too():
    # Workers reach the server over a websocket, and the /docs-style URL
    # generation is not what matters there: routing is.
    scope = await _run(PREFIX, "/v2/workers/ws", scope_type="websocket")

    assert scope["path"] == "/gpustack/v2/workers/ws"


@pytest.mark.asyncio
async def test_non_request_scopes_pass_through_untouched():
    scope = await _run(PREFIX, "/v1/models", scope_type="lifespan")

    assert scope["path"] == "/v1/models"


@pytest.mark.asyncio
async def test_empty_prefix_is_a_no_op():
    scope = await _run("", "/v1/models")

    assert scope["path"] == "/v1/models"


@pytest.mark.asyncio
async def test_raw_path_is_kept_in_step_with_path():
    scope = await _run(PREFIX, "/v1/mo dels", raw_path=b"/v1/mo%20dels")

    assert scope["path"] == "/gpustack/v1/mo dels"
    # Still undecoded: rewriting it from the decoded ``path`` would quietly
    # normalise the escape away.
    assert scope["raw_path"] == b"/gpustack/v1/mo%20dels"


@pytest.mark.asyncio
async def test_absent_raw_path_is_not_invented():
    scope = await _run(PREFIX, "/v1/models")

    assert "raw_path" not in scope


def _mounted_app(prefix, directory, with_middleware):
    """The shape create_app builds: a root_path plus a StaticFiles mount."""
    app = Starlette(
        routes=[
            Route("/version", lambda request: PlainTextResponse("ok")),
        ],
    )
    app.mount("/js", StaticFiles(directory=directory), name="js")
    app.router.redirect_slashes = False
    if with_middleware:
        app.add_middleware(BasePathMiddleware, prefix=prefix)
    return app


@pytest.fixture
def asset_dir(tmp_path):
    (tmp_path / "bundle.js").write_text("console.log('bundle');\n")
    return tmp_path


@pytest.mark.parametrize(
    "requested",
    [
        # What a prefix-preserving proxy forwards.
        "/gpustack/js/bundle.js",
        # What the common nginx recipe forwards: a `proxy_pass` with a trailing
        # slash strips the prefix.
        "/js/bundle.js",
    ],
)
def test_mounted_assets_survive_either_kind_of_proxy(asset_dir, requested):
    client = TestClient(
        _mounted_app(PREFIX, asset_dir, with_middleware=True),
        root_path=PREFIX,
    )

    assert client.get(requested).status_code == 200


def test_a_mount_under_root_path_is_why_this_middleware_exists(asset_dir):
    """Pins the failure the middleware prevents, so it is not deleted as inert.

    ASGI says ``root_path`` is a prefix *of* ``path``, and ``Mount.matches``
    takes that literally: it hands its child ``root_path + matched_path``, here
    ``/gpustack/js``, which the ``StaticFiles`` inside then tries to strip off a
    path that never carried it. Plain routes survive the same mismatch because
    ``get_route_path`` gives up gracefully — which is exactly what makes this
    dangerous to eyeball. The API answers, so the deployment looks fine, while
    every UI asset 404s.
    """
    client = TestClient(
        _mounted_app(PREFIX, asset_dir, with_middleware=False),
        root_path=PREFIX,
    )

    assert client.get("/version").status_code == 200
    assert client.get("/js/bundle.js").status_code == 404
