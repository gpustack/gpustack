from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import AlreadyExistsException
from gpustack.api.tenant import TenantContext
from gpustack.config.config import Config
from gpustack.routes.clusters import get_server_url, update_cluster
from gpustack.schemas.clusters import Cluster, ClusterUpdate
from gpustack.schemas.principals import PrincipalType


def _request(url):
    return SimpleNamespace(url=url)


def _url(scheme="http", netloc="127.0.0.1"):
    return SimpleNamespace(scheme=scheme, netloc=netloc)


def test_get_server_url_strips_trailing_slash_from_cluster_override():
    server_url = get_server_url(
        _request(_url()),
        "http://example.com:30080/",
    )

    assert server_url == "http://example.com:30080"
    assert f"{server_url}/v2/clusters/1/manifests" == (
        "http://example.com:30080/v2/clusters/1/manifests"
    )


def test_get_server_url_strips_trailing_slash_from_external_url():
    config = SimpleNamespace(server_external_url="http://example.com:30080/")

    with patch("gpustack.routes.clusters.get_global_config", return_value=config):
        server_url = get_server_url(_request(_url()), None)

    assert server_url == "http://example.com:30080"
    assert f"{server_url}/v2/clusters/1/manifests" == (
        "http://example.com:30080/v2/clusters/1/manifests"
    )


def test_get_server_url_falls_back_to_request_host():
    # When server_external_url is unset, use the request host (scheme + netloc)
    # as-is. Injection protection is opt-in: ForwardedHostPortMiddleware gates
    # X-Forwarded-Host by trusted_hosts before the handler sees request.url, so
    # anything reaching here is either trusted or the unconfigured pass-through.
    config = SimpleNamespace(server_external_url=None)
    with patch(
        "gpustack.routes.clusters.get_global_config",
        return_value=config,
    ):
        server_url = get_server_url(
            _request(_url(scheme="https", netloc="proxy.example:8443")),
            None,
        )

    assert server_url == "https://proxy.example:8443"


def test_cluster_update_normalizes_server_url():
    cluster = ClusterUpdate(
        name="k8s",
        server_url="http://example.com:30080/",
    )

    assert cluster.server_url == "http://example.com:30080"


def test_get_trusted_hosts_resolution(tmp_path):
    data_dir = str(tmp_path / "data")

    # Explicit trusted_hosts wins verbatim (including the "*" opt-out).
    explicit = Config(data_dir=data_dir, trusted_hosts=["a.example", "*"])
    assert explicit.get_trusted_hosts() == ["a.example", "*"]

    # Derived from server_external_url when trusted_hosts is unset.
    derived = Config(
        data_dir=data_dir, server_external_url="https://gpustack.example:30080"
    )
    assert derived.get_trusted_hosts() == ["gpustack.example"]

    # An explicit empty list behaves like unset: derive from server_external_url.
    empty = Config(
        data_dir=data_dir,
        trusted_hosts=[],
        server_external_url="https://gpustack.example:30080",
    )
    assert empty.get_trusted_hosts() == ["gpustack.example"]

    # Blank/whitespace-only entries drop out and fall through like the empty list.
    blank = Config(
        data_dir=data_dir,
        trusted_hosts=["", "   "],
        server_external_url="https://gpustack.example:30080",
    )
    assert blank.get_trusted_hosts() == ["gpustack.example"]

    # Surrounding whitespace on a real host is trimmed.
    padded = Config(data_dir=data_dir, trusted_hosts=[" a.example "])
    assert padded.get_trusted_hosts() == ["a.example"]

    # Neither set -> ["*"] (trust any host; protection is opt-in).
    assert Config(data_dir=data_dir).get_trusted_hosts() == ["*"]


def test_config_normalizes_server_external_url(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = Config(
        data_dir=str(tmp_path / "data"),
        server_external_url="http://example.com:30080/",
    )

    assert config.server_external_url == "http://example.com:30080"


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(Cluster.__table__.create)
    yield e
    await e.dispose()


def _admin_ctx():
    user = MagicMock()
    user.id = 99
    user.is_admin = True
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
        accessible_cluster_ids=set(),
    )


@pytest.mark.asyncio
async def test_rename_keeps_cluster_names_unique_within_the_org(engine):
    """Create already refuses a name the Org holds; without the same rule on
    rename, resolving a cluster by name is left with two rows and no way to
    choose."""
    ctx = _admin_ctx()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add_all(
            [
                Cluster(id=1, name="alpha", owner_principal_id=1),
                Cluster(id=2, name="beta", owner_principal_id=1),
                # Names are unique within an Org, not globally.
                Cluster(id=3, name="alpha", owner_principal_id=2),
            ]
        )
        await session.commit()

        with pytest.raises(AlreadyExistsException):
            await update_cluster(session, ctx, 2, ClusterUpdate(name="alpha"))

        config = SimpleNamespace(
            server_external_url="http://example.com:30080",
            system_default_container_registry=None,
        )
        with patch("gpustack.routes.clusters.get_global_config", return_value=config):
            # Another Org holding the name is no obstacle, and a cluster does
            # not collide with itself.
            crossing = await update_cluster(session, ctx, 3, ClusterUpdate(name="beta"))
            assert crossing.name == "beta"
            kept = await update_cluster(session, ctx, 1, ClusterUpdate(name="alpha"))
            assert kept.name == "alpha"
