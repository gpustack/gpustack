"""The manifest endpoint's CPU-worker switch, driven through the handler.

What the chart tests and the values tests cannot see is this layer's own
mistake: a query parameter that reaches the render as its own opposite, or a
refusal that leaves the caller holding a 500 instead of the message naming what
to change. Both are one line in ``get_cluster_manifests`` and neither shows up
in a render.

The handler is called directly with the collaborators it loads by id mocked
out — no database and no HTTP client. ``render_bootstrap`` is left real, since
the values it produces are the thing being asserted; that needs the packaged
chart, hence the skip.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.api.tenant import TenantContext
from gpustack.k8s.chart import chart_available
from gpustack.routes import clusters as clusters_route
from gpustack.schemas.clusters import ClusterProvider, ClusterRegistrationTokenPublic
from gpustack.schemas.principals import PrincipalType
from gpustack_runtime.detector import ManufacturerEnum

pytestmark = pytest.mark.skipif(
    not chart_available(),
    reason="the packaged chart is unavailable; run `make deps`",
)

OWNER_PRINCIPAL = 7


def _admin_ctx() -> TenantContext:
    user = MagicMock()
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )


@pytest.fixture
def cluster(monkeypatch):
    """A Kubernetes cluster the handler can render a manifest for.

    ``get_registration_from_cluster`` is stubbed rather than fed a worker
    config: resolving the image goes through the server-wide config, which is
    not what these assert, and an untagged image would be refused for a reason
    that has nothing to do with the CPU worker.
    """
    record = SimpleNamespace(
        id=1,
        name="k8s",
        deleted_at=None,
        owner_principal_id=OWNER_PRINCIPAL,
        provider=ClusterProvider.Kubernetes,
        k8s_options=None,
        worker_config=None,
        system_default_container_registry=None,
    )
    principal = SimpleNamespace(id=OWNER_PRINCIPAL, kind=PrincipalType.USER, name="u")

    monkeypatch.setattr(
        clusters_route.Cluster, "one_by_id", AsyncMock(return_value=record)
    )
    monkeypatch.setattr(
        clusters_route.Principal, "one_by_id", AsyncMock(return_value=principal)
    )
    monkeypatch.setattr(
        clusters_route,
        "get_registration_from_cluster",
        lambda request, cluster: ClusterRegistrationTokenPublic(
            token="tok",
            server_url="http://gpustack.example.com:30080",
            image="docker.io/gpustack/gpustack:dev",
            env={},
            args=[],
        ),
    )
    monkeypatch.setattr(
        clusters_route,
        "get_global_config",
        lambda: SimpleNamespace(
            namespace="gpustack-system",
            operator_image=None,
            system_default_container_registry=None,
        ),
    )
    return record


async def _manifest(runtime=None, disable_cpu_worker=False) -> str:
    # Both passed explicitly: calling the handler as a function skips FastAPI's
    # parameter resolution, so an omitted argument arrives as the `Query(...)`
    # marker itself rather than as its default — and a truthy one at that.
    response = await clusters_route.get_cluster_manifests(
        request=MagicMock(),
        session=MagicMock(),
        ctx=_admin_ctx(),
        id=1,
        runtime=runtime,
        disable_cpu_worker=disable_cpu_worker,
    )
    return response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_the_switch_reaches_the_values_it_renders(cluster):
    # The whole point of the parameter, and the one thing a sign flip here
    # would not change anywhere else: the chart values the cluster is handed.
    manifest = await _manifest(
        runtime=[ManufacturerEnum.NVIDIA], disable_cpu_worker=True
    )
    assert "cpuEnabled: false" in manifest


@pytest.mark.asyncio
async def test_the_cpu_worker_is_on_when_nothing_asks_otherwise(cluster):
    manifest = await _manifest(runtime=[ManufacturerEnum.NVIDIA])
    assert "cpuEnabled: true" in manifest


@pytest.mark.asyncio
async def test_disabling_it_without_a_runtime_is_the_callers_error(cluster):
    # A manifest that deploys nothing is refused where the request can still be
    # corrected. Raised as a ValueError deeper down, so what this pins is the
    # translation: an InvalidException (4xx) rather than a 500 or a manifest.
    with pytest.raises(InvalidException) as refused:
        await _manifest(disable_cpu_worker=True)
    assert "no worker at all" in str(refused.value.message)


@pytest.mark.asyncio
async def test_the_no_gpu_sentinel_does_not_count_as_a_runtime(cluster):
    # `unknown` is what a cluster with no detected GPU carries. It renders no
    # DaemonSet, so accepting it here would hand over the empty release the
    # check above exists to refuse.
    with pytest.raises(InvalidException):
        await _manifest(runtime=[ManufacturerEnum.UNKNOWN], disable_cpu_worker=True)
