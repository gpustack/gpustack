"""New clusters default their workers to direct proxy mode.

``proxy_mode`` is a worker setting; the cluster only carries a default in
``worker_config`` that is handed to each worker at registration. Direct mode
performs better under load but needs the instance port range reachable, so the
default is scoped to clusters created after this change: existing clusters must
keep the mode they were using across an upgrade.
"""

from unittest.mock import MagicMock

import pytest

import gpustack.routes.clusters as clusters
from gpustack.routes.clusters import (
    apply_cluster_defaults,
    create_update_check,
)
from gpustack.schemas.clusters import (
    Cluster,
    ClusterCreate,
    ClusterProvider,
    ClusterUpdate,
)
from gpustack.schemas.config import (
    ModelInstanceProxyModeEnum,
    PredefinedConfigNoDefaults,
)


@pytest.fixture
def server_config(monkeypatch):
    cfg = MagicMock()
    cfg.server_external_url = "https://gpustack.test"
    cfg.shuihua_api_base_url = "https://api.test"
    monkeypatch.setattr(clusters, "get_global_config", lambda: cfg)
    return cfg


@pytest.mark.parametrize(
    "provider",
    [ClusterProvider.Docker, ClusterProvider.Kubernetes, ClusterProvider.DigitalOcean],
)
def test_create_defaults_new_clusters_to_direct(server_config, provider):
    request = ClusterCreate(name="c", provider=provider, credential_id=1)

    create_update_check(provider, request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT


@pytest.mark.parametrize(
    "mode",
    [ModelInstanceProxyModeEnum.WORKER, ModelInstanceProxyModeEnum.TUNNEL],
)
def test_create_keeps_an_explicit_proxy_mode(mode):
    request = ClusterCreate(
        name="c", provider=ClusterProvider.Docker, worker_config={"proxy_mode": mode}
    )

    apply_cluster_defaults(request)

    assert request.worker_config.proxy_mode is mode


def test_create_fills_only_an_empty_proxy_mode():
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Docker,
        worker_config={"debug": True},
    )

    apply_cluster_defaults(request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT
    assert request.worker_config.debug is True


def test_update_without_worker_config_is_left_alone():
    request = ClusterUpdate(name="renamed")

    apply_cluster_defaults(request, existing=Cluster(name="c"))

    assert request.worker_config is None
    assert "worker_config" not in request.model_fields_set


def test_update_carries_the_stored_proxy_mode_over():
    stored = Cluster(
        name="c",
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.DIRECT
        ),
    )
    request = ClusterUpdate(name="c", worker_config={"debug": True})

    apply_cluster_defaults(request, existing=stored)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT


def test_update_does_not_move_a_cluster_created_before_the_default_changed():
    """A cluster with nothing stored must not pick up direct on an edit."""
    request = ClusterUpdate(name="c", worker_config={"debug": True})

    apply_cluster_defaults(request, existing=Cluster(name="c"))

    assert request.worker_config.proxy_mode is None


def test_update_still_honours_an_explicit_proxy_mode():
    stored = Cluster(
        name="c",
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.WORKER
        ),
    )
    request = ClusterUpdate(name="c", worker_config={"proxy_mode": "direct"})

    apply_cluster_defaults(request, existing=stored)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT


def test_shuihua_keeps_its_tunnel_default(server_config):
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        system_default_container_registry="registry.example.com",
    )

    create_update_check(ClusterProvider.Shuihua, request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.TUNNEL
