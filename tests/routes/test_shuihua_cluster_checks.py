"""Offline tests for the Shuihua cluster create/update rules.

No database and no network: the checks operate on the request models plus the
cluster row the caller is editing, with the server config stubbed.
"""

from unittest.mock import MagicMock

import pytest

import gpustack.routes.clusters as clusters
from gpustack.api.exceptions import InvalidException
from gpustack.routes.clusters import (
    apply_shuihua_defaults,
    check_shuihua_requirements,
    create_update_check,
    image_ref_registry,
    is_docker_hub_registry,
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

PRIVATE_REGISTRY = "swr.cn-south-1.myhuaweicloud.com"


@pytest.fixture
def server_config(monkeypatch):
    """Server-level settings the checks read, with nothing configured."""
    cfg = MagicMock()
    cfg.server_external_url = "https://gpustack.test"
    cfg.system_default_container_registry = None
    cfg.shuihua_api_base_url = "https://api.test"
    monkeypatch.setattr(clusters, "get_global_config", lambda: cfg)
    return cfg


# --------------------------------------------------------------------------
# image reference parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_ref, expected",
    [
        # No slash at all: a bare name, so the registry is implicit. The tag
        # colon must not be mistaken for a port.
        ("gpustack:v1", None),
        ("gpustack", None),
        # A single path segment before the name is a Docker Hub namespace.
        ("gpustack/gpustack:v1", None),
        # A dot, a port or localhost in the first segment makes it a host.
        ("reg.cn/gpustack:v1", "reg.cn"),
        (
            "swr.cn-south-1.myhuaweicloud.com/gpustack/gpustack:v1",
            "swr.cn-south-1.myhuaweicloud.com",
        ),
        ("localhost:5000/gpustack:v1", "localhost:5000"),
        ("localhost/gpustack:v1", "localhost"),
    ],
)
def test_image_ref_registry(image_ref, expected):
    assert image_ref_registry(image_ref) == expected


@pytest.mark.parametrize(
    "registry, expected",
    [
        ("docker.io", True),
        ("index.docker.io", True),
        ("registry-1.docker.io", True),
        ("registry.hub.docker.com", True),
        ("DOCKER.IO", True),
        ("docker.io/gpustack", True),
        ("reg.cn", False),
        ("mydocker.io.cn", False),
    ],
)
def test_is_docker_hub_registry(registry, expected):
    assert is_docker_hub_registry(registry) is expected


# --------------------------------------------------------------------------
# proxy mode defaulting
# --------------------------------------------------------------------------


def test_create_defaults_proxy_mode_to_tunnel():
    """Instances sit behind NAT with no mapping for the worker's own port."""
    request = ClusterCreate(name="c", provider=ClusterProvider.Shuihua, credential_id=1)

    apply_shuihua_defaults(request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.TUNNEL


def test_create_fills_only_an_empty_proxy_mode():
    """An operator whose network routes to the instances may pick another mode."""
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        worker_config={"proxy_mode": "direct"},
    )

    apply_shuihua_defaults(request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT


def test_update_without_worker_config_is_left_alone():
    """Cluster.update applies model_fields_set only.

    Assigning a worker_config here would add it to that set and overwrite the
    cluster's stored config with a bare default.
    """
    request = ClusterUpdate(name="renamed")

    apply_shuihua_defaults(request, existing=Cluster(name="c"))

    assert request.worker_config is None
    assert "worker_config" not in request.model_fields_set


def test_update_keeps_the_stored_proxy_mode():
    """Sending worker_config replaces it wholesale.

    Defaulting blindly would silently move a cluster the operator had put on
    WORKER over to the tunnel.
    """
    stored = Cluster(
        name="c",
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.WORKER
        ),
    )
    request = ClusterUpdate(name="c", worker_config={"debug": True})

    apply_shuihua_defaults(request, existing=stored)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.WORKER


def test_update_still_honours_an_explicit_proxy_mode():
    stored = Cluster(
        name="c",
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.WORKER
        ),
    )
    request = ClusterUpdate(name="c", worker_config={"proxy_mode": "direct"})

    apply_shuihua_defaults(request, existing=stored)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.DIRECT


# --------------------------------------------------------------------------
# registry requirement
# --------------------------------------------------------------------------


def test_create_requires_a_registry(server_config):
    """The instances cannot reach Docker Hub, and the server has no default."""
    request = ClusterCreate(name="c", provider=ClusterProvider.Shuihua, credential_id=1)

    with pytest.raises(InvalidException):
        check_shuihua_requirements(request)


def test_create_accepts_a_registry_on_the_cluster(server_config):
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        system_default_container_registry=PRIVATE_REGISTRY,
    )

    check_shuihua_requirements(request)


def test_create_accepts_the_server_level_registry(server_config):
    """get_cluster_image_name falls back to it, so rejecting would be a false alarm."""
    server_config.system_default_container_registry = PRIVATE_REGISTRY
    request = ClusterCreate(name="c", provider=ClusterProvider.Shuihua, credential_id=1)

    check_shuihua_requirements(request)


@pytest.mark.parametrize("registry", ["docker.io", "index.docker.io"])
def test_create_rejects_docker_hub(server_config, registry):
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        system_default_container_registry=registry,
    )

    with pytest.raises(InvalidException):
        check_shuihua_requirements(request)


@pytest.mark.parametrize(
    "override, rejected",
    [
        # A full override decides where the pull comes from, so it is what gets
        # checked -- and one with no host resolves to Docker Hub.
        ("gpustack/gpustack:v1", True),
        ("gpustack:v1", True),
        ("docker.io/gpustack/gpustack:v1", True),
        (f"{PRIVATE_REGISTRY}/gpustack/gpustack:v1", False),
    ],
)
def test_create_checks_an_image_override_over_the_registry(
    server_config, override, rejected
):
    server_config.system_default_container_registry = PRIVATE_REGISTRY
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        worker_config={"image_name_override": override},
    )

    if rejected:
        with pytest.raises(InvalidException):
            check_shuihua_requirements(request)
    else:
        check_shuihua_requirements(request)


def test_update_accepts_the_registry_already_on_the_cluster(server_config):
    """A partial PUT need not resend it, and Cluster.update keeps what it
    doesn't touch, so ignoring the stored value rejected a configured cluster."""
    stored = Cluster(name="c", system_default_container_registry=PRIVATE_REGISTRY)
    request = ClusterUpdate(name="c", worker_config={"debug": True})

    check_shuihua_requirements(request, existing=stored)


def test_update_untouched_by_the_registry_rule(server_config):
    """A rename sends neither worker_config nor the registry, so nothing to check."""
    check_shuihua_requirements(
        ClusterUpdate(name="renamed"), existing=Cluster(name="c")
    )


def test_update_rejects_switching_to_docker_hub(server_config):
    stored = Cluster(name="c", system_default_container_registry=PRIVATE_REGISTRY)
    request = ClusterUpdate(name="c", system_default_container_registry="docker.io")

    with pytest.raises(InvalidException):
        check_shuihua_requirements(request, existing=stored)


# --------------------------------------------------------------------------
# the whole check, as the routes call it
# --------------------------------------------------------------------------


def test_create_update_check_requires_a_credential(server_config):
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        system_default_container_registry=PRIVATE_REGISTRY,
    )

    with pytest.raises(InvalidException):
        create_update_check(ClusterProvider.Shuihua, request)


def test_create_update_check_passes_a_complete_request(server_config):
    request = ClusterCreate(
        name="c",
        provider=ClusterProvider.Shuihua,
        credential_id=1,
        system_default_container_registry=PRIVATE_REGISTRY,
    )

    create_update_check(ClusterProvider.Shuihua, request)

    assert request.worker_config.proxy_mode is ModelInstanceProxyModeEnum.TUNNEL


def test_create_update_check_allows_editing_a_configured_cluster(server_config):
    """The regression: a PUT carrying worker_config used to 400 unless the
    caller resent the registry, because the check never saw the cluster row."""
    stored = Cluster(
        name="c",
        provider=ClusterProvider.Shuihua,
        system_default_container_registry=PRIVATE_REGISTRY,
        worker_config=PredefinedConfigNoDefaults(
            proxy_mode=ModelInstanceProxyModeEnum.TUNNEL
        ),
    )
    request = ClusterUpdate(name="c", worker_config={"debug": True})

    create_update_check(ClusterProvider.Shuihua, request, existing=stored)


@pytest.mark.parametrize(
    "provider", [ClusterProvider.Docker, ClusterProvider.DigitalOcean]
)
def test_other_providers_are_untouched(server_config, provider):
    request = ClusterCreate(name="c", provider=provider, credential_id=1)

    create_update_check(provider, request)

    assert request.worker_config is None
