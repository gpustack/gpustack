"""The image a cluster's workers are told to pull.

The server-level ``image_name_override`` — under Helm always present, since
the chart pins the exact image the server runs — is a default for clusters
that pin nothing about their own pull location, not a mandate: a cluster's
explicit registry or repository setting must decide the image instead of
being swallowed by the server's full-image reference.
"""

from types import SimpleNamespace

import pytest

import gpustack.config.config as config_module
from gpustack.config.config import get_cluster_image_name
from gpustack.schemas.config import PredefinedConfigNoDefaults

_VERSION = "v9.9.9"
_OVERRIDE = "registry.example.com/gpustack/gpustack:2.2.0"


@pytest.fixture(autouse=True)
def hermetic_resolution(monkeypatch):
    # Pin the appended version tag, and keep registry resolution offline: the
    # real resolver probes Docker Hub reachability whenever nothing overrides
    # it, which has no place in a unit test.
    monkeypatch.setattr(
        "gpustack.extension.resolve_version_info", lambda: (_VERSION, "commit")
    )
    monkeypatch.setattr(
        config_module, "determine_default_registry", lambda override=None: override
    )


def _server_config(monkeypatch, **cfg):
    # Mirrors the server Config defaults: a chart install sets the override
    # (and now the enterprise repository as image_repo) while leaving the
    # registry at the chart's hub.
    cfg.setdefault("image_name_override", _OVERRIDE)
    cfg.setdefault("image_repo", "gpustack/gpustack")
    cfg.setdefault("system_default_container_registry", "registry.example.com")
    monkeypatch.setattr(
        config_module, "get_global_config", lambda: SimpleNamespace(**cfg)
    )


def test_server_override_is_the_default_for_an_unset_cluster(monkeypatch):
    _server_config(monkeypatch)

    assert get_cluster_image_name(None) == _OVERRIDE
    assert get_cluster_image_name(PredefinedConfigNoDefaults()) == _OVERRIDE


def test_cluster_registry_and_repo_win_over_the_server_override(monkeypatch):
    # The Helm chart always injects the server's image as an override; a
    # cluster that pins its own pull location must not have it applied.
    _server_config(monkeypatch)
    worker_config = PredefinedConfigNoDefaults(image_repo="myorg/gpustack")

    image = get_cluster_image_name(worker_config, "mirror.example.com")

    assert image == f"mirror.example.com/myorg/gpustack:{_VERSION}"


def test_cluster_registry_alone_repoints_the_default(monkeypatch):
    _server_config(monkeypatch)

    image = get_cluster_image_name(None, "mirror.example.com")

    assert image == f"mirror.example.com/gpustack/gpustack:{_VERSION}"


def test_cluster_repo_alone_wins_over_the_server_override(monkeypatch):
    _server_config(monkeypatch)
    worker_config = PredefinedConfigNoDefaults(image_repo="myorg/gpustack")

    image = get_cluster_image_name(worker_config)

    assert image == f"registry.example.com/myorg/gpustack:{_VERSION}"


def test_cluster_level_override_decides_the_image_outright(monkeypatch):
    _server_config(monkeypatch)
    worker_config = PredefinedConfigNoDefaults(
        image_name_override="mirror.example.com/myorg/gpustack:custom"
    )

    image = get_cluster_image_name(worker_config, "other.example.com")

    assert image == "mirror.example.com/myorg/gpustack:custom"


def test_an_explicitly_empty_registry_still_counts_as_pinned(monkeypatch):
    # "" is an explicit setting, not "unset": with a truthiness check the
    # server override would swallow it as if nothing had been pinned.
    _server_config(monkeypatch)

    image = get_cluster_image_name(None, "")

    assert image == f"registry.example.com/gpustack/gpustack:{_VERSION}"


def test_an_explicitly_empty_repo_still_counts_as_pinned(monkeypatch):
    _server_config(monkeypatch)
    worker_config = PredefinedConfigNoDefaults(image_repo="")

    image = get_cluster_image_name(worker_config)

    assert image == f"registry.example.com/gpustack/gpustack:{_VERSION}"


def test_server_repo_and_registry_apply_without_an_override(monkeypatch):
    # A cluster that customizes nothing on a server configured through the
    # partial knobs (no chart override) composes from those parts.
    _server_config(
        monkeypatch,
        image_name_override=None,
        image_repo="myorg/gpustack",
        system_default_container_registry="registry.example.com",
    )

    image = get_cluster_image_name(PredefinedConfigNoDefaults())

    assert image == f"registry.example.com/myorg/gpustack:{_VERSION}"
