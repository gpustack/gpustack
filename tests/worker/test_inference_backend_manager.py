from unittest.mock import MagicMock

from gpustack.schemas.inference_backend import (
    InferenceBackend,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.inference_backend_manager import InferenceBackendManager


def create_backend(
    id=1,
    backend_name="vLLM",
    owner_principal_id=None,
    version_configs=None,
    is_built_in=True,
    health_check_path=None,
):
    backend = InferenceBackend(
        id=id,
        backend_name=backend_name,
        owner_principal_id=owner_principal_id,
        is_built_in=is_built_in,
        health_check_path=health_check_path,
    )
    backend.version_configs = VersionConfigDict(root=version_configs or {})
    return backend


def built_in_version(image_name="built-in:img"):
    return VersionConfig(image_name=image_name, built_in_frameworks=["cuda"])


def custom_version(image_name="custom:img"):
    return VersionConfig(image_name=image_name, custom_framework="cuda")


def create_manager(rows):
    """Build a manager whose initial cache is seeded from the given rows,
    mimicking the uncollapsed per-row payload of /inference-backends/all."""
    clientset = MagicMock()
    resp = MagicMock()
    resp.json.return_value = [row.model_dump(mode="json") for row in rows]
    clientset.http_client.get_httpx_client.return_value.get.return_value = resp
    return InferenceBackendManager(clientset)


def test_get_backend_merges_platform_and_owner_scope():
    platform_row = create_backend(
        id=1,
        version_configs={"0.11.0": built_in_version()},
        health_check_path="/health",
    )
    org_row = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"0.11.0-custom": custom_version()},
    )
    manager = create_manager([platform_row, org_row])

    merged = manager.get_backend_by_name("vLLM", 42)
    assert merged is not None
    assert set(merged.version_configs.root.keys()) == {"0.11.0", "0.11.0-custom"}

    # Without an owner only the Platform row is visible.
    platform_only = manager.get_backend_by_name("vLLM")
    assert set(platform_only.version_configs.root.keys()) == {"0.11.0"}

    # Another Org doesn't see Org 42's custom version.
    other_org = manager.get_backend_by_name("vLLM", 7)
    assert set(other_org.version_configs.root.keys()) == {"0.11.0"}


def test_owner_version_overrides_platform_on_key_collision():
    platform_row = create_backend(
        id=1,
        version_configs={"0.11.0-custom": custom_version("platform:img")},
    )
    org_row = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"0.11.0-custom": custom_version("org:img")},
    )
    manager = create_manager([platform_row, org_row])

    merged = manager.get_backend_by_name("vLLM", 42)
    assert merged.version_configs.root["0.11.0-custom"].image_name == "org:img"
    # The merge must not leak back into the cached Platform row.
    assert (
        manager.get_backend_by_name("vLLM")
        .version_configs.root["0.11.0-custom"]
        .image_name
        == "platform:img"
    )


def test_org_update_event_does_not_clobber_other_scopes():
    platform_row = create_backend(
        id=1,
        version_configs={
            "0.11.0": built_in_version(),
            "0.10.0-custom": custom_version("platform:img"),
        },
    )
    org_a_row = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"a-custom": custom_version("org-a:img")},
    )
    org_b_row = create_backend(
        id=3,
        owner_principal_id=43,
        version_configs={"b-custom": custom_version("org-b:img")},
    )
    manager = create_manager([platform_row, org_a_row, org_b_row])

    # Org A updates its row: renames its custom version.
    updated_org_a = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"a2-custom": custom_version("org-a:img2")},
    )
    manager._handle_event(
        Event(type=EventType.UPDATED, data=updated_org_a.model_dump(mode="json"))
    )

    # Org A sees its renamed version plus the Platform versions.
    org_a_view = manager.get_backend_by_name("vLLM", 42)
    assert set(org_a_view.version_configs.root.keys()) == {
        "0.11.0",
        "0.10.0-custom",
        "a2-custom",
    }

    # Org B's custom version and the Platform's custom version survive.
    org_b_view = manager.get_backend_by_name("vLLM", 43)
    assert set(org_b_view.version_configs.root.keys()) == {
        "0.11.0",
        "0.10.0-custom",
        "b-custom",
    }


def test_update_event_preserves_built_in_versions_within_scope():
    platform_row = create_backend(
        id=1,
        version_configs={
            "0.11.0": built_in_version(),
            "old-custom": custom_version(),
        },
    )
    manager = create_manager([platform_row])

    # An update event carries only the raw DB row (custom versions, no
    # runner enrichment); built-in entries must be preserved and stale
    # custom entries dropped.
    updated = create_backend(
        id=1,
        version_configs={"new-custom": custom_version()},
    )
    manager._handle_event(
        Event(type=EventType.UPDATED, data=updated.model_dump(mode="json"))
    )

    view = manager.get_backend_by_name("vLLM")
    assert set(view.version_configs.root.keys()) == {"0.11.0", "new-custom"}


def test_delete_event_removes_only_that_scope():
    platform_row = create_backend(id=1, version_configs={"0.11.0": built_in_version()})
    org_row = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"a-custom": custom_version()},
    )
    manager = create_manager([platform_row, org_row])

    manager._handle_event(
        Event(type=EventType.DELETED, data=org_row.model_dump(mode="json"))
    )

    # Platform row survives; Org scope is gone.
    view = manager.get_backend_by_name("vLLM", 42)
    assert set(view.version_configs.root.keys()) == {"0.11.0"}

    # Deleting the last scope drops the backend entirely.
    manager._handle_event(
        Event(type=EventType.DELETED, data=platform_row.model_dump(mode="json"))
    )
    assert manager.get_backend_by_name("vLLM") is None


def test_create_event_adds_new_org_scope():
    platform_row = create_backend(id=1, version_configs={"0.11.0": built_in_version()})
    manager = create_manager([platform_row])

    org_row = create_backend(
        id=2,
        owner_principal_id=42,
        version_configs={"a-custom": custom_version()},
    )
    manager._handle_event(
        Event(type=EventType.CREATED, data=org_row.model_dump(mode="json"))
    )

    view = manager.get_backend_by_name("vLLM", 42)
    assert set(view.version_configs.root.keys()) == {"0.11.0", "a-custom"}
