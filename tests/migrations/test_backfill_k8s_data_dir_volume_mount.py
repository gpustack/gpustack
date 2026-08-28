"""Offline tests for the data-dir volume mount backfill migration.

No database: the merge the migration applies per cluster row is a pure function
over the ``k8s_options`` JSON column, so it is checked directly against the
shapes that column is known to hold (MySQL hands back text, PostgreSQL a dict).
"""

import importlib.util
import json
from pathlib import Path

import pytest

from gpustack.schemas.clusters import (
    DATA_DIR_MOUNT_NAME,
    DATA_DIR_MOUNT_PATH,
    DEFAULT_DATA_DIR_HOST_PATH,
    K8sOptions,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "gpustack"
    / "migrations"
    / "versions"
    / "2026_08_28_1000-e4a1c8b7d0f3_backfill_k8s_data_dir_volume_mount.py"
)


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("_backfill_data_dir", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_dir_mount(host_path=DEFAULT_DATA_DIR_HOST_PATH):
    return {
        "name": DATA_DIR_MOUNT_NAME,
        "mountPath": DATA_DIR_MOUNT_PATH,
        "readOnly": False,
        "volumeSource": {"hostPath": {"path": host_path, "type": "DirectoryOrCreate"}},
    }


EXTRA_MOUNT = {
    "name": "model-cache",
    "mountPath": "/mnt/models",
    "volumeSource": {"hostPath": {"path": "/data/models", "type": "Directory"}},
}


@pytest.mark.parametrize(
    "stored",
    [
        None,
        "null",
        {},
        json.dumps({}),
        {"volumeMounts": []},
        {"namespace": "gpustack-system-abc123"},
        json.dumps({"namespace": "gpustack-system-abc123"}),
    ],
    ids=[
        "null-column",
        "json-null-text",
        "empty-dict",
        "empty-dict-text",
        "empty-mount-list",
        "namespace-only",
        "namespace-only-text",
    ],
)
def test_a_cluster_without_the_mount_is_backfilled(migration, stored):
    updated = migration.backfilled(stored)

    assert updated is not None
    assert updated["volumeMounts"] == [_data_dir_mount()]


def test_the_rest_of_k8s_options_is_preserved(migration):
    """Dropping namespace would repoint the cluster's manifests, and dropping
    gpuInstanceOptions would flip its purpose."""
    updated = migration.backfilled(
        {
            "namespace": "gpustack-system-abc123",
            "gpuInstanceOptions": {"gpuInstanceTypeMixedOnNode": True},
        }
    )

    assert updated["namespace"] == "gpustack-system-abc123"
    assert updated["gpuInstanceOptions"] == {"gpuInstanceTypeMixedOnNode": True}


@pytest.mark.parametrize(
    "existing",
    [
        [_data_dir_mount()],
        [_data_dir_mount("/mnt/data/gpustack")],
        [_data_dir_mount(), EXTRA_MOUNT],
        # Written before the reserved name was enforced.
        [
            {
                "name": "data-dir",
                "mountPath": DATA_DIR_MOUNT_PATH,
                "volumeSource": {
                    "hostPath": {"path": "/mnt/data", "type": "Directory"}
                },
            }
        ],
    ],
    ids=["default", "custom-host-path", "with-user-mount", "legacy-name"],
)
def test_a_populated_list_is_left_alone(migration, existing):
    """A non-empty volumeMounts already holds the data dir at index 0: the field
    did not exist before v2.2.0 and every write since then puts it there. The
    host path in such a row may have been set deliberately."""
    assert migration.backfilled({"volumeMounts": existing}) is None


def test_the_snake_case_key_form_is_recognized(migration):
    """``volume_mounts`` reads back fine (populate_by_name), but the column is
    written by alias, so the backfill leaves a single camelCase key."""
    assert migration.backfilled({"volume_mounts": [_data_dir_mount()]}) is None

    updated = migration.backfilled({"volume_mounts": [], "namespace": "ns"})

    assert "volume_mounts" not in updated
    assert updated["volumeMounts"] == [_data_dir_mount()]


# --------------------------------------------------------------------------
# downgrade: only the exact entry the upgrade wrote is dropped
# --------------------------------------------------------------------------


def _snake_case(mount: dict) -> dict:
    """The same mount with every key in the snake_case form."""
    host_path = mount["volumeSource"]["hostPath"]
    return {
        "name": mount["name"],
        "mount_path": mount["mountPath"],
        "read_only": mount["readOnly"],
        "volume_source": {"host_path": dict(host_path)},
    }


@pytest.mark.parametrize(
    "mount, expected",
    [
        (_data_dir_mount(), True),
        # Every key form has to be read on every level of the mount, not just
        # the top one; a partial reading answers "no" for a shape it was meant
        # to recognize and leaves the entry behind on downgrade.
        (_snake_case(_data_dir_mount()), True),
        # A host path the user chose: the pre-upgrade code renders it fine, so
        # it is not ours to remove.
        (_data_dir_mount("/mnt/data/gpustack"), False),
        (_snake_case(_data_dir_mount("/mnt/data/gpustack")), False),
        # Not the reserved name.
        ({**_data_dir_mount(), "name": "data-dir"}, False),
        # No volume source at all.
        ({"name": DATA_DIR_MOUNT_NAME, "mountPath": DATA_DIR_MOUNT_PATH}, False),
        (EXTRA_MOUNT, False),
        ("not-a-mount", False),
    ],
    ids=[
        "camel-default",
        "snake-default",
        "camel-custom-host-path",
        "snake-custom-host-path",
        "legacy-name",
        "no-volume-source",
        "user-mount",
        "not-a-dict",
    ],
)
def test_only_the_entry_the_upgrade_wrote_is_recognized(migration, mount, expected):
    assert migration._is_backfilled_data_dir_mount(mount) is expected


def test_the_backfilled_value_parses_as_k8s_options(migration):
    """The row has to load through the live schema after the migration."""
    updated = migration.backfilled({"namespace": "gpustack-system-abc123"})

    k8s_options = K8sOptions.model_validate(updated)

    assert k8s_options.namespace == "gpustack-system-abc123"
    mount = k8s_options.volume_mounts[0]
    assert mount.name == DATA_DIR_MOUNT_NAME
    assert mount.mount_path == DATA_DIR_MOUNT_PATH
    assert mount.read_only is False
    assert mount.volume_source.host_path.path == DEFAULT_DATA_DIR_HOST_PATH
    assert mount.volume_source.host_path.type == "DirectoryOrCreate"
