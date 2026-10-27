"""backfill the gpustack data dir volume mount on Kubernetes clusters

Before v2.2.0 the worker DaemonSet hardcoded its ``/var/lib/gpustack`` volume.
v2.2.0 made the host path configurable by rendering it from
``clusters.k8s_options.volumeMounts`` instead, but nothing backfilled the entry
for clusters that already existed, so an upgraded cluster is left with no
``volumeMounts`` at all: its worker DaemonSets render with no persistent data
dir (worker data lost on every pod restart), and — until the route layer
started synthesizing the mount — every edit of the cluster was rejected for the
missing entry.

Backfilled with the pre-v2.2.0 host path, ``/var/lib/gpustack``, so the data
already on the nodes is picked up as is rather than the worker starting over in
a fresh directory.

Revision ID: e4a1c8b7d0f3
Revises: b7e2c4d15a80
Create Date: 2026-08-28 10:00:00.000000

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e4a1c8b7d0f3'
down_revision: Union[str, None] = 'b7e2c4d15a80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Frozen copies of the reserved-mount constants in schemas/clusters.py. A
# migration describes one moment in the schema's history, so it must not follow
# the live values if those are ever changed.
DATA_DIR_MOUNT_NAME = 'gpustack-data-dir'
DATA_DIR_MOUNT_PATH = '/var/lib/gpustack'
DATA_DIR_HOST_PATH = '/var/lib/gpustack'

clusters_jsonable = sa.table(
    'clusters',
    sa.column('id', sa.Integer),
    sa.column('k8s_options', sa.JSON),
)


def _as_dict(value) -> dict:
    """Read a JSON column that may come back as text, a dict, or NULL."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value else {}
        except json.JSONDecodeError:
            value = {}
    # Covers None, the JSON literal "null" (which parses to None), and any
    # other non-dict shape the column might somehow be holding.
    if not isinstance(value, dict):
        return {}
    return value


def _either(mapping: dict, camel: str, snake: str):
    """Read a key that may be stored under either form.

    The live column is written by ``pydantic_column_type``, which encodes by
    alias, so camelCase is what is there today; a migration outlives that
    guarantee, and reading both costs nothing. Every key form has to be covered
    on every level of the mount, not just the top one — a partial reading would
    silently answer "no" for a shape it was meant to recognize.
    """
    value = mapping.get(camel)
    return value if value is not None else mapping.get(snake)


def _is_backfilled_data_dir_mount(mount) -> bool:
    """Whether a mount is exactly what :func:`backfilled` writes."""
    if not isinstance(mount, dict):
        return False
    volume_source = _either(mount, 'volumeSource', 'volume_source') or {}
    host_path = _either(volume_source, 'hostPath', 'host_path') or {}
    return (
        mount.get('name') == DATA_DIR_MOUNT_NAME
        and _either(mount, 'mountPath', 'mount_path') == DATA_DIR_MOUNT_PATH
        and host_path.get('path') == DATA_DIR_HOST_PATH
    )


def _volume_mounts(k8s_options: dict) -> list:
    """The stored volume mounts, under either key form."""
    for key in ('volumeMounts', 'volume_mounts'):
        mounts = k8s_options.get(key)
        if isinstance(mounts, list):
            return mounts
    return []


def backfilled(k8s_options) -> Union[dict, None]:
    """The cluster's ``k8s_options`` with the data-dir mount in the reserved slot.

    None when nothing needs writing. An empty ``volumeMounts`` is the signal to
    backfill: the field did not exist before v2.2.0, and every write since then
    puts the data dir at index 0, so a non-empty list already has it — and its
    host path may have been set deliberately (or backfilled by hand with the
    workaround for the issue this migration fixes).
    """
    k8s_options = _as_dict(k8s_options)
    mounts = _volume_mounts(k8s_options)
    if mounts:
        return None

    # The column is written by ``pydantic_column_type``, which encodes by
    # alias, so the camelCase key is the one already in the row.
    k8s_options.pop('volume_mounts', None)
    k8s_options['volumeMounts'] = [
        {
            'name': DATA_DIR_MOUNT_NAME,
            'mountPath': DATA_DIR_MOUNT_PATH,
            'readOnly': False,
            'volumeSource': {
                'hostPath': {
                    'path': DATA_DIR_HOST_PATH,
                    'type': 'DirectoryOrCreate',
                }
            },
        }
    ]
    return k8s_options


def upgrade() -> None:
    conn = op.get_bind()

    rows = conn.execute(
        sa.text(
            "SELECT id, k8s_options FROM clusters WHERE provider = 'Kubernetes'"
        )
    ).fetchall()

    for cluster_id, k8s_options in rows:
        updated = backfilled(k8s_options)
        if updated is None:
            continue

        conn.execute(
            sa.update(clusters_jsonable)
            .where(clusters_jsonable.c.id == cluster_id)
            .values(k8s_options=updated)
        )


def downgrade() -> None:
    conn = op.get_bind()

    rows = conn.execute(
        sa.text(
            "SELECT id, k8s_options FROM clusters WHERE provider = 'Kubernetes'"
        )
    ).fetchall()

    for cluster_id, k8s_options in rows:
        k8s_options = _as_dict(k8s_options)
        mounts = _volume_mounts(k8s_options)
        if not mounts:
            continue

        # Only drop the exact entry this migration would have written, in the
        # slot it would have written it. One pointing at a different host path
        # was configured by the user, and the pre-upgrade code renders it fine —
        # it is the *absence* of the mount that the older server tolerated, not
        # a foreign host path.
        if not _is_backfilled_data_dir_mount(mounts[0]):
            continue

        kept = mounts[1:]
        k8s_options.pop('volume_mounts', None)
        if kept:
            k8s_options['volumeMounts'] = kept
        else:
            k8s_options.pop('volumeMounts', None)

        conn.execute(
            sa.update(clusters_jsonable)
            .where(clusters_jsonable.c.id == cluster_id)
            .values(k8s_options=k8s_options or None)
        )
