"""backfill k8s data-dir volume mount

Revision ID: c8f3a1b9d2e4
Revises: b7e2c4d15a80
Create Date: 2026-08-27 17:45:00.000000

K8s clusters created before v2.2.0 have no ``k8s_options.volumeMounts``.
The worker DaemonSet now renders ``/var/lib/gpustack`` from
``volumeMounts[0]`` instead of hardcoding it, and create/update require
that entry. Backfill the default hostPath mount so upgraded clusters
keep persisting worker data and can be edited without a validation
deadlock (see https://github.com/gpustack/gpustack/issues/6145).
"""

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c8f3a1b9d2e4"
down_revision: Union[str, None] = "b7e2c4d15a80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DEFAULT_DATA_DIR_MOUNT = {
    "name": "gpustack-data-dir",
    "mountPath": "/var/lib/gpustack",
    "readOnly": False,
    "volumeSource": {
        "hostPath": {"path": "/var/lib/gpustack", "type": "DirectoryOrCreate"}
    },
}


def _parse_json(value):
    if isinstance(value, str):
        try:
            value = json.loads(value) if value else {}
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return value


def _needs_data_dir_mount(k8s_options: dict) -> bool:
    mounts = k8s_options.get("volumeMounts") or k8s_options.get("volume_mounts")
    if not mounts:
        return True
    first = mounts[0] if isinstance(mounts, list) else None
    if not isinstance(first, dict):
        return True
    source = first.get("volumeSource") or first.get("volume_source") or {}
    if not isinstance(source, dict):
        return True
    return source.get("hostPath") is None and source.get("host_path") is None


def upgrade() -> None:
    conn = op.get_bind()
    clusters_jsonable = sa.table(
        "clusters",
        sa.column("id", sa.Integer),
        sa.column("k8s_options", sa.JSON),
    )
    rows = conn.execute(
        sa.text("SELECT id, k8s_options FROM clusters WHERE provider = 'Kubernetes'")
    ).fetchall()
    for cluster_id, k8s_options in rows:
        k8s_options = _parse_json(k8s_options)
        if not _needs_data_dir_mount(k8s_options):
            continue
        mounts = k8s_options.get("volumeMounts") or k8s_options.get("volume_mounts") or []
        if not isinstance(mounts, list):
            mounts = []
        k8s_options.pop("volume_mounts", None)
        k8s_options["volumeMounts"] = [_DEFAULT_DATA_DIR_MOUNT] + [
            m
            for m in mounts
            if not (isinstance(m, dict) and m.get("name") == "gpustack-data-dir")
        ]
        conn.execute(
            sa.update(clusters_jsonable)
            .where(clusters_jsonable.c.id == cluster_id)
            .values(k8s_options=k8s_options)
        )


def downgrade() -> None:
    pass
