"""Backfill metered_usage.sku_count from the snapshotted shape

``sku_count`` is the per-instance unit multiplier — GPU card count, or CPU
base-flavor unit count (e.g. 3 for a 3c6g instance on a 1c2g flavor). Older
metering wrote ``gpu_count or 1``, so every CPU row landed ``1`` regardless of
size. The Instance Types breakdown now groups by ``(sku, sku_count)`` (indexed
columns, not the JSON ``dimensions`` blob), so historical CPU rows would
collapse all sizes into one row until their ``sku_count`` is corrected.

Recompute ``sku_count`` from the row's own ``dimensions`` (which already carry
the correct ``cpu_milli`` / ``unit_cpu_milli`` / ``gpu_count``), for both the
live ``metered_usage`` table and its archive. GPU rows already stored the card
count, so this only changes CPU rows in practice. Done in Python, in batches,
so it is dialect-agnostic (the JSON extraction differs across PostgreSQL /
MySQL / SQLite).

Revision ID: d5e8f1a2b3c4
Revises: c4d7e8f9a0b1
Create Date: 2026-06-29 12:00:00.000000

"""
import json
from typing import Optional, Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd5e8f1a2b3c4'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Only instance-uptime rows carry an instance shape; storage rows are untouched.
_UPTIME_METER = 'instance.uptime'
_TABLES = ('metered_usage', 'metered_usage_archive')
_BATCH = 1000


def _coerce_dims(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _resolve_sku_count(dims: dict) -> int:
    """Mirror of gpustack.server.resource_usage_collector._resolve_sku_count,
    derived from a stored ``dimensions`` blob (kept inline so the migration has
    no app-import dependency)."""
    def _int(v) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    gpu_count = _int(dims.get('gpu_count')) or 0
    if gpu_count > 0:
        return gpu_count
    for total_key, unit_key in (
        ('cpu_milli', 'unit_cpu_milli'),
        ('memory_mib', 'unit_memory_mib'),
    ):
        total = _int(dims.get(total_key))
        unit = _int(dims.get(unit_key))
        if total and unit and unit > 0:
            return max(1, round(total / unit))
    return 1


def _table_exists(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        if not _table_exists(bind, table):
            continue
        rows = bind.execute(
            sa.text(
                f"SELECT id, sku_count, dimensions FROM {table} "
                f"WHERE meter_key = :m"
            ),
            {"m": _UPTIME_METER},
        ).fetchall()

        updates = []
        for row in rows:
            new_count = _resolve_sku_count(_coerce_dims(row[2]))
            if new_count != row[1]:
                updates.append({"id": row[0], "c": new_count})

        stmt = sa.text(f"UPDATE {table} SET sku_count = :c WHERE id = :id")
        for i in range(0, len(updates), _BATCH):
            bind.execute(stmt, updates[i : i + _BATCH])


def downgrade() -> None:
    # Irreversible data backfill: the original (mostly ``1``) CPU sku_count
    # values are not recoverable and were wrong anyway. No-op.
    pass
