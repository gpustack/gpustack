"""v2.3.0 database changes

Bundles the pre-release schema changes for v2.3.0:

1. GPU instance types: a ``gpu_instance_types`` table holding the per-cluster
   catalog of offerable types, plus ``gpu_instances.type_snapshot`` recording
   the type an instance was created from, so later edits to the catalog don't
   retroactively change existing instances.

2. ``models.scaling_schedule`` for scheduled scaling: a per-model cron
   timetable that drives the model's replica count. The column stores the
   serialized ``ScalingSchedule`` (enabled flag, ``baseline_replicas``, and the
   list of ``start_cron`` + ``duration_seconds`` + ``replicas`` window rules);
   NULL means no schedule is configured.

3. Cleanup of SYSTEM principals (and their registration API keys) leaked by
   cluster / worker deletes — data only, no schema change. See
   :func:`_cleanup_orphan_system_principals`.

Revision ID: 367a3982fcde
Revises: c4d7e8f9a0b1
Create Date: 2026-07-15 16:00:00.000000

"""
import logging
from typing import List, Sequence, Set, Tuple, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack

# revision identifiers, used by Alembic.
revision: str = '367a3982fcde'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger(__name__)

# Bootstrap SYSTEM principals are named ``system/cluster-<suffix>`` /
# ``system/worker-<suffix>`` — see ``gpustack.schemas.users.system_name_prefix``
# and ``gpustack.routes.workers.system_name_prefix``. Scoping the cleanup to
# these two prefixes keeps it away from any other SYSTEM row (legacy
# ``is_system=true`` rows promoted by the multi-tenancy migration may carry
# unrelated names).
_CLUSTER_PRINCIPAL_PREFIX = 'system/cluster-'
_WORKER_PRINCIPAL_PREFIX = 'system/worker-'

_CHUNK_SIZE = 500


def upgrade() -> None:
    op.create_table(
        'gpu_instance_types',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('spec', gpustack.schemas.common.JSON(), nullable=False),
        sa.Column('status', gpustack.schemas.common.JSON(), nullable=True),
        sa.Column('snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(['cluster_id'], ['clusters.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('snapshot', name='uq_gpu_instance_type_snapshot'),
    )

    with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'type_snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=True
            )
        )

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('scaling_schedule', sa.JSON(), nullable=True))

    _cleanup_orphan_system_principals()


def downgrade() -> None:
    # The orphan principal cleanup is data-only — the deleted rows (and their
    # credentials) can't be reconstructed, so there is nothing to undo for it.
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('scaling_schedule')

    with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
        batch_op.drop_column('type_snapshot')

    op.drop_table('gpu_instance_types')


def _cleanup_orphan_system_principals() -> None:
    """Remove SYSTEM principals leaked by cluster / worker deletes.

    ``Cluster.system_principal`` / ``Worker.system_principal`` are ORM-level
    ``cascade="delete"`` relationships declared ``lazy="noload"``, and the
    delete paths didn't eager load them — ``_handle_cascade_delete`` reads the
    attribute off the instance, so the cascade silently no-oped. The FK runs
    the other way (``clusters.system_principal_id`` /
    ``workers.system_principal_id`` → ``principals.id ON DELETE SET NULL``),
    so nothing cleaned up on the DB side either: every deleted cluster /
    worker leaked its bootstrap principal *and* the registration ``api_keys``
    row hanging off it — a credential that still authenticates.

    The delete paths now eager load the relationship; this removes the rows
    that already leaked.
    """
    conn = op.get_bind()

    candidates = _orphan_candidates(conn)
    if not candidates:
        logger.info("No orphan SYSTEM principals found.")
        return

    # Second guard, on top of the ``system_principal_id`` check: a *live*
    # cluster / worker whose link the multi-tenancy backfill never resolved
    # would otherwise look orphaned, and deleting its principal would break
    # worker registration for it. Spare anything whose name a surviving row
    # could still be using.
    live_names = _live_principal_names(conn)
    orphans = [(pid, name) for pid, name in candidates if name not in live_names]
    spared = len(candidates) - len(orphans)
    if spared:
        logger.info(
            f"Skipped {spared} SYSTEM principal(s) whose name matches a "
            "surviving cluster / worker."
        )
    if not orphans:
        return

    ids = [pid for pid, _ in orphans]
    # Delete the keys explicitly rather than leaning on
    # ``api_keys.user_id ON DELETE CASCADE`` — the constraint's ``ondelete`` is
    # only guaranteed for DBs that went through the migration that created it.
    #
    # The keys go first, and deliberately *outside* the principal deletes'
    # SAVEPOINT: the live credential is the part that matters, so a principal
    # the DB refuses to drop (see ``_delete_principals``) should still lose its
    # key. Rolling the two back together would resurrect a registration token
    # for a cluster / worker that no longer exists. What's left behind is an
    # inert principal row — whatever still references it (usage attribution,
    # ``cluster_access.granted_by``) needs the row, not the key.
    keys = _delete_api_keys(conn, ids)
    deleted, skipped = _delete_principals(conn, ids)
    logger.info(
        f"Cleaned up {deleted} orphan SYSTEM principal(s) and {keys} stale "
        "registration API key(s)."
    )
    if skipped:
        logger.warning(
            f"Left {len(skipped)} orphan SYSTEM principal(s) in place, still "
            f"referenced by other rows: {skipped}"
        )


def _orphan_candidates(conn) -> List[Tuple[int, str]]:
    """Bootstrap SYSTEM principals no cluster / worker row points at.

    ``kind`` is compared against a literal because it is a native enum on
    PostgreSQL (``principaltype``); a bound parameter would need an explicit
    cast. The ``NOT EXISTS`` checks deliberately ignore ``deleted_at``: a
    soft-deleted worker still owns its principal until its teardown finishes.
    """
    rows = conn.execute(
        sa.text(
            """
            SELECT p.id, p.name
              FROM principals p
             WHERE p.kind = 'SYSTEM'
               AND (p.name LIKE :cluster_prefix OR p.name LIKE :worker_prefix)
               AND NOT EXISTS (
                   SELECT 1 FROM clusters c WHERE c.system_principal_id = p.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM workers w WHERE w.system_principal_id = p.id
               )
            """
        ),
        {
            'cluster_prefix': f'{_CLUSTER_PRINCIPAL_PREFIX}%',
            'worker_prefix': f'{_WORKER_PRINCIPAL_PREFIX}%',
        },
    ).all()
    return [(row[0], row[1]) for row in rows]


def _live_principal_names(conn) -> Set[str]:
    """Principal names a surviving cluster / worker row could still be using.

    Clusters are covered fully: the name is either
    ``system/cluster-<hashed_suffix>`` (route-created) or
    ``system/cluster-<id>`` (the v2.0 seed of the default cluster). Workers are
    only partially covered — registration derives the suffix from a fresh
    ``secrets.token_hex(6)`` that is never stored on the worker row, so a live
    worker with a NULL ``system_principal_id`` and a route-created principal
    can't be matched here. That combination requires the multi-tenancy backfill
    to have missed the row, which post-v2.2 registrations can't produce.
    """
    names: Set[str] = set()
    for cluster_id, hashed_suffix in conn.execute(
        sa.text("SELECT id, hashed_suffix FROM clusters")
    ).all():
        # ``hashed_suffix`` is NOT NULL since v2.0, but legacy installs can
        # still carry an empty value (the v2.2 migration guards it the same
        # way) — don't fabricate a ``system/cluster-None`` entry from one.
        # Such a cluster is still covered by the ``system_principal_id``
        # check in :func:`_orphan_candidates`.
        if hashed_suffix:
            names.add(f'{_CLUSTER_PRINCIPAL_PREFIX}{hashed_suffix}')
        names.add(f'{_CLUSTER_PRINCIPAL_PREFIX}{cluster_id}')
    for (worker_id,) in conn.execute(sa.text("SELECT id FROM workers")).all():
        names.add(f'{_WORKER_PRINCIPAL_PREFIX}{worker_id}')
    return names


def _chunked(values: List[int]) -> List[List[int]]:
    return [
        values[start : start + _CHUNK_SIZE]
        for start in range(0, len(values), _CHUNK_SIZE)
    ]


def _delete_api_keys(conn, principal_ids: List[int]) -> int:
    stmt = sa.text("DELETE FROM api_keys WHERE user_id IN :ids").bindparams(
        sa.bindparam('ids', expanding=True)
    )
    deleted = 0
    for chunk in _chunked(principal_ids):
        deleted += conn.execute(stmt, {'ids': chunk}).rowcount or 0
    return deleted


def _delete_principals(conn, principal_ids: List[int]) -> Tuple[int, List[int]]:
    """Delete the principals, reporting any the DB refuses to drop.

    An unexpected reference (an FK without ``ON DELETE CASCADE`` — say a stray
    ``model_usages`` row) shouldn't abort the upgrade or take the rest of the
    batch with it, so each batch runs in a SAVEPOINT and falls back to
    row-by-row on conflict.
    """
    batch_stmt = sa.text("DELETE FROM principals WHERE id IN :ids").bindparams(
        sa.bindparam('ids', expanding=True)
    )
    row_stmt = sa.text("DELETE FROM principals WHERE id = :id")
    deleted = 0
    skipped: List[int] = []
    for chunk in _chunked(principal_ids):
        try:
            with conn.begin_nested():
                result = conn.execute(batch_stmt, {'ids': chunk})
            deleted += result.rowcount or 0
            continue
        except sa.exc.IntegrityError:
            pass
        for principal_id in chunk:
            try:
                with conn.begin_nested():
                    result = conn.execute(row_stmt, {'id': principal_id})
                deleted += result.rowcount or 0
            except sa.exc.IntegrityError:
                skipped.append(principal_id)
    return deleted, skipped
