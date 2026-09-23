"""pd disaggregation database changes

Everything prefill-decode disaggregation adds to the schema, in one revision:

1. **Prefill-decode disaggregation.** ``models.roles`` / ``disaggregation`` are
   user intent — a Model with ``roles`` set is a *group*: one pool, one router,
   one generation at a time. ``models.state`` / ``state_message`` /
   ``role_status`` / ``stale`` / ``degradations`` are server-owned status, all
   written by ``sync_model_status`` from one scan of the model's instances.
   ``ready_replicas`` stays a plain count of RUNNING instances: under PD a
   count no longer implies servability (3P1D with the router down is four
   RUNNING instances and zero service), so servability lives in ``state`` and
   per-role detail in ``role_status``. ``role_status`` is persisted rather than
   computed per request because the list endpoint returns models without their
   instances.

   On ``model_instances``, ``role`` / ``group_id`` / ``spec_digest`` /
   ``named_ports`` carry group membership and generation. ``group_id`` is a
   generation, not a replica index; pairing binds to it rather than to peer
   addresses because serving ports change on every rebuild. It is indexed
   because the reconcilers group by it.

2. **Cluster topology and gather.** ``clusters.topology`` records how far apart
   a cluster's workers are; ``models.gather`` is the per-deployment override of
   how tightly a group's members must sit together. Both NULL by default: a
   cluster with no layers declared gives every worker a leaf of its own and
   still schedules, and a NULL ``gather`` means "no requirement", which is what
   every model had before the field existed. ``gather`` is deliberately not
   part of ``model_spec_digest`` — it is a preference for the *next* scheduling
   decision, and a digest bump would restart every member to relocate none.

3. **Benchmark target mode.** A run aims either at one instance (an engine,
   straight at its own port) or at a route (the deployment, through the
   entrance clients call). Existing rows are backfilled to ``instance``, which
   is what they measured and also the default for a create body that says
   nothing.

4. **Measured per-interval ITL.** ``inter_token_latency_*`` holds one value per
   REQUEST (guidellm's field of that name, the industry's TPOT), so a single
   decode stall is divided away by that request's other gaps. The
   ``itl_per_chunk_*`` columns hold the other metric: the measured gaps between
   consecutive streamed outputs, one sample per gap, pooled across requests —
   what vLLM / SGLang / evalscope report as ITL, and what a stall shows up in.
   ``_max`` rides along because the worst single gap is the finding for a stall
   hunt. Left NULL rather than backfilled: NULL means "not measured", which has
   to stay distinguishable from "measured, and the gaps were 0 ms". Both
   ``benchmarks`` and ``benchmark_results`` get them because both mirror
   ``BenchmarkMetricsLite``.

5. **Soft scale-down and the restart guard.**
   ``model_instances.draining_since`` is on the row rather than in server
   memory because a restart mid-window would otherwise leave a member no router
   knows about and nothing will ever delete. ``models.restarting_since`` lets
   ``POST /models/{id}/restart`` refuse a second teardown while the first is
   still rebuilding — the fact is not derivable from the rows, so it is
   recorded, and it lapses on its own so a group that never converges can
   still be restarted.

Every column is additive and nullable: a deployment that sets none of them
behaves exactly as it did before.

Revision ID: f4a5b6c7d8e9
Revises: b4c5d6e7f8a9
Create Date: 2026-09-21 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = 'f4a5b6c7d8e9'
down_revision: Union[str, None] = 'b4c5d6e7f8a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_MODEL_COLUMNS = [
    lambda: sa.Column('roles', sa.JSON(), nullable=True),
    lambda: sa.Column('disaggregation', sa.JSON(), nullable=True),
    lambda: sa.Column('state', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('state_message', sa.Text(), nullable=True),
    lambda: sa.Column('role_status', sa.JSON(), nullable=True),
    lambda: sa.Column('stale', sa.Boolean(), nullable=True),
    lambda: sa.Column('degradations', sa.JSON(), nullable=True),
    lambda: sa.Column('gather', sa.JSON(), nullable=True),
    lambda: sa.Column('restarting_since', sa.DateTime(), nullable=True),
]

_MODEL_INSTANCE_COLUMNS = [
    lambda: sa.Column('role', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('group_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('spec_digest', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('named_ports', sa.JSON(), nullable=True),
    lambda: sa.Column('draining_since', sa.DateTime(), nullable=True),
]

_CLUSTER_COLUMNS = [
    lambda: sa.Column('topology', sa.JSON(), nullable=True),
]

_GROUP_ID_INDEX = 'ix_model_instances_group_id'

_ITL_TABLES = ('benchmarks', 'benchmark_results')
_ITL_COLUMNS = (
    'itl_per_chunk_mean',
    'itl_per_chunk_p95',
    'itl_per_chunk_p99',
    'itl_per_chunk_max',
)


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(
        index["name"] == index_name for index in inspector.get_indexes(table_name)
    )


def _add_missing(table: str, factories) -> None:
    """Add whichever of these columns aren't there yet, in one batch.

    One batch per table: on SQLite, batch mode recreates and copies the whole
    table per block, so a block per column would rewrite it once per column.
    """
    if not table_exists(table):
        return
    missing = [f() for f in factories if not column_exists(table, f().name)]
    if not missing:
        return
    with op.batch_alter_table(table, schema=None) as batch_op:
        for column in missing:
            batch_op.add_column(column)


def _drop_present(table: str, factories) -> None:
    if not table_exists(table):
        return
    present = [f().name for f in reversed(factories) if column_exists(table, f().name)]
    if not present:
        return
    with op.batch_alter_table(table, schema=None) as batch_op:
        for name in present:
            batch_op.drop_column(name)


def _backfill_state() -> None:
    """Seed ``state`` from the counters it is derived from.

    Every reader already falls back to those counters while the column is NULL,
    so this is not what keeps a fleet routable across the upgrade. It exists to
    keep the row self-consistent between the migration and the first reconcile,
    so that nothing has to hold two answers for one model.

    **The mapping must be exactly what the writer produces**, and only two
    values are reachable for a role-less model — which is every model that
    exists before this revision. ``state`` answers "can this serve", so one
    ready replica is ``running``; being short of the requested count is a
    ``ratio_unmet`` degradation beside it, not a lifecycle value. ``partial``
    means "members up and still unservable", which a role-less model cannot be,
    so writing it here would invent a value the writer never emits *and* one
    the servability gate reads as unroutable — the upgrade itself would take
    every partially-scaled model out of service. A model at ``replicas = 0``
    lands on ``pending``, which is what the writer returns for it too:
    "stopped" is an intent, not a state, and the enum has no value for it.
    """
    models = sa.table(
        'models',
        sa.column('state', sa.String()),
        sa.column('replicas', sa.Integer()),
        sa.column('ready_replicas', sa.Integer()),
    )
    op.execute(
        models.update()
        .where(models.c.state.is_(None))
        .values(
            state=sa.case(
                (models.c.ready_replicas <= 0, 'pending'),
                else_='running',
            )
        )
    )


def _add_benchmark_target_mode() -> None:
    if not table_exists("benchmarks") or column_exists("benchmarks", "target_mode"):
        return

    # Added nullable, backfilled, then left nullable: SQLite cannot add a NOT
    # NULL column with a server default in one step, and the model supplies the
    # default on every write anyway. A NULL that somehow survives reads as
    # `instance` on the way out, which is what such a row was.
    op.add_column(
        "benchmarks",
        sa.Column("target_mode", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.execute(
        "UPDATE benchmarks SET target_mode = 'instance' WHERE target_mode IS NULL"
    )


def _drop_benchmark_target_mode() -> None:
    if not table_exists("benchmarks") or not column_exists("benchmarks", "target_mode"):
        return
    op.drop_column("benchmarks", "target_mode")


def _add_itl_per_chunk() -> None:
    for table in _ITL_TABLES:
        if not table_exists(table):
            continue
        for column in _ITL_COLUMNS:
            # Checked per column, not per table: a run interrupted midway
            # through would otherwise skip the whole table on retry.
            if column_exists(table, column):
                continue
            op.add_column(table, sa.Column(column, sa.Float(), nullable=True))


def _drop_itl_per_chunk() -> None:
    for table in _ITL_TABLES:
        if not table_exists(table):
            continue
        for column in _ITL_COLUMNS:
            if not column_exists(table, column):
                continue
            op.drop_column(table, column)


def upgrade() -> None:
    _add_missing('models', _MODEL_COLUMNS)
    _add_missing('model_instances', _MODEL_INSTANCE_COLUMNS)
    _add_missing('clusters', _CLUSTER_COLUMNS)
    _backfill_state()

    if column_exists('model_instances', 'group_id') and not _index_exists(
        'model_instances', _GROUP_ID_INDEX
    ):
        op.create_index(
            _GROUP_ID_INDEX,
            'model_instances',
            ['group_id'],
            unique=False,
        )

    _add_benchmark_target_mode()
    _add_itl_per_chunk()


def downgrade() -> None:
    _drop_itl_per_chunk()
    _drop_benchmark_target_mode()

    if table_exists('model_instances') and _index_exists(
        'model_instances', _GROUP_ID_INDEX
    ):
        op.drop_index(_GROUP_ID_INDEX, table_name='model_instances')

    _drop_present('clusters', _CLUSTER_COLUMNS)
    _drop_present('model_instances', _MODEL_INSTANCE_COLUMNS)
    _drop_present('models', _MODEL_COLUMNS)
