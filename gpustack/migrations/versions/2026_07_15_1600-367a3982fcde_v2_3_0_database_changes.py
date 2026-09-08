"""v2.3.0 database changes

Bundles the pre-release schema changes for v2.3.0:

1. GPU instance types: a ``gpu_instance_types`` table holding the per-cluster
   catalog of offerable types, ``gpu_instances.type_snapshot`` recording the
   type an instance was created from (so later edits to the catalog don't
   retroactively change existing instances), and ``models.gpu_type_selector``,
   the per-model constraint that picks which of those types a model deploys
   onto — the catalog, the instance's frozen copy, and the model's choice being
   one feature.

   ``gpu_instance_types.derived_from_node`` records whether the operator derived
   the type from a node's resource flavors (the
   ``schedule.gpustack.ai/derived-from-node`` label). It cannot be inferred from
   anything else the row holds — the cluster-level setting is a fleet switch, not
   a row provenance — and it is deliberately outside ``snapshot``, which is the
   row's identity and doubles as ``metered_usage.sku``.

2. ``models.scaling_schedule`` for scheduled scaling: a per-model cron
   timetable that drives the model's replica count. The column stores the
   serialized ``ScalingSchedule`` (enabled flag, ``baseline_replicas``, and the
   list of ``start_cron`` + ``duration_seconds`` + ``replicas`` window rules);
   NULL means no schedule is configured.

3. ``api_keys.secret_key_digest``: a cheap second verifier
   (``sha256$<salt>$<hash>``) for the secrets GPUStack generates itself, so
   authentication no longer has to pay argon2 on the request path. Nullable and
   additive — ``hashed_secret_key`` keeps its argon2 value for every key, so an
   older version pointed at the same database still verifies everything. Existing
   rows stay NULL: the plaintext is returned to the caller once at creation and
   never stored, so it can only be filled in on a later successful verification.

4. Cleanup of SYSTEM principals (and their registration API keys) leaked by
   cluster / worker deletes — data only, no schema change. See
   :func:`_cleanup_orphan_system_principals`.
   
5. Benchmark load curves: a ``benchmark_results`` child table holding one row
   per measured point, and the ``benchmarks`` config/result columns behind it
   (load axis, adaptive auto-tune budget, the nine latency-SLO thresholds, the
   token-length distribution, seed policy, stop conditions, and the computed
   best operating points). The parent's flat ``*_mean`` columns stay as the
   representative (throughput-peak) point, so this is an additive change.

6. Metering keyed on the instance type snapshot, with fractional counts and a
   per-shape natural key. Three coupled changes — splitting them would leave an
   intermediate state that can neither meter a sliced accelerator correctly nor
   split a mid-hour reconfiguration into separate billing segments:

   - ``metered_usage.sku_count`` / ``metered_usage_archive.sku_count``:
     ``INTEGER`` -> ``NUMERIC(20, 8)``. A sliced accelerator bills a fraction of
     a card (0.5 = half a card, ~0.119 for a MIG ``1g.10gb``), which an integer
     column silently rounded up to a whole card. Purely a widening: existing
     ``1`` / ``2`` / ``4`` read back as ``1.0`` / ``2.0`` / ``4.0``, so no row's
     meaning changes and no backfill is needed.
   - ``uq_metered_usage``: ``(meter_key, resource_id, bucket_start)`` ->
     ``(meter_key, resource_id, bucket_start, sku, sku_count)``. The old key was
     "one row per resource-hour", so a spec change inside one UTC hour re-rated
     the whole hour by whichever shape landed last. Adding the two columns that
     decide the amount makes each shape its own row. Also a widening — existing
     rows were already unique on the narrower key — so it needs no data fix.
   - New columns: ``metered_usage.definition_snapshot`` (indexed, the type's
     cluster-independent definition hash, for cross-cluster aggregation and bulk
     pricing), ``metered_usage.instance_type_name`` (because ``sku`` becomes an
     opaque ``sha1:<40hex>`` and every human-facing label has to come from
     somewhere), and ``gpu_instance_types.definition_snapshot`` (indexed) as the
     source of the first.

   The two metered tables must stay column-identical: archival is a bulk
   ``INSERT ... SELECT``.

   ``gpu_instance_types.definition_snapshot`` is NOT backfilled: it is a hash
   over the pydantic ``spec`` model dumped by field name, while the persisted
   JSON is alias(camelCase)-keyed, so reproducing it in raw SQL would risk a
   value that silently disagrees with the one the running code computes.
   ``GPUInstanceTypeController`` fills every active row in place on its first
   watch re-LIST after the upgrade (which happens on every server start), and
   the metering read path derives the value on the fly for soft-deleted rows,
   which are never re-LISTed.

7. ``models.native_anthropic_api``: whether a deployment's inference server
   implements the Anthropic Messages API itself, so the gateway can forward an
   inbound ``/v1/messages`` untouched instead of translating it to
   ``/v1/chat/completions``. Declared per deployment rather than derived from
   the backend because the answer belongs to the running image, and the image
   is only settled per instance — one deployment can spread over workers of
   different accelerators whose images need not agree, while a single ai-proxy
   provider entry has to cover the whole deployment. NOT NULL with a ``false``
   server default, which backfills existing rows to the pre-existing behavior
   and leaves no NULL/false ambiguity for callers.

8. ``model_instances.dp_rank`` for the vLLM data-parallel node-per-instance
   path: every DP node becomes a standalone ``ModelInstance`` carrying its own
   rank. Nullable and additive — NULL for every non-DP instance. See
   :func:`_upgrade_model_instance_dp_rank`.

Revision ID: 367a3982fcde
Revises: c4d7e8f9a0b1
Create Date: 2026-07-15 16:00:00.000000

"""
import logging
from typing import List, Sequence, Set, Tuple, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector
import sqlmodel
import gpustack
from gpustack.schemas.common import JSON, UTCDateTime
from gpustack.migrations.utils import column_exists, table_exists

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

_METERED_TABLES = ('metered_usage', 'metered_usage_archive')

# Precision mirrors gpustack.schemas.metered_usage.SKU_COUNT_{PRECISION,SCALE};
# duplicated as literals so the migration does not import app code.
_SKU_COUNT_TYPE = sa.Numeric(precision=20, scale=8)

_UQ_METERED_USAGE = 'uq_metered_usage'
_UQ_NARROW = ['meter_key', 'resource_id', 'bucket_start']
_UQ_WIDE = _UQ_NARROW + ['sku', 'sku_count']

# ── Benchmark load curves ─────────────────────────────────────────────────────
# Flat metric columns shared by ``benchmarks`` (the representative point) and
# ``benchmark_results`` (the per-point grid) — i.e. BenchmarkMetricsLite.
_METRIC_FLOAT_COLS = [
    'requests_per_second_mean',
    'request_latency_mean',
    'time_per_output_token_mean',
    'inter_token_latency_mean',
    'time_to_first_token_mean',
    'tokens_per_second_mean',
    'output_tokens_per_second_mean',
    'input_tokens_per_second_mean',
    'request_concurrency_mean',
    'request_concurrency_max',
]
_METRIC_INT_COLS = [
    'request_total',
    'request_successful',
    'request_errored',
    'request_incomplete',
]
# Measured percentiles of the SLO-relevant latency metrics. Both aggregations
# are stored because a threshold can only be evaluated against its own
# percentile. NOT backfilled: rows written before this revision have neither,
# and simply never match a p95/p99 threshold.
_METRIC_PCT_COLS = [
    'time_to_first_token_p95',
    'time_to_first_token_p99',
    # Decode-only per-token time (guidellm's inter_token_latency): the industry
    # TPOT, and what the `slo_*_tpot_ms` thresholds are evaluated against.
    'inter_token_latency_p95',
    'inter_token_latency_p99',
    # The includes-TTFT variant. Still recorded, no longer judged on.
    'time_per_output_token_p95',
    'time_per_output_token_p99',
    'request_latency_p95',
    'request_latency_p99',
]

# All nullable columns added to ``benchmarks`` (name, type, server_default).
_BENCHMARK_COLUMNS = [
    # Load axis: fixed_rate (open-loop req/s) or concurrency (closed-loop).
    ('load_type', sqlmodel.sql.sqltypes.AutoString(), None),
    # Latency-SLO targets: 3 metrics x 3 aggregations, every one an optional
    # "<= ms" threshold. A point meets the SLO when every threshold that was
    # set holds AND its success rate clears the floor.
    ('slo_avg_ttft_ms', sa.Float(), None),
    ('slo_p95_ttft_ms', sa.Float(), None),
    ('slo_p99_ttft_ms', sa.Float(), None),
    ('slo_avg_tpot_ms', sa.Float(), None),
    ('slo_p95_tpot_ms', sa.Float(), None),
    ('slo_p99_tpot_ms', sa.Float(), None),
    ('slo_avg_latency_ms', sa.Float(), None),
    ('slo_p95_latency_ms', sa.Float(), None),
    ('slo_p99_latency_ms', sa.Float(), None),
    # Measured percentiles of the representative point (mirrors the sub-table).
    *[(c, sa.Float(), None) for c in _METRIC_PCT_COLS],
    # Computed conclusions + run constraints + multi-turn.
    ('slo_met_rate', sa.Float(), None),
    ('recommended_rate', sa.Float(), None),
    ('peak_rate', sa.Float(), None),
    ('validity', JSON(), None),
    ('turns', sa.Integer(), None),
    ('warmup', sa.Float(), None),
    ('cooldown', sa.Float(), None),
    ('max_errors', sa.Integer(), None),
    ('max_error_rate', sa.Float(), None),
    ('stop_on_saturation', sa.Boolean(), None),
    # Token-length distribution (spread the load instead of one fixed length).
    ('dataset_input_stdev', sa.Integer(), None),
    ('dataset_input_min', sa.Integer(), None),
    ('dataset_input_max', sa.Integer(), None),
    ('dataset_output_stdev', sa.Integer(), None),
    ('dataset_output_min', sa.Integer(), None),
    ('dataset_output_max', sa.Integer(), None),
    # Seed policy. `_increment` gives each point its own seed so successive
    # points don't replay one another's prefix cache; `_random` records where
    # the base seed came from (generated vs pinned for a reproducible re-run).
    ('dataset_seed_increment', sa.Boolean(), sa.true()),
    ('dataset_seed_random', sa.Boolean(), sa.true()),
    # Manual stages / shared prefix buckets.
    ('stages', JSON(), None),
    ('prefix_buckets', JSON(), None),
    # Global duration cap for non-stage runs.
    ('max_seconds', sa.Float(), None),
    # Adaptive auto-tune: the toggle plus its hard search range and budget.
    ('auto_tune', sa.Boolean(), None),
    ('lower_bound', sa.Float(), None),
    ('upper_bound', sa.Float(), None),
    ('max_points', sa.Integer(), None),
    ('max_total_seconds', sa.Float(), None),
]


def _gpu_instance_type_late_columns() -> List[sa.Column]:
    """The ``gpu_instance_types`` columns added after the pre-release revision.

    Returned as fresh objects because they are consumed by either
    ``create_table`` or ``add_column`` — a ``Column`` cannot be attached twice.
    """
    return [
        # Cluster-independent twin of ``snapshot``: the same definition rolled
        # out to N clusters gives N snapshots but one of these. Non-unique by
        # design.
        sa.Column(
            'definition_snapshot',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
        ),
        # Provenance of the type: set when the operator derived it from a node's
        # resource flavors. ``server_default`` is what lets the same definition
        # serve the ``add_column`` path, where the table may already hold rows a
        # NOT NULL column would otherwise reject.
        sa.Column(
            'derived_from_node',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    ]


def upgrade() -> None:
    """Every step checks for its own object first.

    This bundle folded in several revisions that shipped separately during
    pre-release, so a development database may already carry some of these objects.
    Skipping what is present lets such a database be stamped back to the previous
    revision and migrated forward instead of rebuilt.

    The guards have to cover the WHOLE revision to be worth anything. They used to
    sit only in `_upgrade_benchmark_load_curves()`, which is reached fourth: a
    database that already had `gpu_instance_types` failed on the first statement and
    never got to the guarded part, so the promise above held for one section and was
    false for the revision.
    """
    if not table_exists('gpu_instance_types'):
        op.create_table(
            'gpu_instance_types',
            sa.Column(
                'created_at', gpustack.schemas.common.UTCDateTime(), nullable=False
            ),
            sa.Column(
                'updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False
            ),
            sa.Column(
                'deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True
            ),
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column('spec', gpustack.schemas.common.JSON(), nullable=False),
            sa.Column('status', gpustack.schemas.common.JSON(), nullable=True),
            sa.Column('snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            *_gpu_instance_type_late_columns(),
            sa.ForeignKeyConstraint(
                ['cluster_id'], ['clusters.id'], ondelete='CASCADE'
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('snapshot', name='uq_gpu_instance_type_snapshot'),
        )
    else:
        # The table predates these columns on databases that ran the pre-release
        # revisions which created it. One batch over whichever are missing, not a
        # block per column: on SQLite each block recreates and copies the whole
        # table. An `elif` chain would be outright wrong — a database missing BOTH
        # columns would take the first branch and never get the second one.
        missing = [
            c
            for c in _gpu_instance_type_late_columns()
            if not column_exists('gpu_instance_types', c.name)
        ]
        if missing:
            with op.batch_alter_table('gpu_instance_types', schema=None) as batch_op:
                for column in missing:
                    batch_op.add_column(column)

    if not _index_exists(
        'gpu_instance_types', 'ix_gpu_instance_types_definition_snapshot'
    ):
        op.create_index(
            'ix_gpu_instance_types_definition_snapshot',
            'gpu_instance_types',
            ['definition_snapshot'],
            unique=False,
        )

    # Backs the fleet-wide list read: WHERE deleted_at IS NULL AND cluster_id
    # IN (...) ORDER BY created_at DESC, plus its COUNT. The table previously
    # only ever served a single cluster's rows to the controller, so it never
    # needed this; the list route now scans it on every page load.
    if not _index_exists(
        'gpu_instance_types',
        'idx_gpu_instance_types_deleted_at_cluster_id_created_at',
    ):
        op.create_index(
            'idx_gpu_instance_types_deleted_at_cluster_id_created_at',
            'gpu_instance_types',
            ['deleted_at', 'cluster_id', 'created_at'],
            unique=False,
        )

    _upgrade_metering_sku_shape()

    if not column_exists('gpu_instances', 'type_snapshot'):
        with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    'type_snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=True
                )
            )

    # One batch for all the columns. On SQLite, batch mode recreates and copies
    # the whole table per block, so separate blocks would rewrite `models` once
    # each; on PostgreSQL / MySQL each block is a plain ALTER and the grouping is
    # only tidiness. Written for the worst case, since the revision has to run on
    # all three.
    #
    # ``native_anthropic_api`` carries its own default rather than being
    # nullable: false is a real answer ("translate, as before"), and a NULL that
    # means the same thing as false only leaves callers guessing which to send.
    # Adding a NOT NULL column with a constant default is metadata-only on
    # PostgreSQL 11+ and MySQL 8, so the backfill costs nothing.
    models_columns = [
        sa.Column('scaling_schedule', sa.JSON(), nullable=True),
        sa.Column('gpu_type_selector', sa.JSON(), nullable=True),
        sa.Column(
            'native_anthropic_api',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    ]
    models_columns = [c for c in models_columns if not column_exists('models', c.name)]
    if models_columns:
        with op.batch_alter_table('models', schema=None) as batch_op:
            for column in models_columns:
                batch_op.add_column(column)

    _upgrade_benchmark_load_curves()

    if not column_exists('api_keys', 'secret_key_digest'):
        with op.batch_alter_table('api_keys', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    'secret_key_digest',
                    sqlmodel.sql.sqltypes.AutoString(),
                    nullable=True,
                )
            )

    _upgrade_model_instance_dp_rank()

    # Data-only and idempotent: it deletes orphans, so a replay finds none.
    _cleanup_orphan_system_principals()


def downgrade() -> None:
    _downgrade_metering_sku_shape()
    _downgrade_benchmark_load_curves()
    _downgrade_model_instance_dp_rank()

    # The orphan principal cleanup is data-only — the deleted rows (and their
    # credentials) can't be reconstructed, so there is nothing to undo for it.
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('secret_key_digest')

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('native_anthropic_api')
        batch_op.drop_column('gpu_type_selector')
        batch_op.drop_column('scaling_schedule')

    with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
        batch_op.drop_column('type_snapshot')

    op.drop_index(
        'idx_gpu_instance_types_deleted_at_cluster_id_created_at',
        table_name='gpu_instance_types',
    )
    op.drop_index(
        'ix_gpu_instance_types_definition_snapshot',
        table_name='gpu_instance_types',
    )
    op.drop_table('gpu_instance_types')


def _index_exists(table_name: str, index_name: str) -> bool:
    """Whether ``index_name`` is already defined on ``table_name``."""
    inspector = Inspector.from_engine(op.get_bind())
    return any(ix['name'] == index_name for ix in inspector.get_indexes(table_name))


def _unique_constraint_columns(table_name: str, constraint_name: str) -> List[str]:
    """Columns of a named unique constraint, or ``[]`` when it is absent."""
    inspector = Inspector.from_engine(op.get_bind())
    for uc in inspector.get_unique_constraints(table_name):
        if uc['name'] == constraint_name:
            return list(uc['column_names'])
    return []


def _sku_count_is_widened(table_name: str) -> bool:
    """Whether ``sku_count`` already carries the fractional type."""
    inspector = Inspector.from_engine(op.get_bind())
    for col in inspector.get_columns(table_name):
        if col['name'] == 'sku_count':
            return isinstance(col['type'], sa.Numeric)
    return False


def _upgrade_metering_sku_shape() -> None:
    """Fractional ``sku_count`` + the per-shape natural key + display columns.

    Each step guards on its own object, per this revision's contract.
    """
    for table in _METERED_TABLES:
        if not table_exists(table):
            continue
        new_columns = [
            sa.Column(
                'definition_snapshot',
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=True,
            ),
            sa.Column(
                'instance_type_name',
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=True,
            ),
        ]
        new_columns = [c for c in new_columns if not column_exists(table, c.name)]
        widen = not _sku_count_is_widened(table)
        if not new_columns and not widen:
            continue
        # One batch per table: batch mode copies and moves the whole table per
        # block, so a block per column would rewrite it repeatedly.
        with op.batch_alter_table(table, schema=None) as batch_op:
            if widen:
                batch_op.alter_column(
                    'sku_count',
                    existing_type=sa.Integer(),
                    type_=_SKU_COUNT_TYPE,
                    existing_nullable=False,
                    existing_server_default='1',
                )
            for column in new_columns:
                batch_op.add_column(column)

    # Only the hot table is indexed: the archive is read by ad-hoc audit
    # queries, not the dashboard, so an index there would only tax archival.
    if table_exists('metered_usage') and not _index_exists(
        'metered_usage', 'ix_metered_usage_definition_snapshot'
    ):
        op.create_index(
            'ix_metered_usage_definition_snapshot',
            'metered_usage',
            ['definition_snapshot'],
            unique=False,
        )

    # Drop + recreate rather than alter: a unique constraint's column list is
    # not alterable. Existing rows are already unique on the narrower key, so
    # the wider one is satisfied by construction and the window between the two
    # statements cannot admit a duplicate the old key would have rejected.
    if not table_exists('metered_usage'):
        return
    current = _unique_constraint_columns('metered_usage', _UQ_METERED_USAGE)
    if sorted(current) == sorted(_UQ_WIDE):
        return
    with op.batch_alter_table('metered_usage', schema=None) as batch_op:
        if current:
            batch_op.drop_constraint(_UQ_METERED_USAGE, type_='unique')
        batch_op.create_unique_constraint(_UQ_METERED_USAGE, _UQ_WIDE)


def _downgrade_metering_sku_shape() -> None:
    """Undo :func:`_upgrade_metering_sku_shape`.

    Guarded step by step, like the upgrade and for the same reason: a partially
    applied upgrade (one that failed between two of its blocks) must still be
    reversible, and an unguarded ``drop_column`` on an object that was never
    added aborts the whole downgrade at the first such step.

    LOSSY, and not only in the obvious direction. Narrowing ``sku_count`` back to
    an integer converts, and the conversion ROUNDS on both PostgreSQL and MySQL:
    a half-card row (0.5) comes back as 1 — over-reporting — while a small MIG
    share (0.119) comes back as 0, which zeroes that row's contribution outright.
    Neither is a number anyone can reconcile, so a downgrade is an escape hatch
    for a failed upgrade, not a supported way to run on the old schema with data
    metered by the new one.
    """
    if table_exists('metered_usage'):
        # Narrowing the natural key can collide: any resource-hour that was split
        # into per-shape rows by the new key violates the old one. Fold those back
        # into a single row first — sum the seconds and keep the shape of the
        # latest-settled row, which is exactly the (wrong, but pre-existing)
        # behaviour the old key produced.
        _collapse_per_shape_rows()
        if sorted(
            _unique_constraint_columns('metered_usage', _UQ_METERED_USAGE)
        ) != sorted(_UQ_NARROW):
            with op.batch_alter_table('metered_usage', schema=None) as batch_op:
                if _unique_constraint_columns('metered_usage', _UQ_METERED_USAGE):
                    batch_op.drop_constraint(_UQ_METERED_USAGE, type_='unique')
                batch_op.create_unique_constraint(_UQ_METERED_USAGE, _UQ_NARROW)

        if _index_exists('metered_usage', 'ix_metered_usage_definition_snapshot'):
            op.drop_index(
                'ix_metered_usage_definition_snapshot', table_name='metered_usage'
            )

    for table in _METERED_TABLES:
        if not table_exists(table):
            continue
        drop = [
            c
            for c in ('instance_type_name', 'definition_snapshot')
            if column_exists(table, c)
        ]
        narrow = _sku_count_is_widened(table)
        if not drop and not narrow:
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            for column in drop:
                batch_op.drop_column(column)
            if narrow:
                batch_op.alter_column(
                    'sku_count',
                    existing_type=_SKU_COUNT_TYPE,
                    type_=sa.Integer(),
                    existing_nullable=False,
                    existing_server_default='1',
                )


def _collapse_per_shape_rows() -> None:
    """Merge rows that share the pre-migration natural key back into one.

    Keeps the row with the greatest ``settled_until`` (ties broken by ``id``) —
    the shape the old collector would have left in place — adds the other rows'
    ``quantity`` onto it, and deletes them.
    """
    bind = op.get_bind()
    groups = bind.execute(
        sa.text(
            "SELECT meter_key, resource_id, bucket_start FROM metered_usage "
            "GROUP BY meter_key, resource_id, bucket_start HAVING COUNT(*) > 1"
        )
    ).fetchall()
    for meter_key, resource_id, bucket_start in groups:
        params = {"m": meter_key, "b": bucket_start}
        # resource_id is nullable, so it needs IS NULL rather than = NULL. Bind
        # :r only when the clause actually references it — passing a parameter
        # the statement never mentions is an error on some backends.
        if resource_id is None:
            rid_clause = "resource_id IS NULL"
        else:
            rid_clause = "resource_id = :r"
            params["r"] = resource_id
        rows = bind.execute(
            sa.text(
                "SELECT id, quantity FROM metered_usage "
                f"WHERE meter_key = :m AND {rid_clause} AND bucket_start = :b "
                "ORDER BY settled_until DESC, id DESC"
            ),
            params,
        ).fetchall()
        if len(rows) < 2:
            continue
        keep_id = rows[0][0]
        total = sum(r[1] or 0 for r in rows)
        drop_ids = [r[0] for r in rows[1:]]
        bind.execute(
            sa.text("UPDATE metered_usage SET quantity = :q WHERE id = :id"),
            {"q": total, "id": keep_id},
        )
        bind.execute(
            sa.text("DELETE FROM metered_usage WHERE id IN :ids").bindparams(
                sa.bindparam("ids", expanding=True)
            ),
            {"ids": drop_ids},
        )


def _upgrade_benchmark_load_curves() -> None:
    """Add the per-point results table and the benchmark config/result columns.

    Each step checks for its own object first, as every other step of this revision
    does — see `upgrade`, which states why that matters for the revision as a whole.
    """
    if not table_exists('benchmark_results'):
        op.create_table(
            'benchmark_results',
            sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
            sa.Column('benchmark_id', sa.Integer(), nullable=False),
            sa.Column('input_tokens', sa.Integer(), nullable=True),
            sa.Column('rate', sa.Float(), nullable=True),
            sa.Column(
                'strategy_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True
            ),
            # Probe order, not load order: the ramp doubles, then bisects, so
            # the row sequence is the only record of how the curve was walked.
            sa.Column('sequence', sa.Integer(), nullable=False, server_default='0'),
            *[sa.Column(c, sa.Float(), nullable=True) for c in _METRIC_FLOAT_COLS],
            *[sa.Column(c, sa.Float(), nullable=True) for c in _METRIC_PCT_COLS],
            *[sa.Column(c, sa.Integer(), nullable=True) for c in _METRIC_INT_COLS],
            sa.Column('raw_metrics', JSON(), nullable=True),
            sa.Column('created_at', UTCDateTime(), nullable=False),
            sa.Column('updated_at', UTCDateTime(), nullable=False),
            sa.Column('deleted_at', UTCDateTime(), nullable=True),
            sa.ForeignKeyConstraint(
                ['benchmark_id'], ['benchmarks.id'], ondelete='CASCADE'
            ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(
            'ix_benchmark_results_benchmark_id',
            'benchmark_results',
            ['benchmark_id'],
            unique=False,
        )

    missing = [
        (name, col_type, server_default)
        for name, col_type, server_default in _BENCHMARK_COLUMNS
        if not column_exists('benchmarks', name)
    ]
    if missing:
        with op.batch_alter_table('benchmarks') as batch_op:
            for name, col_type, server_default in missing:
                batch_op.add_column(
                    sa.Column(
                        name, col_type, nullable=True, server_default=server_default
                    )
                )

    # Carry each finished single-point benchmark into the grid as a one-row
    # curve, so the detail page renders old and new runs through one path.
    metric_cols = ", ".join(_METRIC_FLOAT_COLS + _METRIC_INT_COLS)
    op.execute(
        f"""
        INSERT INTO benchmark_results (
            benchmark_id, rate, sequence,
            {metric_cols},
            raw_metrics, created_at, updated_at
        )
        SELECT
            id, request_rate, 0,
            {metric_cols},
            raw_metrics, created_at, updated_at
        FROM benchmarks
        WHERE request_total IS NOT NULL
          AND id NOT IN (SELECT benchmark_id FROM benchmark_results)
        """
    )


def _downgrade_benchmark_load_curves() -> None:
    if table_exists('benchmark_results'):
        op.drop_index(
            'ix_benchmark_results_benchmark_id', table_name='benchmark_results'
        )
        op.drop_table('benchmark_results')

    present = [
        name for name, _, _ in reversed(_BENCHMARK_COLUMNS)
        if column_exists('benchmarks', name)
    ]
    if present:
        with op.batch_alter_table('benchmarks') as batch_op:
            for name in present:
                batch_op.drop_column(name)


def _upgrade_model_instance_dp_rank() -> None:
    """``model_instances.dp_rank`` for the vLLM data-parallel node-per-instance
    path: each DP node is a standalone ``ModelInstance`` carrying its own rank
    (0 = coordinator). NULL for every other instance, so the column is fully
    backward compatible."""
    if column_exists('model_instances', 'dp_rank'):
        return
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dp_rank', sa.Integer(), nullable=True))


def _downgrade_model_instance_dp_rank() -> None:
    if not column_exists('model_instances', 'dp_rank'):
        return
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('dp_rank')


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
