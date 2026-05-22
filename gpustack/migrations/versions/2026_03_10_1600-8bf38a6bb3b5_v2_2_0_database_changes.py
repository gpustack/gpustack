"""v2.2.0 add worker version

Revision ID: 8bf38a6bb3b5
Revises: 8ad0f94c92e8
Create Date: 2026-03-10 16:00:00.000000

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import gpustack
from gpustack.migrations.utils import column_exists, table_exists
import gpustack.utils.sql_enum as sql_enum
from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.stmt import model_user_after_drop_view_stmt

# revision identifiers, used by Alembic.
revision: str = '8bf38a6bb3b5'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

model_instance_proxy_mode = sa.Enum(
    'WORKER',
    'DIRECT',
    'DELEGATED',
    name='modelinstanceproxymodeenum',
)
proxy_mode_to_add = ['TUNNEL']


def upgrade() -> None:
    conn = op.get_bind()

    with op.batch_alter_table('workers', schema=None) as batch_op:
        if not column_exists('workers', 'worker_version'):
            batch_op.add_column(
                sa.Column('worker_version', sa.String(100), nullable=True)
            )
        if not column_exists('workers', 'proxy_address'):
            batch_op.add_column(sa.Column('proxy_address', sa.String(255), nullable=True))
    
    sql_enum.add_enum_values(
        {'workers': 'proxy_mode'},
        model_instance_proxy_mode,
        *proxy_mode_to_add,
    )

    ### k8s volume mount
    if not column_exists('clusters', 'k8s_volume_mounts'):
        with op.batch_alter_table('clusters', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    'k8s_volume_mounts',
                    gpustack.schemas.common.JSON(),
                    nullable=True,
                )
            )
    ### end

    ### custom API_KEY
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        if not column_exists('api_keys', 'is_custom'):
            batch_op.add_column(sa.Column('is_custom', sa.Boolean(), nullable=True))
        if not column_exists('api_keys', 'scope'):
            batch_op.add_column(
                sa.Column('scope', gpustack.schemas.common.JSON(), nullable=True)
            )

    # Update existing API keys to have full access scope
    op.execute("UPDATE api_keys SET scope = '[\"*\"]' WHERE scope IS NULL")
    op.execute("UPDATE api_keys SET is_custom = false WHERE is_custom IS NULL")

    # Set scope to NOT NULL
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column(
            'scope',
            existing_type=gpustack.schemas.common.JSON(),
            nullable=False,
        )
        batch_op.alter_column(
            'is_custom',
            existing_type=sa.Boolean(),
            nullable=False,
        )
        
    ### Usage
    with op.batch_alter_table("model_usages", schema=None) as batch_op:
        if not column_exists("model_usages", "cluster_name"):
            batch_op.add_column(sa.Column("cluster_name", sa.String(255), nullable=True))
        if not column_exists("model_usages", "user_name"):
            batch_op.add_column(sa.Column("user_name", sa.String(255), nullable=True))
        if not column_exists("model_usages", "api_key_id"):
            batch_op.add_column(sa.Column("api_key_id", sa.Integer(), nullable=True))
        if not column_exists("model_usages", "api_key_name"):
            batch_op.add_column(sa.Column("api_key_name", sa.String(255), nullable=True))
        if not column_exists("model_usages", "provider_name"):
            batch_op.add_column(sa.Column("provider_name", sa.String(255), nullable=True))
        if not column_exists("model_usages", "provider_type"):
            batch_op.add_column(sa.Column("provider_type", sa.String(255), nullable=True))
        if not column_exists("model_usages", "api_key_is_custom"):
            batch_op.add_column(
                sa.Column("api_key_is_custom", sa.Boolean(), nullable=True)
            )
        if not column_exists("model_usages", "prompt_cached_token_count"):
            batch_op.add_column(
                sa.Column(
                    "prompt_cached_token_count",
                    sa.BigInteger(),
                    nullable=False,
                    server_default="0",
                )
            )
        if not column_exists("model_usages", "model_route_id"):
            batch_op.add_column(
                sa.Column("model_route_id", sa.Integer(), nullable=True)
            )
        if not column_exists("model_usages", "model_route_name"):
            batch_op.add_column(
                sa.Column("model_route_name", sa.String(255), nullable=True)
            )

        batch_op.drop_constraint("fk_model_usages_user_id_users", type_="foreignkey")
        batch_op.drop_constraint("fk_model_usages_model_id_models", type_="foreignkey")
        batch_op.drop_constraint(
            "fk_model_usages_provider_id_model_providers", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_model_usages_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_model_id_models",
            "models",
            ["model_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_provider_id_model_providers",
            "model_providers",
            ["provider_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_api_key_id_api_keys",
            "api_keys",
            ["api_key_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_model_route_id_model_routes",
            "model_routes",
            ["model_route_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_model_usages_model_route_id",
            ["model_route_id"],
            unique=False,
        )

    conn.execute(
        sa.text(
            """
            UPDATE model_usages
            SET
                cluster_name = (
                    SELECT clusters.name
                    FROM models
                    LEFT JOIN clusters ON clusters.id = models.cluster_id
                    WHERE models.id = model_usages.model_id
                ),
                user_name = (
                    SELECT users.username
                    FROM users
                    WHERE users.id = model_usages.user_id
                ),
                api_key_id = (
                    SELECT api_keys.id
                    FROM api_keys
                    WHERE api_keys.access_key = model_usages.access_key
                ),
                api_key_name = (
                    SELECT api_keys.name
                    FROM api_keys
                    WHERE api_keys.access_key = model_usages.access_key
                ),
                api_key_is_custom = (
                    SELECT api_keys.is_custom
                    FROM api_keys
                    WHERE api_keys.access_key = model_usages.access_key
                )
            """
        )
    )
    provider_rows = conn.execute(
        sa.text("SELECT id, name, config FROM model_providers")
    ).fetchall()
    provider_snapshots = []
    for provider_id, provider_name, config in provider_rows:
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                config = {}
        provider_type = config.get("type") if isinstance(config, dict) else None
        provider_snapshots.append(
            {
                "provider_id": provider_id,
                "provider_name": provider_name,
                "provider_type": provider_type,
            }
        )
    if provider_snapshots:
        conn.execute(
            sa.text(
                """
                UPDATE model_usages
                SET
                    provider_name = :provider_name,
                    provider_type = :provider_type
                WHERE provider_id = :provider_id
                """
            ),
            provider_snapshots,
        )

    # Backfill model_route_id / model_route_name on historical model_usages.
    # Pre-upgrade rows have no route attribution; for routes that the
    # platform created automatically alongside a local model
    # (``created_by_model = true``), we can reconstruct the link by taking
    # the route's first target and matching it back to the usage rows it
    # would have served — by ``model_id`` for model-backed targets, or by
    # ``(provider_id, provider_model_name)`` against
    # ``model_usages.(provider_id, model_name)`` for provider-backed ones.
    # Anything we can't attribute deterministically is left NULL so it
    # surfaces in the "Untracked" bucket rather than being misattributed.
    #
    # Set-based UPDATEs (two passes, one per target kind) instead of
    # per-route Python iteration, so the migration scales with the size of
    # model_usages, not the route count. PG / MySQL diverge on multi-table
    # UPDATE syntax, so we branch on dialect; the inner SELECT that picks
    # each route's first surviving target is identical.
    first_targets_select = """
        SELECT
            mr.id AS route_id,
            mr.name AS route_name,
            mrt.model_id AS target_model_id,
            mrt.provider_id AS target_provider_id,
            mrt.provider_model_name AS target_provider_model_name
        FROM model_routes mr
        JOIN model_route_targets mrt ON mrt.route_id = mr.id
        WHERE mr.created_by_model = TRUE
          AND mr.deleted_at IS NULL
          AND mrt.deleted_at IS NULL
          AND mrt.id = (
              SELECT MIN(inner_mrt.id)
              FROM model_route_targets inner_mrt
              WHERE inner_mrt.route_id = mr.id
                AND inner_mrt.deleted_at IS NULL
          )
    """

    dialect_name = conn.dialect.name
    if dialect_name == "postgresql":
        conn.execute(
            sa.text(
                f"""
                WITH first_targets AS ({first_targets_select})
                UPDATE model_usages mu
                SET model_route_id = ft.route_id,
                    model_route_name = ft.route_name
                FROM first_targets ft
                WHERE mu.model_route_id IS NULL
                  AND ft.target_model_id IS NOT NULL
                  AND mu.model_id = ft.target_model_id
                """
            )
        )
        conn.execute(
            sa.text(
                f"""
                WITH first_targets AS ({first_targets_select})
                UPDATE model_usages mu
                SET model_route_id = ft.route_id,
                    model_route_name = ft.route_name
                FROM first_targets ft
                WHERE mu.model_route_id IS NULL
                  AND ft.target_provider_id IS NOT NULL
                  AND ft.target_provider_model_name IS NOT NULL
                  AND mu.provider_id = ft.target_provider_id
                  AND mu.model_name = ft.target_provider_model_name
                """
            )
        )
    elif dialect_name == "mysql":
        conn.execute(
            sa.text(
                f"""
                UPDATE model_usages mu
                JOIN ({first_targets_select}) ft
                  ON ft.target_model_id IS NOT NULL
                 AND mu.model_id = ft.target_model_id
                SET mu.model_route_id = ft.route_id,
                    mu.model_route_name = ft.route_name
                WHERE mu.model_route_id IS NULL
                """
            )
        )
        conn.execute(
            sa.text(
                f"""
                UPDATE model_usages mu
                JOIN ({first_targets_select}) ft
                  ON ft.target_provider_id IS NOT NULL
                 AND ft.target_provider_model_name IS NOT NULL
                 AND mu.provider_id = ft.target_provider_id
                 AND mu.model_name = ft.target_provider_model_name
                SET mu.model_route_id = ft.route_id,
                    mu.model_route_name = ft.route_name
                WHERE mu.model_route_id IS NULL
                """
            )
        )
    else:
        raise Exception(
            f"Unsupported database dialect: {dialect_name}. "
            "Only PostgreSQL and MySQL are supported."
        )
    ### end
    
    ### Ensure model_instances has injected_backend_parameters column
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        if not column_exists('model_instances', 'injected_backend_parameters'):
            batch_op.add_column(
                sa.Column(
                    'injected_backend_parameters',
                    gpustack.schemas.common.JSON(),
                    nullable=True,
                )
            )
    ### end

    ### Per-request usage details (hot + cold-storage backup)
    def _details_columns():
        return [
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=True),
            sa.Column('user_name', sa.String(255), nullable=True),
            sa.Column('model_id', sa.Integer(), nullable=True),
            sa.Column('model_name', sa.String(255), nullable=False),
            sa.Column('model_route_id', sa.Integer(), nullable=True),
            sa.Column('model_route_name', sa.String(255), nullable=True),
            # Tenant scope snapshot. Plain integer (no FK) for the same
            # audit-survival reasons as the other id columns on this table:
            # ``ON DELETE SET NULL`` on the live Principal would erase the
            # historical owner attribution that breakdown / billing relies
            # on. Sourced at flush time from the model.
            sa.Column('owner_principal_id', sa.Integer(), nullable=True),
            # Consumer tenant scope snapshot, denormalized from the API key
            # owner. This can differ from owner_principal_id for cross-Org
            # shared model usage.
            sa.Column('consumer_principal_id', sa.Integer(), nullable=True),
            sa.Column('provider_id', sa.Integer(), nullable=True),
            sa.Column('provider_name', sa.String(255), nullable=True),
            sa.Column('provider_type', sa.String(255), nullable=True),
            sa.Column('cluster_id', sa.Integer(), nullable=True),
            sa.Column('cluster_name', sa.String(255), nullable=True),
            sa.Column('api_key_id', sa.Integer(), nullable=True),
            sa.Column('api_key_name', sa.String(255), nullable=True),
            sa.Column('access_key', sa.String(255), nullable=True),
            sa.Column('api_key_is_custom', sa.Boolean(), nullable=True),
            sa.Column('date', sa.Date(), nullable=False),
            sa.Column('prompt_token_count', sa.BigInteger(), nullable=False),
            sa.Column('completion_token_count', sa.BigInteger(), nullable=False),
            sa.Column(
                'prompt_cached_token_count',
                sa.BigInteger(),
                nullable=False,
                server_default='0',
            ),
            sa.Column('operation', sa.String(32), nullable=True),
            # Proxy-reported wall-clock (naive UTC), distinct from created_at
            # so reconciliation jobs anchor on request semantics, not row
            # write time.
            sa.Column('started_at', UTCDateTime(), nullable=True),
            sa.Column('completed_at', UTCDateTime(), nullable=True),
            sa.Column('created_at', UTCDateTime(), nullable=False),
            sa.Column('updated_at', UTCDateTime(), nullable=False),
            sa.Column('deleted_at', UTCDateTime(), nullable=True),
        ]

    def _create_details_indexes(table_name: str) -> None:
        op.create_index(
            f'ix_{table_name}_date', table_name, ['date'], unique=False
        )
        op.create_index(
            f'ix_{table_name}_model_id', table_name, ['model_id'], unique=False
        )
        op.create_index(
            f'ix_{table_name}_user_id', table_name, ['user_id'], unique=False
        )
        op.create_index(
            f'ix_{table_name}_consumer_principal_id',
            table_name,
            ['consumer_principal_id'],
            unique=False,
        )
        op.create_index(
            f'ix_{table_name}_api_key_id',
            table_name,
            ['api_key_id'],
            unique=False,
        )
        op.create_index(
            f'ix_{table_name}_model_route_id',
            table_name,
            ['model_route_id'],
            unique=False,
        )
        # completed_at drives quota-reconciliation range scans.
        op.create_index(
            f'ix_{table_name}_completed_at',
            table_name,
            ['completed_at'],
            unique=False,
        )
        # created_at backs the archiver's fallback predicate for legacy rows
        # whose completed_at is NULL — without this index, that branch of
        # the BitmapOr would degrade to a seq scan as the table grows.
        op.create_index(
            f'ix_{table_name}_created_at',
            table_name,
            ['created_at'],
            unique=False,
        )

    # Hot table — no FKs by design (audit/billing rows must keep the
    # original entity ids even after the parents are deleted).
    if not table_exists('model_usage_details'):
        op.create_table(
            'model_usage_details',
            *_details_columns(),
            sa.PrimaryKeyConstraint('id'),
        )
        _create_details_indexes('model_usage_details')

    # Cold archive — same FK-less layout as the hot table, plus no sequence
    # on id (archival reuses the source id from model_usage_details) to
    # maximize bulk-insert throughput.
    if not table_exists('model_usage_details_archive'):
        archive_columns = _details_columns()
        archive_columns[0] = sa.Column(
            'id', sa.Integer(), nullable=False, autoincrement=False
        )
        op.create_table(
            'model_usage_details_archive',
            *archive_columns,
            sa.PrimaryKeyConstraint('id'),
        )
        _create_details_indexes('model_usage_details_archive')

    ### LoRA
    _upgrade_lora()
    ### end

    ### inference_backends parameter_format & common_parameters
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        if not column_exists('inference_backends', 'parameter_format'):
            batch_op.add_column(
                sa.Column('parameter_format', sa.String(length=255), nullable=True)
            )
        if not column_exists('inference_backends', 'common_parameters'):
            batch_op.add_column(
                sa.Column('common_parameters', sa.JSON(), nullable=True)
            )
    ### end



def _upgrade_lora() -> None:
    with op.batch_alter_table("model_files", schema=None) as batch_op:
        if not column_exists("model_files", "is_lora"):
            batch_op.add_column(
                sa.Column(
                    "is_lora",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )
        if not column_exists("model_files", "base_model"):
            batch_op.add_column(
                sa.Column("base_model", sa.String(length=512), nullable=True)
            )

    with op.batch_alter_table("models", schema=None) as batch_op:
        if not column_exists("models", "lora_list"):
            batch_op.add_column(
                sa.Column("lora_list", gpustack.schemas.common.JSON(), nullable=True)
            )

    with op.batch_alter_table("model_instances", schema=None) as batch_op:
        if not column_exists("model_instances", "mounted_loras"):
            batch_op.add_column(
                sa.Column(
                    "mounted_loras", gpustack.schemas.common.JSON(), nullable=True
                )
            )

    with op.batch_alter_table("model_routes", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("created_model_id", sa.Integer(), nullable=True)
        )

    op.execute(
        """
        UPDATE model_routes
        SET created_model_id = (
            SELECT mrt.model_id FROM model_route_targets mrt
            WHERE mrt.route_id = model_routes.id
              AND mrt.model_id IS NOT NULL
            LIMIT 1
        )
        WHERE created_by_model = true
        """
    )

    # View depends on model_routes.*; drop it so DROP COLUMN below
    # isn't blocked on PG. init_db re-creates it at next server start.
    op.execute(model_user_after_drop_view_stmt)

    with op.batch_alter_table("model_routes", schema=None) as batch_op:
        batch_op.drop_column("created_by_model")

    op.create_index(
        "ix_model_routes_created_model_id_name",
        "model_routes",
        ["created_model_id", "name"],
        unique=False,
    )

    with op.batch_alter_table("model_route_targets", schema=None) as batch_op:
        batch_op.alter_column(
            "provider_model_name",
            new_column_name="overridden_model_name",
            existing_type=sa.String(length=512),
            existing_nullable=True,
        )


def _downgrade_lora() -> None:
    with op.batch_alter_table("model_route_targets", schema=None) as batch_op:
        batch_op.alter_column(
            "overridden_model_name",
            new_column_name="provider_model_name",
            existing_type=sa.String(length=512),
            existing_nullable=True,
        )

    op.drop_index("ix_model_routes_created_model_id_name", table_name="model_routes")

    # Same dependency dance as upgrade — DROP COLUMN created_model_id
    # below is blocked while non_admin_user_models references it.
    op.execute(model_user_after_drop_view_stmt)

    with op.batch_alter_table("model_routes", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "created_by_model",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )

    op.execute(
        "UPDATE model_routes SET created_by_model = true WHERE created_model_id IS NOT NULL"
    )

    with op.batch_alter_table("model_routes", schema=None) as batch_op:
        batch_op.drop_column("created_model_id")

    with op.batch_alter_table("model_instances", schema=None) as batch_op:
        batch_op.drop_column("mounted_loras")

    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_column("lora_list")

    with op.batch_alter_table("model_files", schema=None) as batch_op:
        batch_op.drop_column("base_model")
        batch_op.drop_column("is_lora")


def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('worker_version')
        batch_op.drop_column('proxy_address')

    sql_enum.remove_enum_values(
        {'workers': ('proxy_mode', 'WORKER')},
        model_instance_proxy_mode,
        *proxy_mode_to_add,
    )

    ### k8s volume mount
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.drop_column('k8s_volume_mounts')
    ### end

    ### custom API_KEY
    op.execute("DELETE FROM api_keys WHERE is_custom = true")
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('is_custom')
        batch_op.drop_column('scope')
    ### end
    
    ### Usage
    with op.batch_alter_table("model_usages", schema=None) as batch_op:
        batch_op.drop_index("ix_model_usages_model_route_id")
        # Drop FKs added by the upgrade (api_key_id / model_route_id are net
        # new; user/model/provider were swapped from CASCADE to SET NULL).
        batch_op.drop_constraint(
            "fk_model_usages_model_route_id_model_routes", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_model_usages_api_key_id_api_keys", type_="foreignkey"
        )
        batch_op.drop_constraint("fk_model_usages_user_id_users", type_="foreignkey")
        batch_op.drop_constraint("fk_model_usages_model_id_models", type_="foreignkey")
        batch_op.drop_constraint(
            "fk_model_usages_provider_id_model_providers", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_model_usages_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_model_id_models",
            "models",
            ["model_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_model_usages_provider_id_model_providers",
            "model_providers",
            ["provider_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.drop_column("model_route_name")
        batch_op.drop_column("model_route_id")
        batch_op.drop_column("api_key_is_custom")
        batch_op.drop_column("provider_type")
        batch_op.drop_column("provider_name")
        batch_op.drop_column("api_key_name")
        batch_op.drop_column("api_key_id")
        batch_op.drop_column("user_name")
        batch_op.drop_column("cluster_name")
        batch_op.drop_column("prompt_cached_token_count")
    ### end
    
    ### Remove injected_backend_parameters column from model_instances
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('injected_backend_parameters')
    ### end

    ### Drop per-request usage details tables
    for details_table in ("model_usage_details_archive", "model_usage_details"):
        if not table_exists(details_table):
            continue
        op.drop_index(f'ix_{details_table}_created_at', table_name=details_table)
        op.drop_index(f'ix_{details_table}_completed_at', table_name=details_table)
        op.drop_index(f'ix_{details_table}_model_route_id', table_name=details_table)
        op.drop_index(f'ix_{details_table}_api_key_id', table_name=details_table)
        op.drop_index(f'ix_{details_table}_user_id', table_name=details_table)
        op.drop_index(f'ix_{details_table}_model_id', table_name=details_table)
        op.drop_index(f'ix_{details_table}_date', table_name=details_table)
        op.drop_table(details_table)

    ### LoRA
    _downgrade_lora()
    ### end

    ### Remove inference_backends parameter_format & common_parameters
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.drop_column('common_parameters')
        batch_op.drop_column('parameter_format')
    ### end
