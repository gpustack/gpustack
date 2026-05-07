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
    ### end



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
        # Drop FKs added by the upgrade (api_key_id is net new;
        # user/model/provider were swapped from CASCADE to SET NULL).
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
    ### end
