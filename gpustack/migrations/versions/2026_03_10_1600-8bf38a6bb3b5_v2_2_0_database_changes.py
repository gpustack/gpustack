"""v2.2.0 add worker version

Revision ID: 8bf38a6bb3b5
Revises: 8ad0f94c92e8
Create Date: 2026-03-10 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import gpustack
from gpustack.migrations.utils import column_exists
import gpustack.utils.sql_enum as sql_enum

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
        if not column_exists("model_usages", "api_key_is_custom"):
            batch_op.add_column(
                sa.Column("api_key_is_custom", sa.Boolean(), nullable=True)
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
        batch_op.drop_column("api_key_name")
        batch_op.drop_column("api_key_id")
        batch_op.drop_column("user_name")
        batch_op.drop_column("cluster_name")
    ### end
