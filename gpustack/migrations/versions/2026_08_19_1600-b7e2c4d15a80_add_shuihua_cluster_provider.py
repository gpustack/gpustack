"""add Shuihua cluster provider

Adds the ``Shuihua`` value to the ``clusterprovider`` enum so cloud credentials,
clusters and workers can be created against the Shuihua open API.

Revision ID: b7e2c4d15a80
Revises: d5e8f0a1b2c3
Create Date: 2026-08-19 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack.utils.sql_enum as sql_enum

# revision identifiers, used by Alembic.
revision: str = 'b7e2c4d15a80'
down_revision: Union[str, None] = 'd5e8f0a1b2c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

cluster_provider_enum = sa.Enum(
    'Docker',
    'Kubernetes',
    'DigitalOcean',
    name='clusterprovider',
)

cluster_provider_to_add = ['Shuihua']

# Every table carrying the enum. PostgreSQL alters the shared type once and
# ignores this; MySQL inlines the enum per column, so each one is rewritten.
cluster_provider_table_columns = {
    'cloud_credentials': 'provider',
    'clusters': 'provider',
    'workers': 'provider',
}


def upgrade() -> None:
    sql_enum.add_enum_values(
        cluster_provider_table_columns,
        cluster_provider_enum,
        *cluster_provider_to_add,
    )


def downgrade() -> None:
    conn = op.get_bind()

    # Move the data off Shuihua first. This is the load-bearing half: the older
    # code's ClusterProvider has no Shuihua member, so any row still carrying it
    # raises LookupError on load, while an unused label left in the DB type
    # would be harmless. Relabelling to Docker is what
    # sql_enum.remove_enum_values does for the single-column case.
    #
    # The cloud instances behind these workers are NOT reclaimed: a migration
    # cannot call the Shuihua API, so the VMs leak either way. Relabelling keeps
    # the rows (and their external_id) around so they can still be found.
    conn.execute(
        sa.text(
            "UPDATE clusters SET provider = 'Docker', credential_id = NULL "
            "WHERE provider = 'Shuihua'"
        )
    )
    conn.execute(
        sa.text("UPDATE workers SET provider = 'Docker' WHERE provider = 'Shuihua'")
    )
    # A cloud credential has no meaning under Docker — routes/cloud_credentials
    # rejects that provider outright — so these are deleted rather than
    # relabelled. The UPDATE above cleared clusters.credential_id, the only FK
    # pointing at them.
    conn.execute(sa.text("DELETE FROM cloud_credentials WHERE provider = 'Shuihua'"))

    # Then narrow the enum, so that re-running upgrade() afterwards still
    # works: add_enum_values issues a bare ADD VALUE on PostgreSQL, which
    # errors if the label is already there.
    if conn.dialect.name == 'postgresql':
        # Recreate the type and convert all three columns in one pass, rather
        # than sql_enum.remove_enum_values: that recreates per (table, column),
        # so its first DROP TYPE would fail while the other two columns still
        # reference clusterprovider. Same shape as
        # _consolidate_access_policy_enum in the multi-tenancy migration, minus
        # the DROP/SET DEFAULT dance — these columns were created with a
        # Python-side default only, so there is no server DEFAULT to restore.
        conn.execute(
            sa.text(
                "CREATE TYPE clusterprovidertmp AS ENUM "
                "('Docker', 'Kubernetes', 'DigitalOcean')"
            )
        )
        for table, column in cluster_provider_table_columns.items():
            conn.execute(
                sa.text(
                    f"ALTER TABLE {table} ALTER COLUMN {column} TYPE "
                    f"clusterprovidertmp USING {column}::text::clusterprovidertmp"
                )
            )
        conn.execute(sa.text("DROP TYPE clusterprovider"))
        conn.execute(sa.text("ALTER TYPE clusterprovidertmp RENAME TO clusterprovider"))
    elif conn.dialect.name == 'mysql':
        for table, column in cluster_provider_table_columns.items():
            sql_enum.modify_mysql_table_column_enum(
                conn, table, column, [], cluster_provider_to_add
            )
