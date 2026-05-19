"""multi-tenancy foundation

Sets up the entire tenancy storage layer in one upgrade. Built around a
single ``principals`` table that unifies USER / ORG / GROUP under one
identity. Resource ownership (``owner_principal_id``), ACL grants, and
memberships all reference principals directly — no polymorphic
``(type, id)`` pairs.

Logical groups, run in order:

1. New tables — principals / principal_memberships / cluster_access /
   model_route_principals.
2. Seed the platform Org-principal (id=1, kind=ORG, slug=`default`,
   name=`Default`).
3. Insert one USER-principal per existing user (slug=`user-{id}`,
   name=username).
4. Add `users.principal_id` (NOT NULL UNIQUE FK → principals.id),
   backfill from step 3.
5. Backfill memberships in the platform Org for admin users (role=OWNER).
   Regular users get no auto-membership — admins must add them
   explicitly when team-workspace access is wanted.
6. Add `owner_principal_id` to existing tenant-scoped tables: api_keys
   / models / model_instances / model_routes; backfill to the platform
   Org-principal and promote to NOT NULL.
7. BYO cluster: `clusters` / `cloud_credentials` / `worker_pools` get
   `owner_principal_id` (NOT NULL after backfill, ON DELETE CASCADE);
   per-Org default cluster expressed via partial unique index.
8. Cluster-derived denormalized columns: workers / model_files /
   benchmarks / model_providers / model_usages each get
   `owner_principal_id` (nullable; SET NULL on principal delete).
   model_files gains `cluster_id` for cluster_access-based filtering.
9. Inference backends Hybrid: `owner_principal_id` (nullable = Platform
   row), composite unique on (backend_name, owner_principal_id).
10. Backfill `model_route_principals` from the legacy
    `usermodelroutelink` table so ALLOWED_USERS-published routes remain
    visible through the unified ACL table.
11. Extend `accesspolicyenum` with ALLOWED_PRINCIPALS and ORG.
    `non_admin_user_models` and `gpu_devices_view` are recreated by
    `init_db.listen_events` at server startup, not here.
12. Drop the legacy `usermodelroutelink` table.

Revision ID: 7c5e3f9a2d18
Revises: 8bf38a6bb3b5
Create Date: 2026-04-28 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack  # noqa: F401  (keeps SQLModel registrations side-effect-loaded)
import gpustack.utils.sql_enum as sql_enum
from gpustack.migrations.utils import column_exists, table_exists
from gpustack.schemas.principals import (
    PLATFORM_PRINCIPAL_ID,
    PLATFORM_PRINCIPAL_SLUG,
)
from gpustack.schemas.stmt import (
    model_user_after_drop_view_stmt,
    principal_users_after_drop_view_stmt,
)
from gpustack.schemas.users import AuthProviderEnum


# revision identifiers, used by Alembic.
revision: str = '7c5e3f9a2d18'
down_revision: Union[str, None] = '8bf38a6bb3b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PLATFORM_PRINCIPAL_NAME = 'Default'


def _enums():
    """Enums *owned* by this migration — created by it, dropped on
    downgrade. ``authproviderenum`` is *not* in this set; it predates
    this migration (created by the initial-tables migration for
    ``users.source``) and is referenced inline at its single use
    site via ``_existing_auth_provider_enum``.
    """
    org_role = sa.Enum('OWNER', 'MEMBER', name='orgrole')
    principal_type = sa.Enum('ORG', 'GROUP', 'USER', name='principaltype')
    return org_role, principal_type


def _existing_auth_provider_enum(bind):
    """Reference (not declare) the ``authproviderenum`` type that
    the initial-tables migration already created for ``users.source``.
    On Postgres the dialect-specific ``postgresql.ENUM`` with
    ``create_type=False`` suppresses the duplicate ``CREATE TYPE``
    (the generic ``sa.Enum`` silently swallows ``create_type`` via
    ``**kw``, which is why we branch). On MySQL ``sa.Enum`` renders
    as the native ``ENUM(...)`` column type, which doesn't need
    pre-declaration.
    """
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import ENUM as PGEnum

        return PGEnum(
            AuthProviderEnum, name='authproviderenum', create_type=False
        )
    return sa.Enum(AuthProviderEnum, name='authproviderenum')


def upgrade() -> None:
    bind = op.get_bind()
    org_role, principal_type = _enums()

    # On postgres, enum types are created lazily by the first column
    # referencing them; subsequent columns reuse the existing type since
    # SQLAlchemy tracks the same enum instance.

    # ---- 1. New tables ---------------------------------------------------

    if not table_exists('principals'):
        op.create_table(
            'principals',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('kind', principal_type, nullable=False),
            # Slug is globally unique among non-NULL values. USER
            # principals get auto-assigned ``user-{user.id}``; ORG
            # principals carry a user-supplied slug; GROUP principals
            # have NULL slug (groups never appear in URL prefixes).
            sa.Column('slug', sa.String(length=255), nullable=True),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            # Origin of this principal — ``Local`` for admin-managed
            # rows, ``OIDC`` / ``SAML`` for Groups auto-created by
            # IdP sync. Lets the UI badge IdP-managed Groups
            # distinctly.
            sa.Column(
                'source',
                _existing_auth_provider_enum(bind),
                nullable=False,
                server_default='Local',
            ),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('slug', name='uix_principals_slug'),
        )
        # Group names are globally unique among active groups. USER /
        # ORG names are not constrained here (Users key off
        # ``users.username``, Orgs off ``slug``). Groups are
        # peer-level principals in this schema — Org affiliation, if
        # any, is expressed via a row in ``principal_memberships``
        # with ``parent=Org, member=Group``.
        #
        # Postgres supports partial unique indexes natively, so the
        # constraint is enforced at the DB. MySQL has no equivalent —
        # a plain (non-unique) index supports the by-name lookups,
        # and uniqueness is enforced at the application layer (the
        # ``/groups`` admin POST and ``sync_user_group_memberships``
        # both pre-check + handle ``IntegrityError`` retries).
        #
        # PG-/MySQL-compatible dialects (openGauss, OceanBase, TiDB,
        # MariaDB, ...) may report their own ``dialect.name``; we
        # only take the partial-index path when the dialect reports
        # the canonical ``postgresql`` name, otherwise we fall back
        # to the MySQL-compatible plain-index path. That keeps the
        # foundation portable across the wire-protocol-compatible
        # databases the deployment supports without an explicit
        # allowlist that we'd have to keep extending.
        if bind.dialect.name == 'postgresql':
            op.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uix_principals_group_name "
                "ON principals (name) "
                "WHERE kind = 'GROUP' AND deleted_at IS NULL"
            )
        else:
            op.create_index(
                'ix_principals_group_name',
                'principals',
                ['name'],
            )

    if not table_exists('principal_memberships'):
        # Surrogate ``id`` PK so soft-deleted rows can coexist with
        # re-adds. The "at most one active row per (parent, member)"
        # invariant is enforced at the application layer.
        op.create_table(
            'principal_memberships',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('parent_principal_id', sa.Integer(), nullable=False),
            sa.Column('member_principal_id', sa.Integer(), nullable=False),
            # NULL for GROUP memberships (no role tiers); OWNER / MEMBER
            # for ORG memberships.
            sa.Column('role', org_role, nullable=True),
            # Where this membership row originated. ``Local`` for rows
            # the admin (or a route handler) created; ``OIDC`` /
            # ``SAML`` for rows written by IdP group-sync. Sync logic
            # only ever rewrites rows where source matches the current
            # provider — admin-added memberships are untouched.
            sa.Column(
                'source',
                _existing_auth_provider_enum(bind),
                nullable=False,
                server_default='Local',
            ),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['parent_principal_id'], ['principals.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['member_principal_id'], ['principals.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(
            'ix_principal_memberships_parent_member',
            'principal_memberships',
            ['parent_principal_id', 'member_principal_id'],
        )
        # Reverse direction: ``non_admin_user_models`` and the ORG-scope
        # access checks join from a member's user-principal back to its
        # parent orgs/groups. Without this index those joins fall back
        # to a table scan on every page load.
        op.create_index(
            'ix_principal_memberships_member_parent',
            'principal_memberships',
            ['member_principal_id', 'parent_principal_id'],
        )

    if not table_exists('cluster_access'):
        op.create_table(
            'cluster_access',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('principal_id', sa.Integer(), nullable=False),
            sa.Column('granted_by', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['cluster_id'], ['clusters.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['principal_id'], ['principals.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['granted_by'], ['users.id'], ondelete='SET NULL',
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint(
                'cluster_id', 'principal_id',
                name='uix_cluster_access_cluster_principal',
            ),
        )

    if not table_exists('model_route_principals'):
        # Per-route ACL grants for ALLOWED_USERS / ALLOWED_PRINCIPALS.
        # Single ``principal_id`` FK — kind is read from the joined
        # principals row at evaluation time.
        op.create_table(
            'model_route_principals',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('route_id', sa.Integer(), nullable=False),
            sa.Column('principal_id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['route_id'], ['model_routes.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['principal_id'], ['principals.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint(
                'route_id', 'principal_id', name='uix_route_principal'
            ),
        )
        op.create_index(
            'ix_model_route_principals_route_id',
            'model_route_principals',
            ['route_id'],
        )

    # ---- 2. Seed platform Org-principal ---------------------------------

    # Existence check is by slug, not id — slug is the canonical
    # identifier for built-in principals across environments. The id
    # we insert is the seed value (``PLATFORM_PRINCIPAL_ID = 1``);
    # if we ever change that contract, only this INSERT cares.
    op.execute(
        sa.text(
            """
            INSERT INTO principals
                (id, kind, slug, name, description,
                 created_at, updated_at, deleted_at)
            SELECT :id, :kind, :slug, :name, :desc,
                   CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM principals WHERE slug = :slug
            )
            """
        ).bindparams(
            id=PLATFORM_PRINCIPAL_ID,
            kind='ORG',
            slug=PLATFORM_PRINCIPAL_SLUG,
            name=PLATFORM_PRINCIPAL_NAME,
            desc='Built-in platform organization',
        )
    )

    # Realign sequence on postgres so the next inserted principal doesn't collide.
    if bind.dialect.name == 'postgresql':
        op.execute(
            "SELECT setval(pg_get_serial_sequence('principals', 'id'), "
            "GREATEST((SELECT MAX(id) FROM principals), 1))"
        )

    # Resolve the platform principal's actual id from slug. Pre-existing
    # rows from older bootstraps may not be at ``PLATFORM_PRINCIPAL_ID``
    # (e.g. if seeded against a populated DB with auto-increment). Every
    # downstream backfill uses this resolved value rather than the seed
    # constant.
    platform_id = bind.execute(
        sa.text("SELECT id FROM principals WHERE slug = :slug").bindparams(
            slug=PLATFORM_PRINCIPAL_SLUG
        )
    ).scalar()
    if platform_id is None:
        raise RuntimeError(
            f"Platform principal not found by slug={PLATFORM_PRINCIPAL_SLUG!r} "
            "after seed; backfill cannot continue"
        )

    # ---- 3. Backfill USER-principals ------------------------------------

    # One USER-principal per existing user (system users included so
    # ``users.principal_id`` is uniformly NOT NULL). Slug is the
    # canonical ``user-{id}`` form; name copies ``username``.
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO principals
                    (kind, slug, name,
                     created_at, updated_at, deleted_at)
                SELECT 'USER'::principaltype, 'user-' || u.id, u.username,
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
                FROM users u
                WHERE NOT EXISTS (
                    SELECT 1 FROM principals p
                    WHERE p.kind = 'USER'::principaltype
                      AND p.slug = 'user-' || u.id
                )
                """
            )
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT INTO principals
                    (kind, slug, name,
                     created_at, updated_at, deleted_at)
                SELECT 'USER', 'user-' || u.id, u.username,
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
                FROM users u
                WHERE NOT EXISTS (
                    SELECT 1 FROM principals p
                    WHERE p.kind = 'USER'
                      AND p.slug = 'user-' || u.id
                )
                """
            )
        )

    # ---- 4. users.principal_id ------------------------------------------

    if not column_exists('users', 'principal_id'):
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('principal_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_users_principal_id_principals',
                'principals',
                ['principal_id'],
                ['id'],
                ondelete='RESTRICT',
            )

    # Backfill from the matching USER-principal seeded in step 3.
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                UPDATE users u
                SET principal_id = p.id
                FROM principals p
                WHERE p.kind = 'USER'::principaltype
                  AND p.slug = 'user-' || u.id
                  AND u.principal_id IS NULL
                """
            )
        )
    else:
        op.execute(
            sa.text(
                """
                UPDATE users
                SET principal_id = (
                    SELECT p.id FROM principals p
                    WHERE p.kind = 'USER' AND p.slug = 'user-' || users.id
                )
                WHERE principal_id IS NULL
                """
            )
        )

    # Validate every user got a principal_id; loud failure beats a silent
    # NOT NULL violation later.
    unfilled = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM users WHERE principal_id IS NULL"
        )
    ).scalar()
    if unfilled and int(unfilled) > 0:
        raise RuntimeError(
            f"{unfilled} user(s) missing principal_id after backfill"
        )

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column(
            'principal_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_unique_constraint(
            'uix_users_principal_id', ['principal_id']
        )

    # ---- 5. Backfill admin memberships in the platform Org --------------

    # Only admin users get auto-enrolled in the platform Org. Regular
    # users start with no team memberships — admins must add them
    # explicitly when team-workspace access is wanted. (System users
    # also skipped — workers / cluster service accounts aren't
    # tenants.)
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO principal_memberships
                    (parent_principal_id, member_principal_id, role,
                     created_at, updated_at, deleted_at)
                SELECT :platform_id, u.principal_id, 'OWNER'::orgrole,
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
                FROM users u
                WHERE u.is_admin = true
                  AND COALESCE(u.is_system, false) = false
                  AND NOT EXISTS (
                    SELECT 1 FROM principal_memberships m
                    WHERE m.parent_principal_id = :platform_id
                      AND m.member_principal_id = u.principal_id
                      AND m.deleted_at IS NULL
                  )
                """
            ).bindparams(platform_id=platform_id)
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT INTO principal_memberships
                    (parent_principal_id, member_principal_id, role,
                     created_at, updated_at, deleted_at)
                SELECT :platform_id, u.principal_id, 'OWNER',
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
                FROM users u
                WHERE u.is_admin = 1
                  AND COALESCE(u.is_system, 0) = 0
                  AND NOT EXISTS (
                    SELECT 1 FROM principal_memberships m
                    WHERE m.parent_principal_id = :platform_id
                      AND m.member_principal_id = u.principal_id
                      AND m.deleted_at IS NULL
                  )
                """
            ).bindparams(platform_id=platform_id)
        )

    # ---- 6. api_keys.owner_principal_id ---------------------------------

    if not column_exists('api_keys', 'owner_principal_id'):
        with op.batch_alter_table('api_keys', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

    # api_keys are user-owned (each row carries ``user_id``). Stamp
    # ``owner_principal_id`` to the matching user's USER-principal so
    # the resolver's per-user list filter keeps the key visible to its
    # owner post-upgrade. Falling back to ``PLATFORM_PRINCIPAL_ID``
    # only for orphaned rows where the user is gone or has no
    # principal yet.
    op.execute(
        sa.text(
            """
            UPDATE api_keys
            SET owner_principal_id = u.principal_id
            FROM users u
            WHERE api_keys.user_id = u.id
              AND api_keys.owner_principal_id IS NULL
              AND u.principal_id IS NOT NULL
            """
        )
        if bind.dialect.name == 'postgresql'
        else sa.text(
            """
            UPDATE api_keys
            SET owner_principal_id = (
                SELECT u.principal_id FROM users u WHERE u.id = api_keys.user_id
            )
            WHERE owner_principal_id IS NULL
              AND EXISTS (
                SELECT 1 FROM users u
                WHERE u.id = api_keys.user_id AND u.principal_id IS NOT NULL
              )
            """
        )
    )
    op.execute(
        sa.text(
            "UPDATE api_keys SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id', existing_type=sa.Integer(), nullable=False
        )
        # Unique scope was (user_id, name); add owner_principal_id so the
        # same key name can coexist across owner namespaces.
        try:
            batch_op.drop_constraint('uix_user_id_name', type_='unique')
        except Exception:
            pass
        batch_op.create_unique_constraint(
            'uix_user_owner_name', ['user_id', 'owner_principal_id', 'name']
        )
        batch_op.create_foreign_key(
            'fk_api_keys_owner_principal_id_principals',
            'principals',
            ['owner_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- 7. models.owner_principal_id -----------------------------------

    if not column_exists('models', 'owner_principal_id'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE models SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_models_owner_principal_id_principals',
            'principals',
            ['owner_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- 8. model_instances.owner_principal_id --------------------------

    if not column_exists('model_instances', 'owner_principal_id'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE model_instances SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_model_instances_owner_principal_id_principals',
            'principals',
            ['owner_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- 9. model_routes.owner_principal_id -----------------------------

    if not column_exists('model_routes', 'owner_principal_id'):
        with op.batch_alter_table('model_routes', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE model_routes SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )

    with op.batch_alter_table('model_routes', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_model_routes_owner_principal_id_principals',
            'principals',
            ['owner_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- 10. BYO cluster: clusters / cloud_credentials / worker_pools ---

    # Clusters / cloud credentials / worker_pools always belong to a
    # principal. Cross-principal sharing goes through cluster_access.
    # ON DELETE CASCADE: deleting the owner principal takes its infra
    # rows with it.
    if not column_exists("clusters", "owner_principal_id"):
        with op.batch_alter_table("clusters", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("owner_principal_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_clusters_owner_principal_id_principals",
                "principals",
                ["owner_principal_id"],
                ["id"],
                ondelete="CASCADE",
            )

    if not column_exists("cloud_credentials", "owner_principal_id"):
        with op.batch_alter_table("cloud_credentials", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("owner_principal_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_cloud_credentials_owner_principal_id_principals",
                "principals",
                ["owner_principal_id"],
                ["id"],
                ondelete="CASCADE",
            )

    if not column_exists("worker_pools", "owner_principal_id"):
        with op.batch_alter_table("worker_pools", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("owner_principal_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_worker_pools_owner_principal_id_principals",
                "principals",
                ["owner_principal_id"],
                ["id"],
                ondelete="CASCADE",
            )

    # Backfill any pre-existing rows with NULL owner → platform principal
    # so the NOT NULL constraint below holds (admin's existing infra
    # lands in the Default Org as expected).
    op.execute(
        sa.text(
            "UPDATE clusters SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )
    op.execute(
        sa.text(
            "UPDATE cloud_credentials SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )
    op.execute(
        sa.text(
            "UPDATE worker_pools SET owner_principal_id = :pid "
            "WHERE owner_principal_id IS NULL"
        ).bindparams(pid=platform_id)
    )

    with op.batch_alter_table("clusters", schema=None) as batch_op:
        batch_op.alter_column(
            "owner_principal_id", existing_type=sa.Integer(), nullable=False
        )
    with op.batch_alter_table("cloud_credentials", schema=None) as batch_op:
        batch_op.alter_column(
            "owner_principal_id", existing_type=sa.Integer(), nullable=False
        )
    with op.batch_alter_table("worker_pools", schema=None) as batch_op:
        batch_op.alter_column(
            "owner_principal_id", existing_type=sa.Integer(), nullable=False
        )

    # At most one default cluster per principal. Partial unique covers
    # active rows only (excluding soft-deleted), letting a principal
    # "rotate" defaults by soft-deleting the old + flipping the new
    # without conflict.
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_clusters_default_per_owner "
            "ON clusters (owner_principal_id) "
            "WHERE is_default = true AND deleted_at IS NULL"
        )
    else:
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_clusters_default_per_owner "
            "ON clusters (owner_principal_id) "
            "WHERE is_default = 1 AND deleted_at IS NULL"
        )

    # ---- 11. Cluster-derived denormalized owner_principal_id ------------

    # Workers, model_files, benchmarks, model_providers, model_usages all
    # need an owner pointer for per-row filtering. Nullable (NULL = on a
    # global cluster, admin-managed); ON DELETE SET NULL keeps rows alive
    # when the owner principal is deleted (principal delete cascades
    # clusters, which cascade their workers anyway).
    for tbl in (
        "workers",
        "model_files",
        "benchmarks",
        "model_providers",
        "model_usages",
    ):
        if not column_exists(tbl, "owner_principal_id"):
            with op.batch_alter_table(tbl, schema=None) as batch_op:
                batch_op.add_column(
                    sa.Column("owner_principal_id", sa.Integer(), nullable=True)
                )
                batch_op.create_foreign_key(
                    f"fk_{tbl}_owner_principal_id_principals",
                    "principals",
                    ["owner_principal_id"],
                    ["id"],
                    ondelete="SET NULL",
                )

    # model_files only had worker_id; add cluster_id for direct
    # cluster_access-based filtering.
    if not column_exists("model_files", "cluster_id"):
        with op.batch_alter_table("model_files", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("cluster_id", sa.Integer(), nullable=True)
            )

    # ---- 12. Inference backends Hybrid ----------------------------------

    # NULL owner_principal_id = Platform-managed (admin curates built-ins);
    # non-NULL = an Org's extension/override. backend_name is no longer
    # globally unique — composite unique on (backend_name, owner_principal_id)
    # lets each owner carry their own row alongside the Platform row.
    if not column_exists("inference_backends", "owner_principal_id"):
        with op.batch_alter_table("inference_backends", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("owner_principal_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_inference_backends_owner_principal_id_principals",
                "principals",
                ["owner_principal_id"],
                ["id"],
                ondelete="CASCADE",
            )
            try:
                batch_op.drop_constraint(
                    "inference_backends_backend_name_key", type_="unique"
                )
            except Exception:
                pass
            try:
                batch_op.drop_index("ix_inference_backends_backend_name")
            except Exception:
                pass
            batch_op.create_unique_constraint(
                "uix_inference_backends_name_owner",
                ["backend_name", "owner_principal_id"],
            )
            batch_op.create_index(
                "ix_inference_backends_backend_name", ["backend_name"]
            )

    # ---- 13. Backfill model_route_principals from usermodelroutelink ----

    user_link_table = (
        'usermodelroutelink' if table_exists('usermodelroutelink') else None
    )
    if user_link_table:
        if bind.dialect.name == 'postgresql':
            op.execute(
                sa.text(
                    f"""
                    INSERT INTO model_route_principals
                        (route_id, principal_id, created_at, updated_at)
                    SELECT uml.route_id, u.principal_id,
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    FROM {user_link_table} uml
                    JOIN users u ON u.id = uml.user_id
                    WHERE uml.route_id IS NOT NULL AND uml.user_id IS NOT NULL
                    ON CONFLICT (route_id, principal_id) DO NOTHING
                    """
                )
            )
        else:
            op.execute(
                sa.text(
                    f"""
                    INSERT OR IGNORE INTO model_route_principals
                        (route_id, principal_id, created_at, updated_at)
                    SELECT uml.route_id, u.principal_id,
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    FROM {user_link_table} uml
                    JOIN users u ON u.id = uml.user_id
                    WHERE uml.route_id IS NOT NULL AND uml.user_id IS NOT NULL
                    """
                )
            )

    # ---- 14. Extend access_policy enum ----------------------------------

    # ORG = scoped to members of the route's owning Organization (default
    # for non-platform Org routes). ALLOWED_PRINCIPALS = explicit per-user
    # / group / org grants via model_route_principals. ALLOWED_USERS
    # stays as the OSS-facing per-user-only policy; rows are stored
    # alongside ALLOWED_PRINCIPALS rows in the unified principals table.
    # `non_admin_user_models` and `gpu_devices_view` are recreated by
    # `init_db.listen_events` after this migration completes, so they
    # don't need to be touched here.
    access_policy_enum = sa.Enum(
        'PUBLIC', 'AUTHED', 'ALLOWED_USERS', name='accesspolicyenum'
    )
    sql_enum.add_enum_values(
        {'model_routes': 'access_policy'},
        access_policy_enum,
        'ALLOWED_PRINCIPALS',
        'ORG',
    )

    # ---- 15. credentials.owner_principal_id + PASSWORD backfill ---------
    #
    # v2.2.0 already reshaped the ``credentials`` columns (rename +
    # nullable + PASSWORD enum value). Here we ground the rows: add the
    # FK back to ``principals`` and turn each user's bcrypt hash into a
    # PASSWORD-typed credential row owned by that user's principal.
    #
    # System users (cluster / worker) carry an empty hash — skip those:
    # a credential with no secret is meaningless and would only pollute
    # the password-lookup hot path.

    if not column_exists('credentials', 'owner_principal_id'):
        with op.batch_alter_table('credentials', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True),
            )
            batch_op.create_foreign_key(
                'fk_credentials_owner_principal_id_principals',
                'principals',
                ['owner_principal_id'],
                ['id'],
                ondelete='CASCADE',
            )
        op.create_index(
            'ix_credentials_owner_principal_id',
            'credentials',
            ['owner_principal_id'],
        )

    if column_exists('users', 'hashed_password'):
        if bind.dialect.name == 'sqlite':
            require_change_options = (
                "json_object('require_password_change', "
                "CASE WHEN u.require_password_change THEN json('true') "
                "ELSE json('false') END)"
            )
        elif bind.dialect.name == 'mysql':
            require_change_options = (
                "JSON_OBJECT('require_password_change', "
                "CAST(u.require_password_change AS JSON))"
            )
        else:
            require_change_options = (
                "jsonb_build_object('require_password_change', "
                "u.require_password_change)"
            )

        op.execute(
            sa.text(
                f"""
                INSERT INTO credentials (
                    credential_type, owner_principal_id,
                    public_key, encoded_secret, options,
                    created_at, updated_at
                )
                SELECT
                    'PASSWORD', u.principal_id,
                    NULL, u.hashed_password, {require_change_options},
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM users u
                WHERE u.hashed_password IS NOT NULL
                  AND u.hashed_password <> ''
                  AND u.principal_id IS NOT NULL
                """
            )
        )

        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.drop_column('hashed_password')
            if column_exists('users', 'require_password_change'):
                batch_op.drop_column('require_password_change')

    # ---- 16. Drop legacy usermodelroutelink -----------------------------

    # The user-grant rows it carried have been moved into
    # ``model_route_principals`` in step 13. Keep the table around any
    # longer and the OSS ALLOWED_USERS write path would silently keep
    # writing to it instead of the unified table.
    #
    # On postgres the legacy ``non_admin_user_models`` view (created by
    # the old code's ``init_db.listen_events``) references
    # ``usermodelroutelink`` and would block ``DROP TABLE`` with a
    # ``DependentObjectsStillExist`` error. Drop the view first; the
    # post-migration server start re-creates the new-shape view from
    # ``schemas/stmt.py:model_user_after_create_view_stmt`` via the
    # ``metadata.after_create`` listener.
    if user_link_table:
        op.execute(model_user_after_drop_view_stmt)
        op.drop_table(user_link_table)


def downgrade() -> None:
    bind = op.get_bind()

    # ---- Drop the multi-tenancy views -----------------------------------
    # Necessary so the column / table drops below don't trip postgres'
    # "cannot drop column referenced by view" guard. ``principal_users``
    # is dropped after the view that depends on it. Neither view is
    # restored here — the rolled-back code's ``init_db.listen_events``
    # recreates them from ``schemas/stmt.py`` on the next server
    # startup, the same way the upgrade path relies on for the
    # multi-tenancy version.
    # We also deliberately leave new accesspolicyenum values alone —
    # postgres cannot drop a single enum value cleanly while other
    # columns reference the type, and unused enum values are harmless.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(principal_users_after_drop_view_stmt)

    # ---- Drop denormalized columns from cluster-derived tables ----------

    for tbl in (
        "model_usages",
        "model_providers",
        "benchmarks",
        "model_files",
        "workers",
    ):
        with op.batch_alter_table(tbl, schema=None) as batch_op:
            try:
                batch_op.drop_constraint(
                    f"fk_{tbl}_owner_principal_id_principals", type_="foreignkey"
                )
            except Exception:
                pass
            try:
                batch_op.drop_column("owner_principal_id")
            except Exception:
                pass

    with op.batch_alter_table("model_files", schema=None) as batch_op:
        try:
            batch_op.drop_column("cluster_id")
        except Exception:
            pass

    # ---- Inference backends Hybrid revert -------------------------------

    with op.batch_alter_table("inference_backends", schema=None) as batch_op:
        try:
            batch_op.drop_index("ix_inference_backends_backend_name")
        except Exception:
            pass
        try:
            batch_op.drop_constraint(
                "uix_inference_backends_name_owner", type_="unique"
            )
        except Exception:
            pass
        try:
            batch_op.drop_constraint(
                "fk_inference_backends_owner_principal_id_principals",
                type_="foreignkey",
            )
        except Exception:
            pass
        try:
            batch_op.drop_column("owner_principal_id")
        except Exception:
            pass

    # ---- BYO cluster column removal -------------------------------------

    op.execute("DROP INDEX IF EXISTS uix_clusters_default_per_owner")
    for tbl in ("worker_pools", "cloud_credentials", "clusters"):
        with op.batch_alter_table(tbl, schema=None) as batch_op:
            try:
                batch_op.drop_constraint(
                    f"fk_{tbl}_owner_principal_id_principals",
                    type_="foreignkey",
                )
            except Exception:
                pass
            try:
                batch_op.drop_column("owner_principal_id")
            except Exception:
                pass

    # ---- model_routes ---------------------------------------------------

    with op.batch_alter_table('model_routes', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_model_routes_owner_principal_id_principals',
                type_='foreignkey',
            )
        except Exception:
            pass
        try:
            batch_op.drop_column('owner_principal_id')
        except Exception:
            pass

    # ---- model_instances ------------------------------------------------

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_model_instances_owner_principal_id_principals',
                type_='foreignkey',
            )
        except Exception:
            pass
        try:
            batch_op.drop_column('owner_principal_id')
        except Exception:
            pass

    # ---- models ---------------------------------------------------------

    with op.batch_alter_table('models', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_models_owner_principal_id_principals',
                type_='foreignkey',
            )
        except Exception:
            pass
        try:
            batch_op.drop_column('owner_principal_id')
        except Exception:
            pass

    # ---- api_keys -------------------------------------------------------

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_api_keys_owner_principal_id_principals',
                type_='foreignkey',
            )
        except Exception:
            pass
        try:
            batch_op.drop_constraint('uix_user_owner_name', type_='unique')
        except Exception:
            pass
        batch_op.create_unique_constraint(
            'uix_user_id_name', ['user_id', 'name']
        )
        try:
            batch_op.drop_column('owner_principal_id')
        except Exception:
            pass

    # ---- credentials: restore password columns on users -----------------
    #
    # Move PASSWORD-row state back to ``users`` (hashed_password +
    # require_password_change), then drop the rows and the
    # owner_principal_id FK so the credentials table looks like its
    # v2.2.0 shape again. The column-shape revert (rename back,
    # public_key NOT NULL, etc.) happens in the v2.2.0 downgrade — keep
    # the layering symmetrical to upgrade.

    if not column_exists('users', 'hashed_password'):
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('hashed_password', sa.String(length=255), nullable=True),
            )
    if not column_exists('users', 'require_password_change'):
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    'require_password_change',
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                ),
            )

    if bind.dialect.name == 'sqlite':
        require_change_extract = (
            "CAST(json_extract(c.options, '$.require_password_change') AS INTEGER)"
        )
    elif bind.dialect.name == 'mysql':
        require_change_extract = (
            "CAST(JSON_EXTRACT(c.options, '$.require_password_change') AS UNSIGNED)"
        )
    else:
        require_change_extract = (
            "COALESCE((c.options->>'require_password_change')::boolean, FALSE)"
        )

    op.execute(
        sa.text(
            f"""
            UPDATE users
            SET
                hashed_password = (
                    SELECT c.encoded_secret FROM credentials c
                    WHERE c.owner_principal_id = users.principal_id
                      AND c.credential_type = 'PASSWORD'
                      AND c.deleted_at IS NULL
                    LIMIT 1
                ),
                require_password_change = COALESCE(
                    (
                        SELECT {require_change_extract} FROM credentials c
                        WHERE c.owner_principal_id = users.principal_id
                          AND c.credential_type = 'PASSWORD'
                          AND c.deleted_at IS NULL
                        LIMIT 1
                    ),
                    FALSE
                )
            """
        )
    )

    op.execute(
        sa.text("DELETE FROM credentials WHERE credential_type = 'PASSWORD'")
    )

    if column_exists('credentials', 'owner_principal_id'):
        try:
            op.drop_index(
                'ix_credentials_owner_principal_id', table_name='credentials'
            )
        except Exception:
            pass
        with op.batch_alter_table('credentials', schema=None) as batch_op:
            try:
                batch_op.drop_constraint(
                    'fk_credentials_owner_principal_id_principals',
                    type_='foreignkey',
                )
            except Exception:
                pass
            batch_op.drop_column('owner_principal_id')

    # ---- users ----------------------------------------------------------

    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'uix_users_principal_id', type_='unique'
            )
        except Exception:
            pass
        try:
            batch_op.drop_constraint(
                'fk_users_principal_id_principals', type_='foreignkey'
            )
        except Exception:
            pass
        try:
            batch_op.drop_column('principal_id')
        except Exception:
            pass

    # ---- Drop new tables (reverse FK order) -----------------------------

    if table_exists('model_route_principals'):
        op.drop_table('model_route_principals')
    if table_exists('cluster_access'):
        op.drop_table('cluster_access')
    if table_exists('principal_memberships'):
        op.drop_table('principal_memberships')
    if table_exists('principals'):
        op.drop_table('principals')

    # ---- Drop enum types on postgres ------------------------------------

    if bind.dialect.name == 'postgresql':
        # Only drop the enums this migration owns. ``authproviderenum``
        # predates this migration and is still in use by
        # ``users.source`` after downgrade.
        for enum in reversed(_enums()):
            try:
                enum.drop(bind, checkfirst=True)
            except Exception:
                pass
