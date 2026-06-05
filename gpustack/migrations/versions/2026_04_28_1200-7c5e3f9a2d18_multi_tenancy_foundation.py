"""multi-tenancy foundation

Sets up the entire tenancy storage layer in one upgrade. The
identity surface is a single ``principals`` table (renamed from
``users``, extended with kind / name / display_name / description /
parent columns). USER, ORG, GROUP, and SYSTEM are all rows in that
table; the discriminator is ``kind``. Resource ownership
(``owner_principal_id``), ACL grants, and memberships all reference
``principals.id`` directly — no polymorphic ``(type, id)`` pairs.

The identity surface uses k8s-style column naming: ``name`` is the
URL-safe identifier (analogous to ``metadata.name`` in k8s), and
``display_name`` is the human-readable label. This matches the GPU
service API the upcoming k8s integration will speak.

Logical groups, run in order:

1. Drop the legacy ``non_admin_user_models`` view (it references
   ``usermodelroutelink``; the server recreates the new-shape view
   from ``schemas/stmt.py`` at startup).
2. Extend the existing ``users`` table with the principal-shaped
   columns (``kind``, ``name``, ``display_name``, ``description``,
   ``parent_principal_id``), backfill USER kind data (``name`` = old
   ``username``, ``display_name`` = ``full_name`` or ``username``),
   then drop ``username`` / ``full_name`` — the new columns subsume
   them.
3. Insert the platform Org-principal row (``name='default'``,
   ``kind='ORG'``) as a row in the still-named ``users`` table.
4. Rename ``users`` → ``principals``. Existing ``user_id`` FKs
   (``api_keys`` / ``cluster_access`` / ``model_usages``) silently
   retarget the renamed table — same column name, same numeric
   values, no FK definition change required.
5. Add the principal-specific constraints / indexes that aren't on
   the old ``users`` table (name unique, group-display-name partial
   unique, parent self-FK).
6. Create the new related tables (``principal_memberships``,
   ``cluster_access``, ``model_route_principals``).
7. Backfill admin memberships in the platform Org (role=OWNER).
   Regular users get no auto-membership.
8. Add ``owner_principal_id`` to existing tenant-scoped tables:
   api_keys / models / model_instances / model_routes; backfill
   from user ownership (api_keys) or platform principal (others)
   and promote to NOT NULL.
9. BYO cluster: ``clusters`` / ``cloud_credentials`` / ``worker_pools``
   get ``owner_principal_id`` (NOT NULL after backfill); per-Org
   default cluster expressed via partial unique index.
10. Cluster-derived denormalized columns: workers / model_files /
    benchmarks / model_providers / model_usages each get
    ``owner_principal_id`` (nullable; SET NULL on principal delete).
    ``model_files`` gains ``cluster_id``.
11. Inference backends Hybrid: ``owner_principal_id`` (nullable =
    Platform row), composite unique on (backend_name,
    owner_principal_id).
12. Backfill ``model_route_principals`` from the legacy
    ``usermodelroutelink`` table (USER-kind principal_id == user.id
    after rename).
13. Recreate ``accesspolicyenum`` as {PUBLIC, AUTHED,
    ALLOWED_PRINCIPALS}, folding the released ``ALLOWED_USERS`` into
    ``ALLOWED_PRINCIPALS`` and converting existing rows.
14. Drop the legacy ``usermodelroutelink`` table.
15. Extract login credentials into ``user_passwords`` and drop
    ``hashed_password`` / ``require_password_change`` from
    ``principals``.
16. Promote system actors to a dedicated ``SYSTEM`` kind, invert the
    cluster/worker→principal FK direction (``clusters.system_principal_id``
    / ``workers.system_principal_id``), and drop ``is_system`` /
    ``role`` / ``cluster_id`` / ``worker_id`` from ``principals``.

Revision ID: 7c5e3f9a2d18
Revises: 8bf38a6bb3b5
Create Date: 2026-04-28 12:00:00.000000

"""
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack  # noqa: F401  (keeps SQLModel registrations side-effect-loaded)
import gpustack.utils.sql_enum as sql_enum
from gpustack.migrations.utils import column_exists, table_exists
from gpustack.schemas.principals import (
    AUTHENTICATED_PRINCIPAL_DISPLAY_NAME,
    AUTHENTICATED_PRINCIPAL_NAME,
    AuthProviderEnum,
    PLATFORM_PRINCIPAL_NAME,
)
from gpustack.schemas.stmt import model_user_after_drop_view_stmt


# revision identifiers, used by Alembic.
revision: str = '7c5e3f9a2d18'
down_revision: Union[str, None] = '8bf38a6bb3b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PLATFORM_PRINCIPAL_DISPLAY_NAME = 'Default'


def _enums_to_create(bind):
    """Builder enums *owned* by this migration — created via explicit
    ``.create(bind, checkfirst=True)`` at the top of ``upgrade()``.
    Returned only so ``downgrade()`` can drop them; column declarations
    use the ``_ref`` variants below to avoid Alembic's
    ``checkfirst=False`` auto-create reissuing ``CREATE TYPE``.
    """
    return [
        sa.Enum('OWNER', 'MEMBER', name='orgrole'),
        sa.Enum('ORG', 'GROUP', 'USER', name='principaltype'),
        sa.Enum('ARGON2', name='passwordalgorithm'),
    ]


def _password_algorithm_ref(bind):
    """Column-reference variant of ``passwordalgorithm``. See
    :func:`_org_role_ref` for the create_type=False rationale.
    """
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import ENUM as PGEnum

        return PGEnum('ARGON2', name='passwordalgorithm', create_type=False)
    return sa.Enum('ARGON2', name='passwordalgorithm')


def _org_role_ref(bind):
    """Column-reference variant of the ``orgrole`` enum. PG uses
    ``create_type=False`` so the table-create event won't try to
    re-issue ``CREATE TYPE`` (Alembic's create_table dispatch passes
    ``checkfirst=False``, which turns the auto-create into a hard
    duplicate-type error on the second referencing table). MySQL
    renders the enum inline at the column type, no pre-declare needed.
    """
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import ENUM as PGEnum

        return PGEnum('OWNER', 'MEMBER', name='orgrole', create_type=False)
    return sa.Enum('OWNER', 'MEMBER', name='orgrole')


def _principal_type_ref(bind):
    """Column-reference variant of the ``principaltype`` enum. See
    :func:`_org_role_ref` for the rationale.
    """
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import ENUM as PGEnum

        return PGEnum(
            'ORG', 'GROUP', 'USER', name='principaltype', create_type=False
        )
    return sa.Enum('ORG', 'GROUP', 'USER', name='principaltype')


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


def _consolidate_access_policy_enum(bind):
    """Make ``ALLOWED_PRINCIPALS`` the single "specific grants" policy.

    The released ``ALLOWED_USERS`` is folded into ``ALLOWED_PRINCIPALS``
    — a USER-kind grant resolves identically — and existing rows on both
    ``models`` and ``model_routes`` are converted.

    The enum is recreated with the final value set, NOT
    ``ALTER TYPE ... ADD VALUE`` + ``UPDATE``: on PostgreSQL a freshly
    added enum label is unusable in the transaction that added it, and
    committing it mid-migration (an autocommit block) would break this
    migration's atomicity — it runs as part of one big transaction, so a
    later failure must roll the whole thing back. The recreate is plain
    transactional DDL. ``accesspolicyenum`` is shared by
    ``models.access_policy`` and ``model_routes.access_policy`` (both
    NOT NULL DEFAULT 'AUTHED'), so both columns convert in one pass; the
    ``USING`` cast remaps ALLOWED_USERS and every other value passes
    through unchanged.
    """
    dialect = bind.dialect.name
    targets = [t for t in ('models', 'model_routes') if table_exists(t)]

    if dialect == 'postgresql':  # also openGauss (PG-compatible dialect)
        # The column DEFAULT is typed against the old enum; drop it
        # before the swap and restore it after.
        for table in targets:
            bind.execute(
                sa.text(
                    f"ALTER TABLE {table} ALTER COLUMN access_policy DROP DEFAULT"
                )
            )
        bind.execute(
            sa.text(
                "CREATE TYPE accesspolicyenum_new AS ENUM "
                "('PUBLIC', 'AUTHED', 'ALLOWED_PRINCIPALS')"
            )
        )
        for table in targets:
            bind.execute(
                sa.text(
                    f"ALTER TABLE {table} ALTER COLUMN access_policy TYPE "
                    "accesspolicyenum_new USING ("
                    "CASE WHEN access_policy::text = 'ALLOWED_USERS' "
                    "THEN 'ALLOWED_PRINCIPALS' ELSE access_policy::text END"
                    ")::accesspolicyenum_new"
                )
            )
        bind.execute(sa.text("DROP TYPE accesspolicyenum"))
        bind.execute(
            sa.text("ALTER TYPE accesspolicyenum_new RENAME TO accesspolicyenum")
        )
        for table in targets:
            bind.execute(
                sa.text(
                    f"ALTER TABLE {table} ALTER COLUMN access_policy "
                    "SET DEFAULT 'AUTHED'"
                )
            )
    elif dialect == 'mysql':  # also OceanBase (MySQL-compatible dialect)
        # Inline ENUM per column (no shared type). Widen to admit
        # ALLOWED_PRINCIPALS, remap the data, then narrow to the final
        # set. MySQL DDL auto-commits, so there's no in-transaction
        # restriction to work around.
        wide = (
            "ENUM('PUBLIC', 'AUTHED', 'ALLOWED_USERS', 'ALLOWED_PRINCIPALS') "
            "NOT NULL DEFAULT 'AUTHED'"
        )
        final = (
            "ENUM('PUBLIC', 'AUTHED', 'ALLOWED_PRINCIPALS') NOT NULL DEFAULT 'AUTHED'"
        )
        for table in targets:
            bind.execute(
                sa.text(f"ALTER TABLE {table} MODIFY COLUMN access_policy {wide}")
            )
            bind.execute(
                sa.text(
                    f"UPDATE {table} SET access_policy = 'ALLOWED_PRINCIPALS' "
                    "WHERE access_policy = 'ALLOWED_USERS'"
                )
            )
            bind.execute(
                sa.text(f"ALTER TABLE {table} MODIFY COLUMN access_policy {final}")
            )


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    org_role = _org_role_ref(bind)
    principal_type = _principal_type_ref(bind)

    # Stamp seeded rows with an explicit UTC instant rather than the
    # dialect's ``CURRENT_TIMESTAMP``. Postgres / MySQL / openGauss
    # convert ``CURRENT_TIMESTAMP`` to the session/server timezone when
    # writing into a ``TIMESTAMP WITHOUT TIME ZONE`` column, but the
    # app's ``UTCDateTime`` decoder assumes the stored naive value IS
    # UTC and tags it accordingly on read — so a non-UTC DB server
    # surfaces the "Default" Org (and other seeded rows) with a
    # ``created_at`` that drifts from real wall time by the server's
    # tz offset.
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    if dialect == 'postgresql':
        # Eagerly create the enum types so subsequent column references
        # can rely on them existing. The ``_ref`` variants used in
        # column definitions have ``create_type=False`` so Alembic's
        # table-create dispatch (which passes ``checkfirst=False``)
        # won't double-issue ``CREATE TYPE``.
        for enum in _enums_to_create(bind):
            enum.create(bind, checkfirst=True)

    # ------------------------------------------------------------------
    # 1. Drop the legacy non_admin_user_models view.
    # ------------------------------------------------------------------
    # The view references ``usermodelroutelink``, which step 14 drops.
    # ``init_db.listen_events`` re-creates the new-shape view on the
    # next server boot from stmt.py.
    op.execute(model_user_after_drop_view_stmt)

    # ------------------------------------------------------------------
    # 2. Extend ``users`` with the principal-shaped columns.
    # ------------------------------------------------------------------
    # USER's identifier collapses into ``name`` (was ``username``) and
    # display label collapses into ``display_name`` (was ``full_name``
    # falling back to ``username``). The old columns are dropped at
    # the end of this step.
    #
    # Preflight: if any existing user has ``username`` equal to the
    # platform Org name we're about to seed, the unique name index
    # would collide. Surface the conflict before mutating anything so
    # the operator can rename the user and retry.
    conflict = bind.execute(
        sa.text(
            "SELECT id, username FROM users WHERE username = :name"
        ).bindparams(name=PLATFORM_PRINCIPAL_NAME)
    ).first()
    if conflict is not None:
        raise RuntimeError(
            f"Cannot upgrade: user id={conflict[0]} username="
            f"{conflict[1]!r} collides with the reserved platform Org "
            f"name {PLATFORM_PRINCIPAL_NAME!r}. Rename that user "
            "before re-running the migration."
        )

    with op.batch_alter_table('users', schema=None) as batch_op:
        if not column_exists('users', 'kind'):
            batch_op.add_column(
                sa.Column('kind', principal_type, nullable=True)
            )
        if not column_exists('users', 'name'):
            batch_op.add_column(
                sa.Column('name', sa.String(length=255), nullable=True)
            )
        if not column_exists('users', 'display_name'):
            batch_op.add_column(
                sa.Column('display_name', sa.String(length=255), nullable=True)
            )
        if not column_exists('users', 'description'):
            batch_op.add_column(
                sa.Column('description', sa.Text(), nullable=True)
            )
        if not column_exists('users', 'parent_principal_id'):
            batch_op.add_column(
                sa.Column('parent_principal_id', sa.Integer(), nullable=True)
            )

    # Backfill USER-kind data for existing user rows.
    op.execute(
        sa.text(
            """
            UPDATE users
            SET kind = 'USER',
                name = username,
                display_name = COALESCE(full_name, username)
            WHERE kind IS NULL
            """
        )
    )

    # Drop the legacy identifier/display columns now that the data
    # lives in ``name`` / ``display_name``. The unique index on
    # ``username`` disappears with it; the new ``name`` unique
    # constraint replaces it in step 5.
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column(
            'kind', existing_type=principal_type, nullable=False
        )
        if column_exists('users', 'full_name'):
            batch_op.drop_column('full_name')
        if column_exists('users', 'username'):
            batch_op.drop_column('username')

    # ------------------------------------------------------------------
    # 3. Seed the platform Org-principal as a row in ``users``.
    # ------------------------------------------------------------------
    # Insert with id = MAX(id) + 1 so it sits above all USER rows;
    # this becomes the stable platform principal id once the rename
    # below renames the table. Idempotent via name existence check.
    bool_false = 'false' if dialect == 'postgresql' else '0'
    bool_true = 'true' if dialect == 'postgresql' else '1'
    # ``hashed_password`` is declared NOT NULL on upgraded v2.1.2
    # databases even though the initial migration created it nullable
    # — the SQLModel class shape evolved without an explicit
    # ``alter_column``. Insert an empty string the same way the v2.0
    # migration seeded its cluster system user (``hashed_password = ''``
    # for non-login rows); the column is dropped in step 16 anyway.
    op.execute(
        sa.text(
            f"""
            INSERT INTO users
                (id, kind, name, display_name, description,
                 is_admin, is_active, require_password_change, is_system,
                 hashed_password,
                 source,
                 created_at, updated_at, deleted_at)
            SELECT
                COALESCE((SELECT MAX(id) FROM users), 0) + 1,
                'ORG', :name, :display_name, :desc,
                {bool_false}, {bool_true}, {bool_false}, {bool_false},
                '',
                'Local',
                :now_utc, :now_utc, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM users WHERE name = :name
            )
            """
        ).bindparams(
            name=PLATFORM_PRINCIPAL_NAME,
            display_name=PLATFORM_PRINCIPAL_DISPLAY_NAME,
            desc='Built-in platform organization',
            now_utc=now_utc,
        )
    )

    # Resolve the platform principal's actual id from name; every
    # downstream backfill uses this resolved value.
    platform_id = bind.execute(
        sa.text(
            "SELECT id FROM users WHERE name = :name AND kind = 'ORG'"
        ).bindparams(name=PLATFORM_PRINCIPAL_NAME)
    ).scalar()
    if platform_id is None:
        raise RuntimeError(
            f"Platform principal not found by name={PLATFORM_PRINCIPAL_NAME!r} "
            "after seed; backfill cannot continue"
        )

    # ------------------------------------------------------------------
    # 4. Rename users → principals.
    # ------------------------------------------------------------------
    # Existing ``api_keys.user_id`` / ``cluster_access.granted_by`` /
    # ``model_usages.user_id`` FKs auto-retarget the renamed table on
    # PG / MySQL. SQLite re-creates dependent constraints via
    # batch_alter_table on subsequent ops. Column names and values
    # don't change.
    op.rename_table('users', 'principals')

    # Sync the PG sequence so future inserts don't collide with the
    # platform principal's manually-assigned id.
    if dialect == 'postgresql':
        op.execute(
            "SELECT setval(pg_get_serial_sequence('principals', 'id'), "
            "GREATEST((SELECT MAX(id) FROM principals), 1))"
        )

    # ------------------------------------------------------------------
    # 5. Principal-specific constraints / indexes on the renamed table.
    # ------------------------------------------------------------------
    with op.batch_alter_table('principals', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_principals_parent_principal_id_principals',
            'principals',
            ['parent_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ``principals.name`` uniqueness is partitioned by kind: USER, ORG,
    # and GROUP each get their own independent name namespace.
    # - USER is carved out so an admin-created Org and a (manually- or
    #   IdP-provisioned) User may share a name. USER names never appear
    #   in the ``<owner-name>/<route>`` inference URL (personal Orgs —
    #   which are USER principals — don't deploy), and login lookups
    #   (``get_by_username`` / IdP provisioning) are kind-scoped to USER,
    #   so a same-named ORG can no longer be mis-resolved as a login.
    # - GROUP has always had its own namespace — IdP-supplied group
    #   names commonly coincide with admin-chosen Org names, and forcing
    #   the two globally unique would break OIDC/SAML group sync.
    # - SYSTEM gets no partial unique index: its rows are internally
    #   generated (``system/cluster-<id>`` / ``system/worker-<id>``), so
    #   they're structurally unique with no user-facing create path, and
    #   the ``SYSTEM`` enum value doesn't even exist until step 17a.
    #
    # Postgres supports partial unique indexes natively. MySQL has
    # no equivalent — fall back to a single plain (non-unique) index
    # for lookup performance; uniqueness within each partition is
    # enforced by the create routes (``create_user`` /
    # ``create_organization`` / ``_insert_group_or_refetch``).
    if dialect == 'postgresql':
        for kind in ('USER', 'ORG', 'GROUP'):
            op.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS uix_principals_{kind.lower()}_name "
                "ON principals (name) "
                f"WHERE kind = '{kind}' AND deleted_at IS NULL"
            )
    else:
        op.create_index('ix_principals_name', 'principals', ['name'])

    # ------------------------------------------------------------------
    # 6. New related tables.
    # ------------------------------------------------------------------
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
            # ``granted_by`` historically pointed at users.id; now that
            # users IS principals, point it at principals.id (same numeric
            # values for the USER-kind rows it captures).
            sa.ForeignKeyConstraint(
                ['granted_by'], ['principals.id'], ondelete='SET NULL',
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

    # ------------------------------------------------------------------
    # 7. Backfill admin memberships in the platform Org.
    # ------------------------------------------------------------------
    # Only admin users get auto-enrolled in the platform Org. Regular
    # users start with no team memberships — admins must add them
    # explicitly when team-workspace access is wanted. (System users
    # also skipped — workers / cluster service accounts aren't tenants.)
    if dialect == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO principal_memberships
                    (parent_principal_id, member_principal_id, role,
                     created_at, updated_at, deleted_at)
                SELECT :platform_id, p.id, 'OWNER'::orgrole,
                       :now_utc, :now_utc, NULL
                FROM principals p
                WHERE p.kind = 'USER'::principaltype
                  AND p.is_admin = true
                  AND COALESCE(p.is_system, false) = false
                  AND NOT EXISTS (
                    SELECT 1 FROM principal_memberships m
                    WHERE m.parent_principal_id = :platform_id
                      AND m.member_principal_id = p.id
                      AND m.deleted_at IS NULL
                  )
                """
            ).bindparams(platform_id=platform_id, now_utc=now_utc)
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT INTO principal_memberships
                    (parent_principal_id, member_principal_id, role,
                     created_at, updated_at, deleted_at)
                SELECT :platform_id, p.id, 'OWNER',
                       :now_utc, :now_utc, NULL
                FROM principals p
                WHERE p.kind = 'USER'
                  AND p.is_admin = 1
                  AND COALESCE(p.is_system, 0) = 0
                  AND NOT EXISTS (
                    SELECT 1 FROM principal_memberships m
                    WHERE m.parent_principal_id = :platform_id
                      AND m.member_principal_id = p.id
                      AND m.deleted_at IS NULL
                  )
                """
            ).bindparams(platform_id=platform_id, now_utc=now_utc)
        )

    # ------------------------------------------------------------------
    # 8. api_keys.owner_principal_id.
    # ------------------------------------------------------------------
    if not column_exists('api_keys', 'owner_principal_id'):
        with op.batch_alter_table('api_keys', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

    # api_keys are user-owned (each row carries ``user_id``). After
    # the users → principals rename, ``user_id`` IS the user's
    # USER-principal id. Stamp ``owner_principal_id`` from it so the
    # per-user list filter keeps each key visible to its owner. Fall
    # back to platform principal for orphaned rows.
    op.execute(
        sa.text(
            """
            UPDATE api_keys
            SET owner_principal_id = user_id
            WHERE owner_principal_id IS NULL
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
        # Unique scope was (user_id, name); widen to
        # (user_id, owner_principal_id, name) so the same key name can
        # coexist across owner namespaces. The new index must be created
        # BEFORE dropping the old one: on MySQL/OceanBase the old unique
        # index is the only thing satisfying the user_id FK index
        # requirement; dropping it first errors with "needed in a foreign
        # key constraint". The new composite keeps ``user_id`` as the
        # leftmost prefix so it can take over that role.
        batch_op.create_unique_constraint(
            'uix_user_owner_name', ['user_id', 'owner_principal_id', 'name']
        )
        batch_op.drop_constraint('uix_user_id_name', type_='unique')
        batch_op.create_foreign_key(
            'fk_api_keys_owner_principal_id_principals',
            'principals',
            ['owner_principal_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ------------------------------------------------------------------
    # 9. models / model_instances / model_routes.owner_principal_id.
    # ------------------------------------------------------------------
    for tbl in ('models', 'model_instances', 'model_routes'):
        if not column_exists(tbl, 'owner_principal_id'):
            with op.batch_alter_table(tbl, schema=None) as batch_op:
                batch_op.add_column(
                    sa.Column('owner_principal_id', sa.Integer(), nullable=True)
                )

        op.execute(
            sa.text(
                f"UPDATE {tbl} SET owner_principal_id = :pid "
                "WHERE owner_principal_id IS NULL"
            ).bindparams(pid=platform_id)
        )

        with op.batch_alter_table(tbl, schema=None) as batch_op:
            batch_op.alter_column(
                'owner_principal_id',
                existing_type=sa.Integer(),
                nullable=False,
            )
            batch_op.create_foreign_key(
                f'fk_{tbl}_owner_principal_id_principals',
                'principals',
                ['owner_principal_id'],
                ['id'],
                ondelete='CASCADE',
            )

    # ------------------------------------------------------------------
    # 10. BYO cluster: clusters / cloud_credentials / worker_pools.
    # ------------------------------------------------------------------
    # Clusters / cloud credentials / worker_pools always belong to a
    # principal. Cross-principal sharing goes through cluster_access.
    # ON DELETE CASCADE: deleting the owner principal takes its infra
    # rows with it.
    for tbl in ('clusters', 'cloud_credentials', 'worker_pools'):
        if not column_exists(tbl, 'owner_principal_id'):
            with op.batch_alter_table(tbl, schema=None) as batch_op:
                batch_op.add_column(
                    sa.Column('owner_principal_id', sa.Integer(), nullable=True)
                )
                batch_op.create_foreign_key(
                    f'fk_{tbl}_owner_principal_id_principals',
                    'principals',
                    ['owner_principal_id'],
                    ['id'],
                    ondelete='CASCADE',
                )

        op.execute(
            sa.text(
                f"UPDATE {tbl} SET owner_principal_id = :pid "
                "WHERE owner_principal_id IS NULL"
            ).bindparams(pid=platform_id)
        )

        with op.batch_alter_table(tbl, schema=None) as batch_op:
            batch_op.alter_column(
                'owner_principal_id',
                existing_type=sa.Integer(),
                nullable=False,
            )

    # At most one default cluster per principal. Postgres expresses
    # this as a partial unique on the active (non-soft-deleted, default)
    # rows so a principal can "rotate" defaults by soft-deleting the
    # old + flipping the new without conflict. MySQL/OceanBase have
    # no partial-index equivalent — fall back to a composite
    # (owner_principal_id, is_default) index that speeds up the
    # default-cluster lookup. A plain (owner_principal_id) index
    # would be redundant with the FK's auto-created index. Uniqueness
    # is enforced by the cluster routes (``create_cluster`` only
    # auto-defaults when none exist; ``set_default_cluster`` unsets
    # prior defaults first).
    if dialect == 'postgresql':
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_clusters_default_per_owner "
            "ON clusters (owner_principal_id) "
            "WHERE is_default = true AND deleted_at IS NULL"
        )
    else:
        op.create_index(
            'ix_clusters_default_per_owner',
            'clusters',
            ['owner_principal_id', 'is_default'],
        )

    # ------------------------------------------------------------------
    # 11. Cluster-derived denormalized owner_principal_id columns.
    # ------------------------------------------------------------------
    # Workers, model_files, benchmarks, model_usages all need an owner
    # pointer for per-row filtering. Nullable (NULL = on a global
    # cluster, admin-managed); ON DELETE SET NULL keeps rows alive
    # when the owner principal is deleted (principal delete cascades
    # clusters, which cascade their workers anyway).
    #
    # ``model_providers`` was originally in this group but later moved
    # to the always-owned tier — see step 11b below.
    for tbl in (
        'workers',
        'model_files',
        'benchmarks',
        'model_usages',
    ):
        if not column_exists(tbl, 'owner_principal_id'):
            with op.batch_alter_table(tbl, schema=None) as batch_op:
                batch_op.add_column(
                    sa.Column('owner_principal_id', sa.Integer(), nullable=True)
                )
                batch_op.create_foreign_key(
                    f'fk_{tbl}_owner_principal_id_principals',
                    'principals',
                    ['owner_principal_id'],
                    ['id'],
                    ondelete='SET NULL',
                )

    # model_usages also needs a consumer-side tenant pointer for "usage made
    # with this Org's API keys" views. It intentionally differs from
    # owner_principal_id when Org A calls a deployment shared by Org B.
    if not column_exists('model_usages', 'consumer_principal_id'):
        with op.batch_alter_table('model_usages', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('consumer_principal_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_model_usages_consumer_principal_id_principals',
                'principals',
                ['consumer_principal_id'],
                ['id'],
                ondelete='SET NULL',
            )
            batch_op.create_index(
                'ix_model_usages_consumer_principal_id',
                ['consumer_principal_id'],
                unique=False,
            )

    for tbl in (
        'model_usages',
        'model_usage_details',
        'model_usage_details_archive',
    ):
        if not table_exists(tbl) or not column_exists(tbl, 'consumer_principal_id'):
            continue
        if dialect == 'mysql':
            op.execute(
                sa.text(
                    f"""
                    UPDATE {tbl} mu
                    JOIN api_keys ak ON ak.access_key = mu.access_key
                    SET mu.consumer_principal_id = ak.owner_principal_id
                    WHERE mu.consumer_principal_id IS NULL
                    """
                )
            )
        else:
            op.execute(
                sa.text(
                    f"""
                    UPDATE {tbl}
                    SET consumer_principal_id = (
                        SELECT api_keys.owner_principal_id
                        FROM api_keys
                        WHERE api_keys.access_key = {tbl}.access_key
                    )
                    WHERE consumer_principal_id IS NULL
                      AND EXISTS (
                        SELECT 1
                        FROM api_keys
                        WHERE api_keys.access_key = {tbl}.access_key
                      )
                    """
                )
            )

    # model_files only had worker_id; add cluster_id for direct
    # cluster_access-based filtering.
    if not column_exists('model_files', 'cluster_id'):
        with op.batch_alter_table('model_files', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('cluster_id', sa.Integer(), nullable=True)
            )

    # Backfill the cluster-derived tenant columns just added above. The
    # ADD COLUMN step leaves every legacy row with NULL ``cluster_id`` /
    # ``owner_principal_id``, which makes them invisible to non-bypass
    # principals (cluster_access filter requires one to match). Derive
    # the scope from the row's worker → cluster, same way the runtime
    # paths (``routes/model_files.create_model_file`` /
    # ``controllers._get_worker_tenant_scopes``) stamp new rows.
    #
    # Order matters: workers.owner_principal_id is sourced from
    # clusters.owner_principal_id (already backfilled in section 10),
    # and model_files / benchmarks read back through workers — so
    # workers first, dependents second.
    if dialect == 'mysql':
        op.execute(
            sa.text(
                """
                UPDATE workers w
                JOIN clusters c ON c.id = w.cluster_id
                SET w.owner_principal_id = c.owner_principal_id
                WHERE w.owner_principal_id IS NULL
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE model_files mf
                JOIN workers w ON w.id = mf.worker_id
                SET mf.cluster_id = w.cluster_id,
                    mf.owner_principal_id = w.owner_principal_id
                WHERE mf.worker_id IS NOT NULL
                  AND (mf.cluster_id IS NULL OR mf.owner_principal_id IS NULL)
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE benchmarks b
                JOIN workers w ON w.id = b.worker_id
                SET b.cluster_id = w.cluster_id,
                    b.owner_principal_id = w.owner_principal_id
                WHERE b.worker_id IS NOT NULL
                  AND (b.cluster_id IS NULL OR b.owner_principal_id IS NULL)
                """
            )
        )
    else:
        op.execute(
            sa.text(
                """
                UPDATE workers
                SET owner_principal_id = (
                    SELECT clusters.owner_principal_id
                    FROM clusters
                    WHERE clusters.id = workers.cluster_id
                )
                WHERE owner_principal_id IS NULL
                  AND EXISTS (
                    SELECT 1 FROM clusters
                    WHERE clusters.id = workers.cluster_id
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE model_files
                SET cluster_id = (
                    SELECT workers.cluster_id
                    FROM workers
                    WHERE workers.id = model_files.worker_id
                ),
                owner_principal_id = (
                    SELECT workers.owner_principal_id
                    FROM workers
                    WHERE workers.id = model_files.worker_id
                )
                WHERE model_files.worker_id IS NOT NULL
                  AND (model_files.cluster_id IS NULL
                       OR model_files.owner_principal_id IS NULL)
                  AND EXISTS (
                    SELECT 1 FROM workers
                    WHERE workers.id = model_files.worker_id
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE benchmarks
                SET cluster_id = (
                    SELECT workers.cluster_id
                    FROM workers
                    WHERE workers.id = benchmarks.worker_id
                ),
                owner_principal_id = (
                    SELECT workers.owner_principal_id
                    FROM workers
                    WHERE workers.id = benchmarks.worker_id
                )
                WHERE benchmarks.worker_id IS NOT NULL
                  AND (benchmarks.cluster_id IS NULL
                       OR benchmarks.owner_principal_id IS NULL)
                  AND EXISTS (
                    SELECT 1 FROM workers
                    WHERE workers.id = benchmarks.worker_id
                  )
                """
            )
        )

    # ------------------------------------------------------------------
    # 11b. ``model_providers``: always-owned (no Global notion).
    # ------------------------------------------------------------------
    # Only ``gpu_instance_templates`` and ``inference_backends`` carry a
    # NULL-owner Global notion. Providers always belong to an Org —
    # admin in "All" mode creates them under the platform Org. Mirrors
    # the ``models`` block in step 9: add nullable so the DDL accepts
    # pre-existing rows, backfill to the platform Org, then lock to
    # NOT NULL with a CASCADE FK.
    if table_exists('model_providers') and not column_exists(
        'model_providers', 'owner_principal_id'
    ):
        with op.batch_alter_table('model_providers', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )

        op.execute(
            sa.text(
                "UPDATE model_providers SET owner_principal_id = :pid "
                "WHERE owner_principal_id IS NULL"
            ).bindparams(pid=platform_id)
        )

        with op.batch_alter_table('model_providers', schema=None) as batch_op:
            batch_op.alter_column(
                'owner_principal_id',
                existing_type=sa.Integer(),
                nullable=False,
            )
            batch_op.create_foreign_key(
                'fk_model_providers_owner_principal_id_principals',
                'principals',
                ['owner_principal_id'],
                ['id'],
                ondelete='CASCADE',
            )

    # ------------------------------------------------------------------
    # 11c. Convert global UNIQUE on ``name`` to per-owner composite for
    # ``models``, ``model_providers``, and ``model_routes``.
    # ------------------------------------------------------------------
    # These tables used to enforce ``name`` as globally unique. With
    # multi-tenancy each row carries an ``owner_principal_id``, so two
    # Orgs can independently define a "qwen3-0.6b" model / route /
    # "openai" provider. Drop the legacy global UNIQUE and replace it
    # with a composite UNIQUE on ``(owner_principal_id, name)``.
    #
    # The drop dance mirrors step 12 (inference_backends): MySQL /
    # OceanBase reflect unique-only indexes via information_schema;
    # Postgres / openGauss go through ``pg_constraint`` + a fallback
    # ``DROP INDEX`` for the index Alembic emitted on column-level
    # ``unique=True``. We avoid ``batch_alter_table`` for the drops so
    # an idempotent re-run doesn't trip on an already-removed
    # constraint.
    for tbl, idx_name, uq_name in (
        ('models', 'ix_models_name', 'uix_models_name_per_owner'),
        (
            'model_providers',
            'ix_model_providers_name',
            'uix_model_providers_name_per_owner',
        ),
        (
            'model_routes',
            'ix_model_routes_name',
            'uix_model_routes_name_per_owner',
        ),
    ):
        if not table_exists(tbl):
            continue
        if dialect == 'postgresql':
            for name_row in bind.execute(
                sa.text(
                    "SELECT c.conname FROM pg_constraint c "
                    "JOIN pg_class t ON t.oid = c.conrelid "
                    "WHERE t.relname = :tbl "
                    "AND c.contype = 'u' "
                    "AND array_length(c.conkey, 1) = 1 "
                    "AND (SELECT a.attname FROM pg_attribute a "
                    "     WHERE a.attrelid = c.conrelid "
                    "     AND a.attnum = c.conkey[1]) = 'name'"
                ).bindparams(tbl=tbl)
            ):
                op.execute(
                    f'ALTER TABLE {tbl} '
                    f'DROP CONSTRAINT IF EXISTS "{name_row[0]}"'
                )
            op.execute(f'DROP INDEX IF EXISTS {idx_name}')
        else:
            unique_idx_names = [
                row[0]
                for row in bind.execute(
                    sa.text(
                        "SELECT DISTINCT INDEX_NAME FROM "
                        "information_schema.STATISTICS "
                        "WHERE TABLE_SCHEMA = DATABASE() "
                        "AND TABLE_NAME = :tbl "
                        "AND COLUMN_NAME = 'name' "
                        "AND NON_UNIQUE = 0 "
                        "AND INDEX_NAME <> 'PRIMARY'"
                    ).bindparams(tbl=tbl)
                )
            ]
            for name in unique_idx_names:
                op.execute(f"ALTER TABLE {tbl} DROP INDEX `{name}`")

        with op.batch_alter_table(tbl, schema=None) as batch_op:
            batch_op.create_unique_constraint(
                uq_name, ['owner_principal_id', 'name']
            )

    # ------------------------------------------------------------------
    # 12. Inference backends Hybrid.
    # ------------------------------------------------------------------
    # NULL owner_principal_id = Platform-managed (admin curates built-ins);
    # non-NULL = an Org's extension/override. backend_name is no longer
    # globally unique — composite unique on (backend_name, owner_principal_id)
    # lets each owner carry their own row alongside the Platform row.
    if not column_exists('inference_backends', 'owner_principal_id'):
        # Drop the stale unique-on-backend_name objects up-front using
        # eager DDL — ``batch_alter_table`` flushes lazily on __exit__, so
        # a try/except around a batch op can't catch "key doesn't exist"
        # errors at the right point. Going through the catalog directly
        # also sidesteps SQLAlchemy inspector inconsistencies (notably,
        # on MySQL/OceanBase neither ``get_unique_constraints`` nor
        # ``get_indexes`` reliably reports the explicit unique index
        # ``ix_inference_backends_backend_name``).
        if dialect == 'postgresql':
            # Find single-column unique constraints whose column is
            # ``backend_name``. ``conkey`` is a 1-based attnum array;
            # avoid ``WITH ORDINALITY`` and set-returning functions in
            # the predicate so this also runs on openGauss.
            for name_row in bind.execute(
                sa.text(
                    "SELECT c.conname FROM pg_constraint c "
                    "JOIN pg_class t ON t.oid = c.conrelid "
                    "WHERE t.relname = 'inference_backends' "
                    "AND c.contype = 'u' "
                    "AND array_length(c.conkey, 1) = 1 "
                    "AND (SELECT a.attname FROM pg_attribute a "
                    "     WHERE a.attrelid = c.conrelid "
                    "     AND a.attnum = c.conkey[1]) = 'backend_name'"
                )
            ):
                op.execute(
                    f'ALTER TABLE inference_backends '
                    f'DROP CONSTRAINT IF EXISTS "{name_row[0]}"'
                )
            op.execute(
                "DROP INDEX IF EXISTS ix_inference_backends_backend_name"
            )
        else:
            # MySQL/OceanBase: every unique object on backend_name is
            # backed by a unique index — drop them all from
            # information_schema. NON_UNIQUE = 0 picks unique-only;
            # exclude PRIMARY just in case.
            unique_idx_names = [
                row[0]
                for row in bind.execute(
                    sa.text(
                        "SELECT DISTINCT INDEX_NAME FROM "
                        "information_schema.STATISTICS "
                        "WHERE TABLE_SCHEMA = DATABASE() "
                        "AND TABLE_NAME = 'inference_backends' "
                        "AND COLUMN_NAME = 'backend_name' "
                        "AND NON_UNIQUE = 0 "
                        "AND INDEX_NAME <> 'PRIMARY'"
                    )
                )
            ]
            for name in unique_idx_names:
                op.execute(
                    f"ALTER TABLE inference_backends DROP INDEX `{name}`"
                )

        with op.batch_alter_table('inference_backends', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('owner_principal_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_inference_backends_owner_principal_id_principals',
                'principals',
                ['owner_principal_id'],
                ['id'],
                ondelete='CASCADE',
            )
            batch_op.create_unique_constraint(
                'uix_inference_backends_name_owner',
                ['backend_name', 'owner_principal_id'],
            )
            batch_op.create_index(
                'ix_inference_backends_backend_name', ['backend_name']
            )

    # ------------------------------------------------------------------
    # 13. Backfill model_route_principals from usermodelroutelink.
    # ------------------------------------------------------------------
    # The old (route_id, user_id) pairs become (route_id, principal_id):
    # after the users → principals rename, user.id IS the USER-principal
    # id, so the copy is a direct projection.
    #
    # No ON CONFLICT / IGNORE clause needed: the target table was
    # just created (step 6) so it's empty, and the source has a
    # composite PRIMARY KEY on (route_id, user_id) which matches the
    # target's UNIQUE(route_id, principal_id) 1:1 — duplicates are
    # structurally impossible.
    user_link_table = (
        'usermodelroutelink' if table_exists('usermodelroutelink') else None
    )
    if user_link_table:
        op.execute(
            sa.text(
                f"""
                INSERT INTO model_route_principals
                    (route_id, principal_id, created_at, updated_at)
                SELECT uml.route_id, uml.user_id,
                       :now_utc, :now_utc
                FROM {user_link_table} uml
                WHERE uml.route_id IS NOT NULL AND uml.user_id IS NOT NULL
                """
            ).bindparams(now_utc=now_utc)
        )

    # ------------------------------------------------------------------
    # 14. Consolidate access_policy -> ALLOWED_PRINCIPALS.
    # ------------------------------------------------------------------
    # ALLOWED_PRINCIPALS is the single "specific grants" policy: explicit
    # per-user / group / org grants via model_route_principals, and the
    # team-private default for non-platform Org routes (owning Org
    # auto-granted on create). It subsumes the released ``ALLOWED_USERS``
    # (a USER-kind grant resolves identically); the enum is recreated
    # without that label and existing rows are converted.
    _consolidate_access_policy_enum(bind)

    # ------------------------------------------------------------------
    # 15. Drop legacy usermodelroutelink.
    # ------------------------------------------------------------------
    # The user-grant rows it carried have been moved into
    # ``model_route_principals`` in step 13. Keep the table around any
    # longer and the OSS ALLOWED_USERS write path would silently keep
    # writing to it instead of the unified table.
    if user_link_table:
        op.drop_table(user_link_table)

    # ------------------------------------------------------------------
    # 16. Extract login credentials into ``user_passwords``.
    # ------------------------------------------------------------------
    # Pulls ``hashed_password`` and ``require_password_change`` off the
    # identity row into a dedicated table. Leaves Principal as pure
    # identity and lets the hash algorithm evolve without a schema
    # break — each row records its own ``algorithm``, the verifier
    # picks the right comparator at login time.
    #
    # Backfill: one row per USER principal that has a non-empty
    # password hash. System users and SSO-only users have empty / NULL
    # hashes — they don't get a row (no login).
    password_algorithm = _password_algorithm_ref(bind)
    if not table_exists('user_passwords'):
        op.create_table(
            'user_passwords',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('owner_principal_id', sa.Integer(), nullable=False),
            sa.Column(
                'algorithm',
                password_algorithm,
                nullable=False,
                server_default='ARGON2',
            ),
            sa.Column('hashed_secret', sa.String(length=255), nullable=False),
            sa.Column(
                'require_password_change',
                sa.Boolean(),
                nullable=False,
                server_default=(bool_false),
            ),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['owner_principal_id'], ['principals.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('user_passwords', schema=None) as batch_op:
            batch_op.create_index(
                'ix_user_passwords_owner_principal_id',
                ['owner_principal_id'],
                unique=False,
            )

    op.execute(
        sa.text(
            """
            INSERT INTO user_passwords
                (owner_principal_id, algorithm, hashed_secret,
                 require_password_change,
                 created_at, updated_at)
            SELECT id, 'ARGON2', hashed_password,
                   require_password_change,
                   :now_utc, :now_utc
              FROM principals
             WHERE kind = 'USER'
               AND hashed_password IS NOT NULL
               AND hashed_password <> ''
            """
        ).bindparams(now_utc=now_utc)
    )

    with op.batch_alter_table('principals', schema=None) as batch_op:
        if column_exists('principals', 'hashed_password'):
            batch_op.drop_column('hashed_password')
        if column_exists('principals', 'require_password_change'):
            batch_op.drop_column('require_password_change')

    # ------------------------------------------------------------------
    # 17. Promote system actors to SYSTEM kind + invert infra FK.
    # ------------------------------------------------------------------
    # Two-piece move that together makes ``principals`` pure identity:
    #
    #   a. ``is_system=true`` rows promote to ``kind='SYSTEM'``. The
    #      flag drops out: every USER-listing / ACL filter that used to
    #      say "is_system = false" now says "kind = 'USER'", which is
    #      what those callers actually meant.
    #   b. The Principal→Cluster / Principal→Worker FK direction is
    #      inverted. ``clusters.system_principal_id`` /
    #      ``workers.system_principal_id`` (UNIQUE, FK→principals.id,
    #      ON DELETE SET NULL) record which SYSTEM principal each infra
    #      row's bootstrap token authenticates as. The old
    #      ``role`` enum (Worker / Cluster) drops out — derivable from
    #      which table claims the principal.
    #
    # Done last because the source columns
    # (``is_system`` / ``cluster_id`` / ``worker_id``) need to survive
    # long enough to populate the inverse links.

    # 17a. Add ``SYSTEM`` to the principaltype enum.
    principal_type_enum = sa.Enum(
        'USER', 'ORG', 'GROUP', name='principaltype'
    )
    sql_enum.add_enum_values(
        {'principals': 'kind'}, principal_type_enum, 'SYSTEM'
    )

    # 17b. Add ``system_principal_id`` to clusters and workers
    # (nullable, UNIQUE, SET NULL on principal delete).
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        if not column_exists('clusters', 'system_principal_id'):
            batch_op.add_column(
                sa.Column('system_principal_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_clusters_system_principal_id_principals',
                'principals',
                ['system_principal_id'],
                ['id'],
                ondelete='SET NULL',
            )
            batch_op.create_unique_constraint(
                'uix_clusters_system_principal_id', ['system_principal_id']
            )
    with op.batch_alter_table('workers', schema=None) as batch_op:
        if not column_exists('workers', 'system_principal_id'):
            batch_op.add_column(
                sa.Column('system_principal_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_workers_system_principal_id_principals',
                'principals',
                ['system_principal_id'],
                ['id'],
                ondelete='SET NULL',
            )
            batch_op.create_unique_constraint(
                'uix_workers_system_principal_id', ['system_principal_id']
            )

    # 17c. Backfill the inverse links from the legacy
    # principals.cluster_id / principals.worker_id columns. Pre-rename
    # invariants: exactly one ``role='Cluster'`` system principal per
    # cluster, and exactly one ``role='Worker'`` system principal per
    # worker. ``LIMIT`` on the subselect is belt-and-braces against any
    # historical duplicates a rough upgrade might have left behind.
    op.execute(
        sa.text(
            f"""
            UPDATE clusters
            SET system_principal_id = (
                SELECT id FROM principals p
                 WHERE p.cluster_id = clusters.id
                   AND p.is_system = {bool_true}
                   AND p.role = 'Cluster'
                 LIMIT 1
            )
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            UPDATE workers
            SET system_principal_id = (
                SELECT id FROM principals p
                 WHERE p.worker_id = workers.id
                   AND p.is_system = {bool_true}
                   AND p.role = 'Worker'
                 LIMIT 1
            )
            """
        )
    )

    # 17d. Promote kind to SYSTEM for service-account rows.
    op.execute(
        sa.text(
            f"UPDATE principals SET kind = 'SYSTEM' WHERE is_system = {bool_true}"
        )
    )

    # 17e. Drop the now-redundant columns from principals.
    #
    # The column drops trip a "depends on" error unless the FK
    # constraints are dropped first, and the constraint names came
    # from the v2.0 migration with shapes like ``fk_users_cluster_id``
    # (not the autogen ``fk_principals_*`` template). Discover the
    # live FK names off the columns we're about to drop, so the
    # upgrade survives whatever historical migration named them.
    # The SQLAlchemy inspector has been observed to miss FKs on
    # OceanBase, so on MySQL-family dialects we also query
    # ``information_schema.KEY_COLUMN_USAGE`` directly and union the
    # results.
    fk_names_to_drop = set()
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys('principals'):
        cols = fk.get('constrained_columns') or []
        if fk.get('name') and any(
            c in ('cluster_id', 'worker_id') for c in cols
        ):
            fk_names_to_drop.add(fk['name'])
    if dialect == 'mysql':
        for row in bind.execute(
            sa.text(
                "SELECT DISTINCT CONSTRAINT_NAME "
                "FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'principals' "
                "AND REFERENCED_TABLE_NAME IS NOT NULL "
                "AND COLUMN_NAME IN ('cluster_id', 'worker_id')"
            )
        ):
            fk_names_to_drop.add(row[0])

    for name in fk_names_to_drop:
        op.drop_constraint(name, 'principals', type_='foreignkey')

    with op.batch_alter_table('principals', schema=None) as batch_op:
        if column_exists('principals', 'cluster_id'):
            batch_op.drop_column('cluster_id')
        if column_exists('principals', 'worker_id'):
            batch_op.drop_column('worker_id')
        if column_exists('principals', 'role'):
            batch_op.drop_column('role')
        if column_exists('principals', 'is_system'):
            batch_op.drop_column('is_system')

    # The ``userrole`` enum (Worker / Cluster) is now unreferenced. On
    # PG, leave the type in place — dropping an enum still referenced
    # by historical migrations would re-fail on a future ``alembic
    # downgrade``; an orphaned enum type is harmless.

    # ------------------------------------------------------------------
    # 18. Seed ``system/authenticated`` + backfill Default-Org grants.
    # ------------------------------------------------------------------
    # ``system/authenticated`` is the built-in GROUP principal that
    # ``_accessible_clusters`` treats as implicitly containing every
    # authenticated caller. A ``cluster_access`` row granting it is
    # therefore a global grant.
    #
    # Default-Org clusters default to "shared with all users" — the
    # ``create_cluster`` route now seeds that row at create time. Here
    # we cover the upgrade case: every Default-Org cluster that
    # pre-dates this revision gets the same row, idempotent via the
    # ``NOT EXISTS`` guard and the composite UNIQUE on
    # ``(cluster_id, principal_id)``.

    # 18a. Seed the GROUP principal. Lives in the GROUP name
    # partition (its own partial uniqueness index, set up above), so
    # it can't collide with the admin-curated USER / ORG / SYSTEM rows
    # in the other partition.
    op.execute(
        sa.text(
            f"""
            INSERT INTO principals
                (kind, name, display_name, description,
                 is_admin, is_active,
                 source,
                 created_at, updated_at, deleted_at)
            SELECT
                'GROUP', :name, :display_name, :desc,
                {bool_false}, {bool_true},
                'Local',
                :now_utc, :now_utc, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM principals
                WHERE kind = 'GROUP' AND name = :name
            )
            """
        ).bindparams(
            name=AUTHENTICATED_PRINCIPAL_NAME,
            display_name=AUTHENTICATED_PRINCIPAL_DISPLAY_NAME,
            desc=(
                'Built-in group containing every authenticated principal. '
                'Grant access to this principal to share a resource with all users.'
            ),
            now_utc=now_utc,
        )
    )

    authenticated_id = bind.execute(
        sa.text(
            "SELECT id FROM principals "
            "WHERE kind = 'GROUP' AND name = :name AND deleted_at IS NULL"
        ).bindparams(name=AUTHENTICATED_PRINCIPAL_NAME)
    ).scalar()
    if authenticated_id is None:
        raise RuntimeError(
            f"Authenticated principal not found by name="
            f"{AUTHENTICATED_PRINCIPAL_NAME!r} after seed; "
            "backfill cannot continue"
        )

    # 18b. Backfill ``cluster_access`` rows for every existing
    # Default-Org cluster. ``platform_id`` was resolved in step 3.
    op.execute(
        sa.text(
            """
            INSERT INTO cluster_access
                (cluster_id, principal_id, granted_by,
                 created_at, updated_at, deleted_at)
            SELECT c.id, :authenticated_id, NULL,
                   :now_utc, :now_utc, NULL
            FROM clusters c
            WHERE c.owner_principal_id = :platform_id
              AND c.deleted_at IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM cluster_access ca
                WHERE ca.cluster_id = c.id
                  AND ca.principal_id = :authenticated_id
              )
            """
        ).bindparams(
            authenticated_id=authenticated_id,
            platform_id=platform_id,
            now_utc=now_utc,
        )
    )


def downgrade() -> None:
    # Splitting principals back into a USER-only ``users`` table is not
    # mechanical: ORG/GROUP rows have ids in the same space as the
    # original users, and FK definitions on legacy tables (api_keys,
    # cluster_access, model_usages) silently retargeted ``principals``
    # during upgrade. A faithful inverse would have to move non-USER
    # rows out, restore those FK targets to ``users``, and renumber.
    # Operators who need to roll back must restore from a pre-upgrade
    # backup of the database.
    raise NotImplementedError(
        "downgrade of the multi-tenancy foundation is not supported; "
        "restore from a pre-upgrade database backup instead"
    )
