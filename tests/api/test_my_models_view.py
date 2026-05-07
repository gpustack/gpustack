"""Integration test for the non_admin_user_models view's principal logic.

This test exercises the actual SQL view against a real Postgres database. It
is opt-in via the GPUSTACK_PG_TEST_URL env var:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15
    GPUSTACK_PG_TEST_URL=postgresql://postgres:postgres@localhost:5432 \
      pytest tests/api/test_my_models_view.py -v

Without that env var, the test is skipped.
"""

import os
import secrets
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, text


PG_BASE_URL = os.environ.get("GPUSTACK_PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_BASE_URL,
    reason="GPUSTACK_PG_TEST_URL not set; skipping real-DB view test",
)


@pytest.fixture
def fresh_db():
    """Create a unique DB, run migrations, hand back a sync engine."""
    db_name = f"gpustack_p4_view_{secrets.token_hex(4)}"
    admin = create_engine(f"{PG_BASE_URL}/postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    admin.dispose()

    db_url = f"{PG_BASE_URL}/{db_name}"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = {**os.environ, "DATABASE_URL": db_url}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
    )

    engine = create_engine(db_url)
    try:
        yield engine
    finally:
        engine.dispose()
        admin = create_engine(f"{PG_BASE_URL}/postgres", isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": db_name},
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
        admin.dispose()


def _seed(engine):
    """Seed minimal data for the visibility matrix.

    Returns ids: (alice, bob, carol, org1=platform, org2, group_alice_bob,
    route_public, route_authed, route_user, route_org, route_group).

    alice: in platform Org via membership; member of group_alice_bob.
    bob:   in platform Org via membership; member of group_alice_bob.
    carol: in org2 only.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO users (username, hashed_password, is_admin,
                    is_active, is_system, source, require_password_change,
                    created_at, updated_at)
                VALUES
                    ('alice','x',false,true,false,'Local',false,NOW(),NOW()),
                    ('bob','x',false,true,false,'Local',false,NOW(),NOW()),
                    ('carol','x',false,true,false,'Local',false,NOW(),NOW())
                """
            )
        )
        ids = conn.execute(
            text(
                "SELECT id, username FROM users "
                "WHERE username IN ('alice','bob','carol') ORDER BY id"
            )
        ).fetchall()
        alice = next(r.id for r in ids if r.username == "alice")
        bob = next(r.id for r in ids if r.username == "bob")
        carol = next(r.id for r in ids if r.username == "carol")

        # Org 1 (platform) was seeded by the foundation migration. Add Org 2.
        conn.execute(
            text(
                "INSERT INTO organizations (id, name, slug, "
                "created_at, updated_at) VALUES "
                "(2, 'Acme', 'acme', NOW(), NOW())"
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                VALUES
                    (:a, 1, 'MEMBER', NOW()),
                    (:b, 1, 'MEMBER', NOW()),
                    (:c, 2, 'MEMBER', NOW())
                """
            ),
            {"a": alice, "b": bob, "c": carol},
        )

        # Group in platform Org with alice + bob.
        group_id = conn.execute(
            text(
                "INSERT INTO user_groups (organization_id, name, created_at, "
                "updated_at) VALUES (1, 'team-a', NOW(), NOW()) RETURNING id"
            )
        ).scalar()
        conn.execute(
            text(
                """
                INSERT INTO user_group_memberships (user_id, group_id, created_at)
                VALUES (:a, :g, NOW()), (:b, :g, NOW())
                """
            ),
            {"a": alice, "b": bob, "g": group_id},
        )

        # Five routes covering every visibility branch. Each lives in
        # platform Org, owner=org/1.
        def _ins_route(name, policy):
            return conn.execute(
                text(
                    f"""
                    INSERT INTO model_routes
                        (name, access_policy, organization_id, created_by_model,
                         targets, ready_targets, generic_proxy,
                         categories, meta,
                         created_at, updated_at)
                    VALUES (:n, '{policy}', 1, false, 0, 0, false,
                            '[]'::jsonb, '{{}}'::jsonb, NOW(), NOW())
                    RETURNING id
                    """
                ),
                {"n": name},
            ).scalar()

        r_public = _ins_route("r-public", "PUBLIC")
        r_authed = _ins_route("r-authed", "AUTHED")
        r_user = _ins_route("r-user", "ALLOWED_PRINCIPALS")
        r_org = _ins_route("r-org", "ALLOWED_PRINCIPALS")
        r_group = _ins_route("r-group", "ALLOWED_PRINCIPALS")

        # Principals
        conn.execute(
            text(
                "INSERT INTO model_route_principals (route_id, principal_type, principal_id) "
                "VALUES (:r, 'USER', :u)"
            ),
            {"r": r_user, "u": alice},
        )
        conn.execute(
            text(
                "INSERT INTO model_route_principals (route_id, principal_type, principal_id) "
                "VALUES (:r, 'ORG', 2)"
            ),
            {"r": r_org},  # granted to Org Acme (carol)
        )
        conn.execute(
            text(
                "INSERT INTO model_route_principals (route_id, principal_type, principal_id) "
                "VALUES (:r, 'GROUP', :g)"
            ),
            {"r": r_group, "g": group_id},  # granted to group team-a (alice + bob)
        )

    return {
        "alice": alice,
        "bob": bob,
        "carol": carol,
        "r_public": r_public,
        "r_authed": r_authed,
        "r_user": r_user,
        "r_org": r_org,
        "r_group": r_group,
    }


def _visible_route_ids(engine, user_id: int):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id FROM non_admin_user_models WHERE user_id = :u ORDER BY id"),
            {"u": user_id},
        ).fetchall()
    return {r.id for r in rows}


def test_view_visibility_matrix(fresh_db):
    ids = _seed(fresh_db)

    alice_visible = _visible_route_ids(fresh_db, ids["alice"])
    # alice: PUBLIC + AUTHED + USER (her) + GROUP (team-a). NOT r_org (Acme).
    assert alice_visible == {
        ids["r_public"],
        ids["r_authed"],
        ids["r_user"],
        ids["r_group"],
    }

    bob_visible = _visible_route_ids(fresh_db, ids["bob"])
    # bob: PUBLIC + AUTHED + GROUP (team-a). NOT r_user (alice only).
    assert bob_visible == {
        ids["r_public"],
        ids["r_authed"],
        ids["r_group"],
    }

    carol_visible = _visible_route_ids(fresh_db, ids["carol"])
    # carol: PUBLIC + AUTHED + ORG (Acme). NOT r_user, NOT r_group.
    assert carol_visible == {
        ids["r_public"],
        ids["r_authed"],
        ids["r_org"],
    }
