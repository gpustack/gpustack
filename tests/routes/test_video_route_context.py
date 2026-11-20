"""Video jobs use the same route and tenant partition as the model list."""

import sqlite3
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import sqlite

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes.openai import _assert_video_route_access
from gpustack.schemas.model_routes import AccessPolicyEnum, ModelRoute


@pytest.fixture
def scope_db():
    database = sqlite3.connect(":memory:")
    database.executescript(
        "CREATE TABLE non_admin_user_models (id INTEGER, user_id INTEGER, "
        "via_principal_id INTEGER, via_principal_kind TEXT, deleted_at TEXT);"
        "CREATE TABLE model_routes (id INTEGER, owner_principal_id INTEGER, "
        "access_policy TEXT, deleted_at TEXT);"
        "CREATE TABLE model_route_principals (id INTEGER, route_id INTEGER, "
        "principal_id INTEGER, deleted_at TEXT);"
    )

    class Session:
        async def exec(self, statement):
            sql = str(
                statement.compile(
                    dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}
                )
            )
            cursor = database.execute(sql)
            return SimpleNamespace(first=cursor.fetchone)

    yield database, Session()
    database.close()


def _context(admin=False, personal=False):
    user = SimpleNamespace(id=11, is_admin=admin)
    ctx = TenantContext(
        user=user,
        is_platform_admin=admin,
        current_principal_id=11 if personal else 7,
        org_role=None,
        current_is_personal_scope=personal,
    )
    return user, ctx


async def _check(session, user, ctx, api_key=None, route_id=9):
    await _assert_video_route_access(
        session, "published-wan", route_id, user, ctx, api_key
    )


@pytest.mark.asyncio
async def test_org_grant_revocation_is_not_masked_by_another_org(scope_db):
    database, session = scope_db
    user, ctx = _context()
    database.executemany(
        "INSERT INTO non_admin_user_models VALUES (?, 11, ?, 'ORG', NULL)",
        [(9, 7), (9, 8), (10, 8)],
    )
    await _check(session, user, ctx)
    database.execute("DELETE FROM non_admin_user_models WHERE via_principal_id = 7")
    with pytest.raises(NotFoundException):
        await _check(session, user, ctx)
    ctx.current_principal_id = 8
    await _check(session, user, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "personal,grant,kind,expected",
    [
        (False, 7, "ORG", True),
        (False, 8, "ORG", False),
        (False, 11, "USER", False),
        (False, None, None, True),
        (True, 7, "ORG", False),
        (True, 11, "USER", True),
        (True, 12, "GROUP", True),
        (True, None, None, True),
    ],
)
async def test_personal_and_org_grants_stay_in_their_context(
    scope_db, personal, grant, kind, expected
):
    database, session = scope_db
    user, ctx = _context(personal=personal)
    database.execute(
        "INSERT INTO non_admin_user_models VALUES (9, 11, ?, ?, NULL)", (grant, kind)
    )
    if expected:
        await _check(session, user, ctx)
    else:
        with pytest.raises(NotFoundException):
            await _check(session, user, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [None, [], ["published-wan"], ["another-model"]])
async def test_api_key_scope_is_applied_to_the_job_route(scope_db, allowed):
    database, session = scope_db
    user, ctx = _context()
    database.execute("INSERT INTO non_admin_user_models VALUES (9, 11, 7, 'ORG', NULL)")
    api_key = SimpleNamespace(allowed_model_names=allowed)
    if allowed == ["another-model"]:
        with pytest.raises(NotFoundException):
            await _check(session, user, ctx, api_key)
    else:
        await _check(session, user, ctx, api_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_user,deleted", [(True, False), (False, True)])
async def test_other_users_and_deleted_routes_do_not_authorize_jobs(
    scope_db, wrong_user, deleted
):
    database, session = scope_db
    user, ctx = _context()
    database.execute(
        "INSERT INTO non_admin_user_models VALUES (9, ?, 7, 'ORG', ?)",
        (12 if wrong_user else 11, "2026-01-01" if deleted else None),
    )
    with pytest.raises(NotFoundException):
        await _check(session, user, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["owner", "grant", "public", "other", "deleted"])
async def test_admin_act_as_keeps_the_current_org_scope(scope_db, mode):
    database, session = scope_db
    user, ctx = _context(admin=True)
    policy = (
        AccessPolicyEnum.PUBLIC
        if mode == "public"
        else AccessPolicyEnum.ALLOWED_PRINCIPALS
    )
    bind = ModelRoute.__table__.c.access_policy.type.bind_processor(sqlite.dialect())
    database.execute(
        "INSERT INTO model_routes VALUES (9, ?, ?, ?)",
        (
            7 if mode == "owner" else 8,
            bind(policy),
            "2026-01-01" if mode == "deleted" else None,
        ),
    )
    if mode == "grant":
        database.execute("INSERT INTO model_route_principals VALUES (1, 9, 7, NULL)")
    if mode in ("other", "deleted"):
        with pytest.raises(NotFoundException):
            await _check(session, user, ctx)
    else:
        await _check(session, user, ctx)
