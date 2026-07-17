"""Bootstrap admin creation (``Server._init_user``).

Covers where the initial admin password comes from and whether the first-login
change is forced:
- a pre-existing file (HA: the Helm-mounted shared Secret; or a prior single
  node bootstrap) is read as-is and forces a change,
- with no file and no explicit password, one is generated, persisted, and
  forces a change,
- an explicit ``--bootstrap-password`` is used verbatim and is not forced.
"""

from types import SimpleNamespace

import pytest

from gpustack.server import server as server_mod
from gpustack.schemas.users import User

PASSWORD_FILE = "initial_admin_password"


@pytest.fixture
def captured(monkeypatch):
    """Stub out the DB writes ``_init_user`` performs and capture what password
    / force-change flag it would persist."""
    state = {"set_password_calls": 0}

    async def fake_first_by_fields(**kwargs):
        return state.get("existing_admin")

    async def fake_create(session, source, update=None, auto_commit=True):
        source.id = 1
        return source

    async def fake_set_password(
        session, user_id, password, require_password_change=False, auto_commit=True
    ):
        state["set_password_calls"] += 1
        state["password"] = password
        state["require_password_change"] = require_password_change

    async def fake_provision(session, user):
        pass

    monkeypatch.setattr(User, "first_by_fields", fake_first_by_fields)
    monkeypatch.setattr(User, "create", fake_create)
    monkeypatch.setattr(server_mod, "set_password", fake_set_password)
    monkeypatch.setattr(server_mod, "provision_bootstrap_admin_orgs", fake_provision)
    return state


def _server(tmp_path, bootstrap_password=None):
    config = SimpleNamespace(
        data_dir=str(tmp_path),
        bootstrap_password=bootstrap_password,
    )
    return SimpleNamespace(_config=config)


class _FakeSession:
    async def commit(self):
        pass


@pytest.mark.asyncio
async def test_reads_password_from_existing_file(tmp_path, captured):
    file = tmp_path / PASSWORD_FILE
    file.write_text("from-secret\n")

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["password"] == "from-secret"
    assert captured["require_password_change"] is True
    # The mounted / existing file is used as-is, never rewritten.
    assert file.read_text() == "from-secret\n"


@pytest.mark.asyncio
async def test_generates_and_persists_when_no_file(tmp_path, captured, monkeypatch):
    monkeypatch.setattr(server_mod, "generate_secure_password", lambda: "generated")

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["password"] == "generated"
    assert captured["require_password_change"] is True
    assert (tmp_path / PASSWORD_FILE).read_text().strip() == "generated"


@pytest.mark.asyncio
async def test_generates_when_file_is_empty(tmp_path, captured, monkeypatch):
    # An empty / whitespace-only file must not yield an empty admin password.
    (tmp_path / PASSWORD_FILE).write_text("   \n")
    monkeypatch.setattr(server_mod, "generate_secure_password", lambda: "generated")

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["password"] == "generated"
    assert captured["require_password_change"] is True
    assert (tmp_path / PASSWORD_FILE).read_text().strip() == "generated"


@pytest.mark.asyncio
async def test_survives_unreadable_password_file(tmp_path, captured, monkeypatch):
    # A path that cannot be read or written as a file (here a directory) must
    # not crash bootstrap; fall back to a generated password.
    (tmp_path / PASSWORD_FILE).mkdir()
    monkeypatch.setattr(server_mod, "generate_secure_password", lambda: "generated")

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["password"] == "generated"
    assert captured["require_password_change"] is True


@pytest.mark.asyncio
async def test_generates_when_file_is_invalid_utf8(tmp_path, captured, monkeypatch):
    # A corrupted / non-UTF-8 file raises UnicodeDecodeError on read; it must
    # not crash bootstrap.
    (tmp_path / PASSWORD_FILE).write_bytes(b"\xff\xfe\x00bad")
    monkeypatch.setattr(server_mod, "generate_secure_password", lambda: "generated")

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["password"] == "generated"
    assert captured["require_password_change"] is True


@pytest.mark.asyncio
async def test_explicit_password_is_not_forced_and_writes_no_file(tmp_path, captured):
    await server_mod.Server._init_user(
        _server(tmp_path, bootstrap_password="operator-chosen"), _FakeSession()
    )

    assert captured["password"] == "operator-chosen"
    assert captured["require_password_change"] is False
    assert not (tmp_path / PASSWORD_FILE).exists()


@pytest.mark.asyncio
async def test_skips_when_admin_already_exists(tmp_path, captured):
    captured["existing_admin"] = User(id=1, name="admin", is_admin=True)

    await server_mod.Server._init_user(_server(tmp_path), _FakeSession())

    assert captured["set_password_calls"] == 0
    assert not (tmp_path / PASSWORD_FILE).exists()
