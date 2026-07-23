"""``/auth/config`` first-login guide.

The guide is keyed off the shared DB ``require_password_change`` flag (so every
replica reports it consistently behind a load balancer), and the retrieval
command is only advertised while the password file is still present.
"""

import errno
import logging
from types import SimpleNamespace

import pytest

from gpustack.routes import auth as auth_mod
from gpustack.schemas.users import User

PASSWORD_FILE = "initial_admin_password"


def _request(tmp_path):
    config = SimpleNamespace(
        external_auth_type=None,
        data_dir=str(tmp_path),
        enable_login_captcha=False,
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(server_config=config))
    )


@pytest.fixture
def stub(monkeypatch):
    state = {"admin": User(id=1, name="admin", is_admin=True), "require_change": True}

    async def fake_first_by_fields(**kwargs):
        state["queried_fields"] = kwargs.get("fields")
        return state["admin"]

    async def fake_is_password_change_required(session, principal_id):
        return state["require_change"]

    monkeypatch.setattr(User, "first_by_fields", fake_first_by_fields)
    monkeypatch.setattr(
        auth_mod, "is_password_change_required", fake_is_password_change_required
    )
    return state


@pytest.mark.asyncio
async def test_guide_shown_when_change_required_and_file_present(tmp_path, stub):
    (tmp_path / PASSWORD_FILE).write_text("pw\n")

    result = await auth_mod.get_auth_config(_request(tmp_path), session=object())

    assert result["first_time_setup"] is True
    assert str(tmp_path / PASSWORD_FILE) in result["get_initial_password_command"]
    # Scoped to the default bootstrap admin, not just any admin.
    assert stub["queried_fields"]["name"] == "admin"
    assert stub["queried_fields"]["is_admin"] is True


@pytest.mark.asyncio
async def test_guide_hidden_when_change_not_required(tmp_path, stub):
    (tmp_path / PASSWORD_FILE).write_text("pw\n")
    stub["require_change"] = False

    result = await auth_mod.get_auth_config(_request(tmp_path), session=object())

    assert "first_time_setup" not in result
    assert "get_initial_password_command" not in result


@pytest.mark.asyncio
async def test_guide_hidden_when_no_admin(tmp_path, stub):
    (tmp_path / PASSWORD_FILE).write_text("pw\n")
    stub["admin"] = None

    result = await auth_mod.get_auth_config(_request(tmp_path), session=object())

    assert "first_time_setup" not in result


@pytest.mark.asyncio
async def test_guide_hidden_when_file_missing(tmp_path, stub):
    # Change still required but the password is no longer retrievable.
    result = await auth_mod.get_auth_config(_request(tmp_path), session=object())

    assert "first_time_setup" not in result


@pytest.mark.asyncio
async def test_no_session_skips_first_login_lookup(tmp_path):
    # Direct (non-FastAPI) callers pass session=None; the external-auth config
    # must still return without touching the DB.
    result = await auth_mod.get_auth_config(_request(tmp_path), session=None)

    assert result["external_auth"] is None
    assert "first_time_setup" not in result


def test_remove_password_file_unremovable_is_not_an_error(
    tmp_path, monkeypatch, caplog
):
    # An un-removable file (e.g. a read-only shared password mount) is left in
    # place without raising or warning: guide visibility is driven by the DB
    # require_password_change flag, not the file's presence.
    (tmp_path / PASSWORD_FILE).write_text("pw\n")

    def failing_unlink(self, *args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr("pathlib.Path.unlink", failing_unlink)
    config = SimpleNamespace(data_dir=str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger=auth_mod.logger.name):
        auth_mod.remove_initial_password_file_if_exists(config)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_command_uses_kubectl_in_kubernetes(tmp_path, stub, monkeypatch):
    (tmp_path / PASSWORD_FILE).write_text("pw\n")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("HOSTNAME", "gpustack-server-1")

    result = await auth_mod.get_auth_config(_request(tmp_path), session=object())

    command = result["get_initial_password_command"]
    assert command.startswith("kubectl exec gpustack-server-1")
    assert str(tmp_path / PASSWORD_FILE) in command
