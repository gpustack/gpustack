import argparse
from types import SimpleNamespace

import pytest

from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.cmd.local_auth import (
    mint_admin_jwt,
    read_local_jwt_secret,
)
from gpustack.cmd import reload_config
from gpustack.cmd.reload_config import apply_runtime_updates, resolve_scope_headers
from gpustack.security import JWTManager


def _args(data_dir, api_key=None, jwt_secret_key=None, admin_username="admin"):
    return argparse.Namespace(
        data_dir=str(data_dir),
        api_key=api_key,
        jwt_secret_key=jwt_secret_key,
        admin_username=admin_username,
    )


# ---------------------------------------------------------------------------
# read_local_jwt_secret
# ---------------------------------------------------------------------------


def test_read_local_jwt_secret_prefers_explicit_flag(tmp_path):
    (tmp_path / "jwt_secret_key").write_text("from-file")
    secret = read_local_jwt_secret(_args(tmp_path, jwt_secret_key="explicit"))
    assert secret == "explicit"


def test_read_local_jwt_secret_reads_file(tmp_path):
    (tmp_path / "jwt_secret_key").write_text("file-secret\n")
    assert read_local_jwt_secret(_args(tmp_path)) == "file-secret"


def test_read_local_jwt_secret_none_when_missing(tmp_path):
    assert read_local_jwt_secret(_args(tmp_path)) is None


def test_read_local_jwt_secret_none_when_empty_file(tmp_path):
    (tmp_path / "jwt_secret_key").write_text("   \n")
    assert read_local_jwt_secret(_args(tmp_path)) is None


# ---------------------------------------------------------------------------
# resolve_scope_headers — role / credential resolution
# ---------------------------------------------------------------------------


def test_legacy_combined_node_targets_both_endpoints(tmp_path):
    # Regression for the role-misdetection bug: a legacy (<= v2.0.0) combined
    # server+worker node has a worker_token and a jwt_secret_key but NO
    # bootstrap_version file. Both endpoints must still be targeted; the server
    # reload must not be silently skipped.
    (tmp_path / "jwt_secret_key").write_text("jwt-secret")
    (tmp_path / "worker_token").write_text("wt-token")
    # deliberately no bootstrap_version

    headers = resolve_scope_headers(_args(tmp_path))

    assert "server" in headers
    assert "worker" in headers
    assert headers["server"]["Cookie"].startswith(f"{SESSION_COOKIE_NAME}=")
    assert headers["worker"]["Authorization"] == "Bearer wt-token"


def test_server_only_node_targets_server_with_jwt(tmp_path):
    (tmp_path / "jwt_secret_key").write_text("jwt-secret")
    (tmp_path / "bootstrap_version").write_text("2.1.1")
    # no worker_token

    headers = resolve_scope_headers(_args(tmp_path))

    assert "server" in headers
    assert "worker" not in headers


def test_worker_token_preferred_over_api_key(tmp_path):
    # worker_auth does not accept an admin API key, so a present local worker
    # token must win for the worker scope.
    (tmp_path / "worker_token").write_text("wt-token")

    headers = resolve_scope_headers(_args(tmp_path, api_key="ak-123"))

    assert headers["server"]["Authorization"] == "Bearer ak-123"
    assert headers["worker"]["Authorization"] == "Bearer wt-token"


def test_api_key_falls_back_for_worker_when_no_token(tmp_path):
    headers = resolve_scope_headers(_args(tmp_path, api_key="ak-123"))

    assert headers["server"]["Authorization"] == "Bearer ak-123"
    assert headers["worker"]["Authorization"] == "Bearer ak-123"


def test_no_credentials_yields_empty(tmp_path):
    assert resolve_scope_headers(_args(tmp_path)) == {}


# ---------------------------------------------------------------------------
# apply_runtime_updates — endpoint fallback behavior
# ---------------------------------------------------------------------------


def test_apply_runtime_updates_ignores_unauthorized_endpoint_after_worker_success(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        reload_config,
        "resolve_scope_headers",
        lambda args: {
            "server": {"Authorization": "Bearer admin"},
            "worker": {"Authorization": "Bearer worker"},
        },
    )

    def fake_put(url, json, headers, timeout):
        calls.append((url, headers))
        if ":30080" in url:
            return SimpleNamespace(status_code=401)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(reload_config.requests, "put", fake_put)

    apply_runtime_updates({"debug": True}, argparse.Namespace())

    assert len(calls) == 2


def test_apply_runtime_updates_still_fails_when_only_endpoint_is_unauthorized(
    monkeypatch,
):
    monkeypatch.setattr(
        reload_config,
        "resolve_scope_headers",
        lambda args: {"server": {"Authorization": "Bearer admin"}},
    )
    monkeypatch.setattr(
        reload_config.requests,
        "put",
        lambda url, json, headers, timeout: SimpleNamespace(status_code=401),
    )

    with pytest.raises(Exception, match="HTTP 401"):
        apply_runtime_updates({"debug": True}, argparse.Namespace())


# ---------------------------------------------------------------------------
# mint_admin_jwt — server accepts the minted token
# ---------------------------------------------------------------------------


def test_minted_jwt_decodes_with_same_secret():
    token = mint_admin_jwt("shared-secret", "admin")
    payload = JWTManager(secret_key="shared-secret").decode_jwt_token(token)
    assert payload["sub"] == "admin"


def test_minted_jwt_uses_admin_username_override(tmp_path):
    (tmp_path / "jwt_secret_key").write_text("jwt-secret")
    headers = resolve_scope_headers(_args(tmp_path, admin_username="root"))

    cookie = headers["server"]["Cookie"]
    token = cookie.split("=", 1)[1]
    payload = JWTManager(secret_key="jwt-secret").decode_jwt_token(token)
    assert payload["sub"] == "root"
