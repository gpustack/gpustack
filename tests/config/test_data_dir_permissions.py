import os
import stat

from gpustack.config import registration
from gpustack.config.config import Config

CREDENTIAL_FILENAMES = (
    "jwt_secret_key",
    registration.registration_token_filename,
    registration.worker_token_filename,
    "initial_admin_password",
)


def file_mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def make_config(data_dir) -> Config:
    # jwt_secret_key is left unset so Config generates and persists it, the
    # path a fresh install takes.
    return Config(token="test", data_dir=str(data_dir))


def test_fresh_install_writes_credentials_owner_only(tmp_path):
    make_config(tmp_path)

    # 0711: other principals — the embedded PostgreSQL's uid above all —
    # keep traversal to reach <data_dir>/postgresql but lose listing; the
    # credential files themselves are what carry the secrecy.
    assert file_mode(tmp_path) == 0o711
    assert file_mode(tmp_path / "jwt_secret_key") == 0o600


def test_existing_loose_credentials_are_tightened_on_start(tmp_path):
    # A data dir whose credentials were left world-readable: every one of
    # them must come back owner-only after Config initialization.
    os.chmod(tmp_path, 0o755)
    for filename in CREDENTIAL_FILENAMES:
        credential = tmp_path / filename
        credential.write_text("credential")
        os.chmod(credential, 0o644)

    make_config(tmp_path)

    assert file_mode(tmp_path) == 0o711
    for filename in CREDENTIAL_FILENAMES:
        assert file_mode(tmp_path / filename) == 0o600, filename


def test_write_token_creates_owner_only_file(tmp_path):
    registration.write_token(str(tmp_path), "token", "gpustack_access_secret")

    path = tmp_path / "token"
    assert path.read_text() == "gpustack_access_secret\n"
    assert file_mode(path) == 0o600


def test_write_token_keeps_an_unchanged_token(tmp_path):
    registration.write_token(str(tmp_path), "token", "same-token")
    registration.write_token(str(tmp_path), "token", "same-token")

    assert (tmp_path / "token").read_text() == "same-token\n"
