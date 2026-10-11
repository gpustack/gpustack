import os
import stat

import pytest

from gpustack.utils.file import (
    CREDENTIAL_FILE_MODE,
    DATA_DIR_MODE,
    ensure_dir,
    restrict_permissions,
    write_credential_file,
)


def file_mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def permissive_umask():
    # 022 is the common default; under it an unforced create lands on
    # 0644/0755, which is what the explicit modes exist to prevent.
    previous = os.umask(0o022)
    yield
    os.umask(previous)


class TestWriteCredentialFile:
    def test_creates_the_file_owner_only(self, tmp_path, permissive_umask):
        path = tmp_path / "jwt_secret_key"

        write_credential_file(str(path), "secret")

        assert path.read_text() == "secret"
        assert file_mode(path) == CREDENTIAL_FILE_MODE

    def test_overwrites_content_and_loose_mode(self, tmp_path, permissive_umask):
        path = tmp_path / "token"
        path.write_text("old")
        os.chmod(path, 0o644)

        write_credential_file(str(path), "new\n")

        assert path.read_text() == "new\n"
        assert file_mode(path) == CREDENTIAL_FILE_MODE


class TestEnsureDir:
    def test_enforces_mode_despite_umask(self, tmp_path, permissive_umask):
        path = tmp_path / "var" / "lib" / "gpustack"

        ensure_dir(str(path), DATA_DIR_MODE)

        assert file_mode(path) == DATA_DIR_MODE

    def test_existing_world_listable_dir_is_tightened(self, tmp_path):
        path = tmp_path / "data"
        path.mkdir()
        os.chmod(path, 0o755)

        ensure_dir(str(path), DATA_DIR_MODE)

        assert file_mode(path) == DATA_DIR_MODE

    def test_stricter_umask_cannot_narrow_the_dir(self, tmp_path):
        # Under umask 077 the makedirs mode alone would drop the traversal
        # bits the embedded PostgreSQL depends on; the chmod reasserts them.
        path = tmp_path / "data"
        previous = os.umask(0o077)
        try:
            ensure_dir(str(path), DATA_DIR_MODE)
        finally:
            os.umask(previous)

        assert file_mode(path) == DATA_DIR_MODE


def test_restrict_permissions_swallows_failures(tmp_path):
    # Read-only mounts (the chart's bootstrap Secret) and vanished paths make
    # the chmod best-effort; raising would turn tightening into an outage.
    restrict_permissions(str(tmp_path / "absent"), CREDENTIAL_FILE_MODE)
