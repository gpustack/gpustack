from gpustack.config.config import Config


def test_is_both_role_false_when_data_dir_missing(tmp_path):
    # data_dir does not exist yet -> fresh install -> server-only
    cfg = Config(data_dir=str(tmp_path / "nonexistent"))

    assert cfg._data_dir_was_fresh is True
    assert cfg._is_both_role() is False


def test_is_both_role_false_when_data_dir_empty(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = Config(data_dir=str(data_dir))

    assert cfg._data_dir_was_fresh is True
    assert cfg._is_both_role() is False


def test_is_both_role_false_when_bootstrap_version_present(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bootstrap_version").write_text("2.0.1")

    cfg = Config(data_dir=str(data_dir))

    assert cfg._data_dir_was_fresh is False
    assert cfg._is_both_role() is False


def test_is_both_role_true_for_legacy_data_dir(tmp_path):
    # Non-empty data_dir without bootstrap_version simulates a v2.0.0 install.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "legacy_marker").write_text("")

    cfg = Config(data_dir=str(data_dir))

    assert cfg._data_dir_was_fresh is False
    assert cfg._is_both_role() is True
