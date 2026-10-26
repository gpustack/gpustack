from pathlib import Path

import pytest

from gpustack.utils.file import getsize


def _cyclic_link(tmp_path: Path, kind: str) -> Path:
    first = tmp_path / "first"
    if kind == "self":
        first.symlink_to("first")
        return first
    second = tmp_path / "second"
    first.symlink_to("second")
    second.symlink_to("first")
    return second if kind == "mutual_second" else first


@pytest.mark.parametrize("kind", ["self", "mutual", "mutual_second"])
def test_getsize_unresolvable_link_reports_missing_path(tmp_path, kind):
    link = _cyclic_link(tmp_path, kind)

    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        getsize(str(link))


@pytest.mark.parametrize("kind", ["self", "mutual"])
def test_getsize_skips_nested_unresolvable_links(tmp_path, kind):
    (tmp_path / "weights").write_bytes(b"weights")
    _cyclic_link(tmp_path, kind)

    assert getsize(str(tmp_path)) == 7


@pytest.mark.parametrize("alias", [False, True])
def test_getsize_regular_file_and_alias(tmp_path, alias):
    path = tmp_path / "weights"
    path.write_bytes(b"weights")
    if alias:
        link = tmp_path / "alias"
        link.symlink_to(path)
        path = link

    assert getsize(str(path)) == 7


@pytest.mark.parametrize("dangling", [False, True])
def test_getsize_missing_path_and_dangling_link(tmp_path, dangling):
    path = tmp_path / "missing"
    if dangling:
        link = tmp_path / "dangling"
        link.symlink_to(path)
        path = link

    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        getsize(str(path))


def test_getsize_skips_nested_dangling_link(tmp_path):
    (tmp_path / "weights").write_bytes(b"weights")
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")

    assert getsize(str(tmp_path)) == 7


def test_getsize_directory_link_to_ancestor(tmp_path):
    (tmp_path / "weights").write_bytes(b"weights")
    child = tmp_path / "child"
    child.mkdir()
    (child / "ancestor").symlink_to(tmp_path, target_is_directory=True)

    assert getsize(str(tmp_path)) == 7


def test_getsize_counts_repeated_directory_alias_once(tmp_path):
    directory = tmp_path / "payload"
    directory.mkdir()
    (directory / "weights").write_bytes(b"weights")
    (tmp_path / "alias_a").symlink_to(directory, target_is_directory=True)
    (tmp_path / "alias_b").symlink_to(directory, target_is_directory=True)

    assert getsize(str(tmp_path)) == 7


def test_getsize_counts_file_alias_separately(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"weights")
    (tmp_path / "alias").symlink_to(path)

    assert getsize(str(tmp_path)) == 14


def test_getsize_empty_directory(tmp_path):
    assert getsize(str(tmp_path)) == 0


@pytest.mark.parametrize("nested", [False, True])
def test_getsize_rejects_cycle_with_collapsed_path_components(tmp_path, nested):
    (tmp_path / "payload").write_bytes(b"weights")
    first = tmp_path / "first"
    first.symlink_to("second/../payload")
    (tmp_path / "second").symlink_to("first")

    if nested:
        assert getsize(str(tmp_path)) == 7
    else:
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            getsize(str(first))


def test_getsize_valid_relative_link_with_parent_component(tmp_path):
    (tmp_path / "weights").write_bytes(b"weights")
    child = tmp_path / "child"
    child.mkdir()
    alias = child / "alias"
    alias.symlink_to("../weights")

    assert getsize(str(alias)) == 7
    assert getsize(str(tmp_path)) == 14
