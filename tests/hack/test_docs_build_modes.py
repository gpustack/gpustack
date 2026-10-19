"""Keep complete multilingual checkouts and prepared single-language builds usable."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hack.mkdocs_i18n import LANGUAGES


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_source(tmp_path):
    root = tmp_path / "source"
    (root / "hack").mkdir(parents=True)
    shutil.copy2(ROOT / "hack/build-docs.sh", root / "hack/build-docs.sh")
    shutil.copy2(ROOT / "hack/serve-docs.sh", root / "hack/serve-docs.sh")
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCS_TEST_LOG'], 'a') as output:\n"
        "    output.write(json.dumps(args) + '\\n')\n"
        "if '--site-dir' in args:\n"
        "    site = pathlib.Path(args[args.index('--site-dir') + 1])\n"
        "    (site / 'search').mkdir(parents=True)\n"
        "    (site / 'search/search_index.json').write_text('{}')\n"
    )
    uv.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = dict(os.environ, DOCS_COMMIT="abcdef1234567", DOCS_TEST_LOG=str(log))
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    return root, env, log


def invoke(root, env, *args):
    return subprocess.run(
        ["bash", str(root / "hack/build-docs.sh"), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )


def test_prepared_single_language_build_needs_no_bilingual_helpers(build_source):
    root, env, log = build_source

    result = invoke(root, env, "--offline")

    assert result.returncode == 0, result.stderr
    assert (root / "gpustack/help/search/search_index.json.gz").is_file()
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("mkdocs" in args and "--site-dir" in args for args in commands)
    assert not any("--group" in args for args in commands)


@pytest.mark.parametrize("mode", ["--offline", "offline"])
@pytest.mark.parametrize(
    "language", [code for code in LANGUAGES if code != "en"] + ["fr"]
)
def test_complete_source_dispatches_all_languages(build_source, mode, language):
    root, env, log = build_source
    (root / f"docs-i18n/{language}").mkdir(parents=True)

    result = invoke(root, env, mode)

    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    assert args[:4] == ["run", "--group", "docs", "python"]
    assert args[4] == str(root / "hack/build_docs.py")
    assert args[5:7] == ["--offline", "--stamp"]
    assert args[7].endswith("(abcdef1)")


def test_unidentified_offline_build_stops_before_invoking_builder(build_source):
    root, env, log = build_source
    (root / "docs-i18n/zh").mkdir(parents=True)
    env["DOCS_COMMIT"] = "unknown"

    result = invoke(root, env, "--offline")

    assert result.returncode != 0
    assert "cannot determine the commit" in result.stderr
    assert not log.exists()


@pytest.mark.parametrize("multilingual", [False, True])
def test_preview_dispatches_nested_translations_and_prepared_sources(
    build_source, multilingual
):
    root, env, log = build_source
    if multilingual:
        (root / "docs-i18n/fr").mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(root / "hack/serve-docs.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    assert args == (
        [
            "run",
            "--group",
            "docs",
            "python",
            str(root / "hack/build_docs.py"),
            "--serve",
        ]
        if multilingual
        else ["run", "mkdocs", "serve", "--livereload"]
    )
