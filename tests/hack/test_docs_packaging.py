"""The packaged documentation includes current and future translation trees."""

import fnmatch
import shlex
from pathlib import Path

import pytest
import yaml

from hack.mkdocs_i18n import LANGUAGES


ROOT = Path(__file__).resolve().parents[2]
TRANSLATIONS = [language for language in LANGUAGES if language != "en"] + ["fr"]


@pytest.mark.parametrize("language", TRANSLATIONS)
@pytest.mark.parametrize(
    "relative",
    ["{language}/guide.md", "{language}/assets/example.svg", "mkdocs/{language}.yml"],
)
def test_translation_changes_trigger_packaging(language, relative):
    workflow = yaml.load(
        (ROOT / ".github/workflows/pack.yaml").read_text(), Loader=yaml.BaseLoader
    )
    patterns = workflow["on"]["pull_request"]["paths"]
    path = "docs-i18n/" + relative.format(language=language)

    assert any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


@pytest.mark.parametrize(
    "relative", ["docs/index.md", "mkdocs.yml", "mkdocs.offline.yml"]
)
def test_english_and_root_configuration_changes_still_trigger_packaging(relative):
    workflow = yaml.load(
        (ROOT / ".github/workflows/pack.yaml").read_text(), Loader=yaml.BaseLoader
    )
    patterns = workflow["on"]["pull_request"]["paths"]

    assert any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def test_docs_image_receives_all_language_sources_and_build_helpers():
    dockerfile = (ROOT / "pack/Dockerfile").read_text()
    stage = dockerfile.split("FROM base AS docs\n", 1)[1].split("\nFROM ", 1)[0]
    sources = [
        Path(source)
        for line in stage.splitlines()
        if line.startswith("COPY ")
        for source in shlex.split(line)[1:-1]
    ]
    required = [
        "docs/index.md",
        "mkdocs.yml",
        "mkdocs.offline.yml",
        "hack/build-docs.sh",
        "hack/build_docs.py",
        "hack/mkdocs_i18n.py",
        "hack/mkdocs_multilingual.py",
        "hack/mkdocs_offline.py",
        "hack/check_offline_docs.py",
    ]
    required.extend(
        f"docs-i18n/{relative.format(language=language)}"
        for language in TRANSLATIONS
        for relative in (
            "{language}/guide.md",
            "{language}/assets/example.svg",
            "mkdocs/{language}.yml",
        )
    )

    for relative in required:
        path = Path(relative)
        assert any(path == source or source in path.parents for source in sources), path
