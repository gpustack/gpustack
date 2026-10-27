"""Japanese sources retain executable examples, destinations, and deep links."""

import re
from pathlib import Path

import markdown
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
ENGLISH = ROOT / "docs"
JAPANESE = ROOT / "docs-i18n/ja"
PAGES = sorted(path.relative_to(ENGLISH) for path in ENGLISH.rglob("*.md"))


def document_body(source):
    return re.sub(r"\A---\n.*?\n---\n", "", source, flags=re.DOTALL)


def code_blocks(source):
    blocks = []
    current = []
    fence = None
    for line in document_body(source).splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            if fence is None:
                fence = stripped[:3]
                current = [line]
            elif stripped.startswith(fence):
                blocks.append("\n".join(current + [line]))
                fence = None
            else:
                current.append(line)
        elif fence:
            current.append(line)
    assert fence is None, "unterminated fenced code example"
    return blocks


def heading_ids(source):
    renderer = markdown.Markdown(
        extensions=["toc", "attr_list", "pymdownx.superfences", "md_in_html"]
    )
    rendered = renderer.convert(document_body(source))
    return re.findall(r'<h[1-6]\s+id="([^"]+)"', rendered)


def test_japanese_covers_all_english_sources():
    assert set(PAGES) == {path.relative_to(JAPANESE) for path in JAPANESE.rglob("*.md")}


@pytest.mark.parametrize("relative", PAGES, ids=str)
def test_japanese_preserves_document_contract(relative):
    english = (ENGLISH / relative).read_text(encoding="utf-8")
    japanese = (JAPANESE / relative).read_text(encoding="utf-8")
    if relative.as_posix() == "index.md":
        assert japanese == english
        return
    assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", japanese)
    if japanese.startswith("---\n"):
        metadata = yaml.safe_load(japanese.split("---", 2)[1]) or {}
        assert metadata.get("translation_status") != "pending"
    assert code_blocks(english) == code_blocks(japanese)
    destinations = r'\]\(([^)]+)\)|(?:href|src)=["\']([^"\']+)["\']'
    assert re.findall(destinations, english) == re.findall(destinations, japanese)
    assert heading_ids(english) == heading_ids(japanese)
