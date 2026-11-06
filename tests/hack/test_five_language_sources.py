"""New translations retain every executable example and deep-link contract."""

import re
from collections import Counter
from pathlib import Path

import markdown
import pytest
import yaml

from hack import mkdocs_i18n as i18n


ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted(
    path.relative_to(ROOT / "docs") for path in (ROOT / "docs").rglob("*.md")
)


def body(source):
    return re.sub(r"\A---\n.*?\n---\n", "", source, flags=re.S)


def code_blocks(source):
    return re.findall(
        r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", body(source), re.M | re.S
    )


def heading_ids(source):
    renderer = markdown.Markdown(
        extensions=["toc", "attr_list", "pymdownx.superfences", "md_in_html"]
    )
    return re.findall(r'<h[1-6]\s+id="([^"]+)"', renderer.convert(body(source)))


@pytest.mark.parametrize("language", ["ru", "tr"])
def test_complete_source_inventory(language):
    assert set(PAGES) == {
        path.relative_to(ROOT / f"docs-i18n/{language}")
        for path in (ROOT / f"docs-i18n/{language}").rglob("*.md")
    }


@pytest.mark.parametrize("language", ["ru", "tr"])
@pytest.mark.parametrize("relative", PAGES, ids=str)
def test_translated_document_contract(language, relative):
    english = (ROOT / "docs" / relative).read_text()
    translated = (ROOT / f"docs-i18n/{language}" / relative).read_text()
    if relative.as_posix() == "index.md":
        assert translated == english
        return
    assert body(translated) != body(english)
    assert not re.search(r"⟪\d+⟫|⟦\d+⟧", translated)
    metadata = (
        yaml.safe_load(translated.split("---", 2)[1])
        if translated.startswith("---\n")
        else {}
    )
    if relative.parts[0] == "includes":
        assert not metadata, "auto-appended snippets must not contain page metadata"
    else:
        assert metadata.get("translation_status") == "translated"
    assert code_blocks(english) == code_blocks(translated)
    english_prose = body(english)
    translated_prose = body(translated)
    for block in code_blocks(english):
        english_prose = english_prose.replace(block, "")
    for block in code_blocks(translated):
        translated_prose = translated_prose.replace(block, "")

    def inline_code(text):
        return Counter(re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text))

    assert not (inline_code(english_prose) - inline_code(translated_prose))
    assert re.findall(r"\$\$.*?\$\$", english_prose, re.S) == re.findall(
        r"\$\$.*?\$\$", translated_prose, re.S
    )
    assert heading_ids(english) == heading_ids(translated)
    destinations = r'\]\(([^)]+)\)|(?:href|src)=["\']([^"\']+)["\']'
    assert re.findall(destinations, english) == re.findall(destinations, translated)


@pytest.mark.parametrize("language", ["en", "zh", "ja", "ru", "tr"])
def test_auto_appended_abbreviations_render_without_metadata(language):
    source = ROOT / ("docs" if language == "en" else f"docs-i18n/{language}")
    snippet = source / "includes/abbreviations.md"
    renderer = markdown.Markdown(
        extensions=["abbr", "toc", "pymdownx.snippets"],
        extension_configs={"pymdownx.snippets": {"auto_append": [str(snippet)]}},
    )
    rendered = renderer.convert("# Guide\n\nLLM\n")

    assert '<abbr title="' in rendered
    assert ">LLM</abbr>" in rendered
    assert "translation_status" not in rendered
    assert re.findall(r"<h[1-6]\b", rendered) == ["<h1"]
    assert "<hr" not in rendered


@pytest.mark.parametrize("language", ["zh", "ja", "ru", "tr"])
def test_callout_titles_preserve_code_and_explicit_titles(language):
    source = (
        '!!! note\n    Read this.\n\n```markdown\n!!! note\n```\n\n'
        '???+ warning\n    Be careful.\n\n!!! tip "Custom title"\n'
    )
    translated = i18n.localize_admonitions(source, language)
    assert f'!!! note "{i18n.ADMONITION_TITLES[language]["note"]}"' in translated
    assert f'???+ warning "{i18n.ADMONITION_TITLES[language]["warning"]}"' in translated
    assert '```markdown\n!!! note\n```' in translated
    assert '!!! tip "Custom title"' in translated


@pytest.mark.parametrize("language", ["en", "zh", "ja", "ru", "tr"])
@pytest.mark.parametrize(
    "opening,interior,closing",
    [
        ("```text", "```markdown", "```"),
        ("````text", "````markdown", "````"),
        ("~~~text", "~~~markdown", "~~~"),
        ("```text", "~~~", "```"),
        ("````text", "```", "````"),
    ],
)
def test_callout_localization_preserves_rendered_fenced_code(
    language, opening, interior, closing
):
    fenced = f"{opening}\n{interior}\n!!! note\n{closing}\n"
    source = fenced + "\n!!! warning\n    Outside the code example.\n"
    translated = i18n.localize_admonitions(source, language)
    assert translated.startswith(fenced)

    def rendered_code(text):
        rendered = markdown.Markdown(
            extensions=["pymdownx.superfences", "admonition"]
        ).convert(text)
        return re.findall(r"<code>.*?</code>", rendered, re.S)

    assert rendered_code(source)
    assert rendered_code(translated) == rendered_code(source)
    if language != "en":
        title = i18n.ADMONITION_TITLES[language]["warning"]
        assert f'!!! warning "{title}"' in translated


@pytest.mark.parametrize("language", ["ja", "ru", "tr"])
def test_redirect_text_changes_without_changing_destinations(tmp_path, language):
    page = tmp_path / "index.html"
    original = (
        '<html lang="en"><title>Redirecting...</title>'
        '<meta http-equiv="refresh" content="0; url=../../requirements/">'
        '<body>You\'re being redirected to a '
        '<a href="../../requirements/">new destination</a>.</body></html>'
    )
    page.write_text(original)
    i18n.localize_redirects(tmp_path, language)
    translated = page.read_text()
    assert f'lang="{language}"' in translated
    assert i18n.REDIRECT_MESSAGES[language][0] in translated
    assert 'href="../../requirements/"' in translated
    assert 'content="0; url=../../requirements/"' in translated
    assert "Redirecting..." not in translated
