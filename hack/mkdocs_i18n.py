"""Keep documentation languages on the same page and identify English fallbacks."""

import posixpath
import re
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit

import yaml
from jinja2 import ChoiceLoader, DictLoader, TemplateNotFound


def is_translated_source(source: Path) -> bool:
    """Whether a translation exists and is ready to replace the English page."""
    if not source.is_file():
        return False
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return True
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in ("---", "..."):
            metadata = yaml.safe_load("\n".join(lines[1:index])) or {}
            return (
                not isinstance(metadata, dict)
                or metadata.get("translation_status") != "pending"
            )
    return True


def language_links(page_url: str, language: str) -> List[Dict[str, str]]:
    """Return relative links within the combined site, retaining version prefixes."""
    current = ("zh/" if language == "zh" else "") + page_url
    directory = current if current.endswith("/") else posixpath.dirname(current)
    links = []
    for target_language, name in (("en", "English"), ("zh", "简体中文")):
        target = ("zh/" if target_language == "zh" else "") + page_url
        relative = posixpath.relpath(target or ".", directory or ".")
        if not page_url or page_url.endswith("/"):
            relative = "./" if relative == "." else relative.rstrip("/") + "/"
        links.append({"name": name, "link": relative, "lang": target_language})
    return links


def on_env(env, *, config, files):
    """Render already page-relative language links without normalizing them twice."""
    templates = {}
    for template_name in ("partials/alternate.html", "base.html"):
        try:
            template, _, _ = env.loader.get_source(env, template_name)
        except TemplateNotFound:
            continue
        templates[template_name] = re.sub(
            r"\balt\.link\s*\|\s*url\b", "alt.link", template
        )
    env.loader = ChoiceLoader([DictLoader(templates), env.loader])
    return env


def on_page_context(context, *, page, config, nav):
    """Keep language navigation and source actions on the displayed page."""
    settings = config["extra"]["docs_i18n"]
    config["extra"]["alternate"] = language_links(page.url, settings["language"])
    if (
        settings["language"] == "zh"
        and getattr(page, "edit_url", None)
        and is_translated_page(page, settings)
    ):
        page.edit_url = page.edit_url.replace("/docs/", "/docs-zh/", 1)
    return context


def on_template_context(context, *, template_name, config):
    """Offer both documentation homepages on generated error pages."""
    if template_name == "404.html":
        settings = config["extra"]["docs_i18n"]
        root = (
            "/help/"
            if settings["offline"]
            else urlsplit(config["site_url"] or "/").path
        )
        root = "/" + root.strip("/") + "/" if root.strip("/") else "/"
        if (
            not settings["offline"]
            and settings["language"] == "zh"
            and root.endswith("/zh/")
        ):
            root = root[:-3]
        config["extra"]["alternate"] = [
            {"name": "English", "link": root, "lang": "en"},
            {"name": "简体中文", "link": root + "zh/", "lang": "zh"},
        ]
    return context


def is_translated_page(page, settings: dict) -> bool:
    """Whether the page displays a completed Chinese source file."""
    return (
        page.file.src_uri in settings["translated_sources"]
        and (getattr(page, "meta", None) or {}).get("translation_status") != "pending"
    )


def on_page_markdown(markdown, *, page, config, files):
    """Explain why an untranslated page is English in the Chinese site."""
    settings = config["extra"]["docs_i18n"]
    if settings["language"] != "zh" or is_translated_page(page, settings):
        return markdown
    english = language_links(page.url, "zh")[0]["link"]
    return (
        '!!! info "此页面暂未提供中文翻译"\n\n'
        "    当前显示英文内容，可继续阅读或"
        f"[切换到英文版]({english})。\n\n" + markdown
    )
