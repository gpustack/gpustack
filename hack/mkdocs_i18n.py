"""Keep documentation languages on the same page and identify English fallbacks."""

import hashlib
import json
import posixpath
import re
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit

import yaml
from jinja2 import ChoiceLoader, DictLoader, TemplateNotFound
from mkdocs.plugins import event_priority


LANGUAGES = {"en": "English", "zh": "简体中文", "ja": "日本語"}
FALLBACK_NOTICES = {
    "zh": (
        "此页面暂未提供中文翻译",
        "当前显示英文内容，可继续阅读或",
        "切换到英文版",
    ),
    "ja": (
        "このページの日本語訳はまだありません",
        "英語の内容を表示しています。このまま読むか、",
        "英語版に切り替えてください",
    ),
}

# Material indexes Japanese with TinySegmenter, but its query preprocessor only
# splits Han runs. Use the same tokenizer for plain mixed-script queries; leave
# explicit Lunr operators intact so advanced queries retain their meaning.
JAPANESE_SEARCH_QUERIES = r"""
(() => {
  const listen = self.addEventListener;
  let segmenter;
  self.addEventListener = function(type, handler, options) {
    if (type !== "message") return listen.call(this, type, handler, options);
    return listen.call(this, type, event => {
      const message = event.data;
      if (message && message.type === 2 && typeof message.data === "string" &&
          self.lunr && self.lunr.TinySegmenter &&
          !/[":*~^+\\]|(^|\s)-/.test(message.data)) {
        segmenter = segmenter || new self.lunr.TinySegmenter();
        message.data = message.data.split(/\s+/).map(term =>
          /[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}]/u.test(term)
            ? segmenter.segment(term).join(" ") : term
        ).join(" ");
      }
      return handler(event);
    }, options);
  };
})();
"""


def language_prefix(language: str) -> str:
    """Return a language's directory inside the shared version root."""
    if language not in LANGUAGES:
        raise ValueError(f"unsupported documentation language: {language}")
    return "" if language == "en" else f"{language}/"


def on_config(config):
    """Initialize English navigation for ordinary MkDocs and versioned builds."""
    extra = config["extra"]
    if "docs_i18n" not in extra and not extra.get("docs_i18n_prepared"):
        extra["docs_i18n"] = {
            "language": "en",
            "offline": bool(extra.get("offline")),
            "translated_sources": [],
            "prepared": False,
        }
    return config


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
    current = language_prefix(language) + page_url
    directory = current if current.endswith("/") else posixpath.dirname(current)
    links = []
    for target_language, name in LANGUAGES.items():
        target = language_prefix(target_language) + page_url
        relative = posixpath.relpath(target or ".", directory or ".")
        if not page_url or page_url.endswith("/"):
            relative = "./" if relative == "." else relative.rstrip("/") + "/"
        links.append({"name": name, "link": relative, "lang": target_language})
    return links


def on_env(env, *, config, files):
    """Render already page-relative language links without normalizing them twice."""
    if "docs_i18n" not in config["extra"]:
        return env
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
    settings = config["extra"].get("docs_i18n")
    if not settings:
        return context
    config["extra"]["alternate"] = language_links(page.url, settings["language"])
    if (
        settings["language"] != "en"
        and getattr(page, "edit_url", None)
        and is_translated_page(page, settings)
    ):
        page.edit_url = page.edit_url.replace(
            "/docs/", f"/docs-{settings['language']}/", 1
        )
    return context


def on_template_context(context, *, template_name, config):
    """Offer both documentation homepages on generated error pages."""
    if template_name == "404.html":
        settings = config["extra"].get("docs_i18n")
        if not settings:
            return context
        root = (
            "/help/"
            if settings["offline"]
            else urlsplit(config["site_url"] or "/").path
        )
        root = "/" + root.strip("/") + "/" if root.strip("/") else "/"
        if (
            not settings["offline"]
            and settings["language"] != "en"
            and root.endswith("/" + language_prefix(settings["language"]))
        ):
            root = root[: -len(language_prefix(settings["language"]))]
        config["extra"]["alternate"] = [
            {"name": name, "link": root + language_prefix(language), "lang": language}
            for language, name in LANGUAGES.items()
        ]
    return context


def is_translated_page(page, settings: dict) -> bool:
    """Whether the page displays a completed translated source file."""
    return (
        page.file.src_uri in settings["translated_sources"]
        and (getattr(page, "meta", None) or {}).get("translation_status") != "pending"
    )


def on_page_markdown(markdown, *, page, config, files):
    """Explain English fallback pages in the selected language."""
    settings = config["extra"].get("docs_i18n")
    if (
        not settings
        or settings["language"] == "en"
        or is_translated_page(page, settings)
    ):
        return markdown
    title, description, label = FALLBACK_NOTICES[settings["language"]]
    english = language_links(page.url, settings["language"])[0]["link"]
    return (
        f'!!! info "{title}"\n\n'
        f"    {description}[{label}]({english})。\n\n" + markdown
    )


@event_priority(-100)
def on_post_build(config):
    """Prepare language-aware search and public version navigation."""
    settings = config["extra"].get("docs_i18n") or {}
    language = settings.get("language", "en")
    site = Path(config["site_dir"])
    if language == "ja":
        prepare_japanese_search(site)
    version = config["extra"].get("version") or {}
    if not settings.get("offline") and version.get("provider") == "mike":
        prepare_version_navigation(site, language)


def replace_html_references(site: Path, replacements: Dict[str, str]) -> None:
    """Point generated pages at content-addressed localized JavaScript assets."""
    if replacements:
        for page in site.rglob("*.html"):
            content = page.read_text(encoding="utf-8")
            for original, replacement in replacements.items():
                content = content.replace(original, replacement)
            page.write_text(content, encoding="utf-8")


def prepare_version_navigation(site: Path, language: str) -> None:
    """Keep Material's mike selector inside the same language across versions."""
    prefix = language_prefix(language)
    if not prefix:
        return
    replacements = {}
    bundles = [
        path
        for path in (site / "assets/javascripts").glob("bundle.*.min.js")
        if not path.name.startswith("bundle.docs-")
    ]
    if not bundles:
        raise ValueError(
            "cannot find Material's JavaScript bundle for version navigation"
        )
    helpers = (
        "function __docsVersionRoot(config){return new URL('../',config.base).href;}"
        "function __docsVersionTarget(config,version){return new URL('../'+version+"
        f"'/{prefix}',__docsVersionRoot(config));}}"
    )
    for bundle in bundles:
        content = bundle.read_text(encoding="utf-8")
        fetch_pattern = r'new URL\("\.\./versions\.json",([\w$]+)\.base\)'
        match = re.search(fetch_pattern, content)
        if not match:
            raise ValueError("unsupported Material mike version-navigation JavaScript")
        variable = match[1]
        content, fetch_count = re.subn(
            fetch_pattern,
            lambda entry: f'new URL("../versions.json",__docsVersionRoot({entry[1]}))',
            content,
        )
        target_pattern = r'new URL\(`\.\./\$\{([\w$]+\.version)\}/`,([\w$]+)\.base\)'
        content, target_count = re.subn(
            target_pattern,
            lambda entry: f"__docsVersionTarget({entry[2]},{entry[1]})",
            content,
        )
        current = f"{variable}.base.match(/([^/]+)\\/?$/)"
        current_count = content.count(current)
        content = content.replace(
            current, f"__docsVersionRoot({variable}).match(/([^/]+)\\/?$/)"
        )
        if (fetch_count, target_count, current_count) != (1, 2, 1):
            raise ValueError("unsupported Material mike version-navigation JavaScript")
        content = "(() => {'use strict';" + helpers + content + "})();"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        name = f"bundle.docs-{language}.{digest}.min.js"
        (bundle.parent / name).write_text(content, encoding="utf-8")
        replacements[bundle.name] = name
    replace_html_references(site, replacements)


def prepare_japanese_search(site: Path) -> None:
    """Normalize a built Japanese index and its local browser search worker."""
    index = site / "search/search_index.json"
    if index.is_file():
        # Material applies installed jieba to every Han run, including Japanese.
        # Its zero-width boundaries disagree with TinySegmenter's query tokens.
        data = json.loads(index.read_text(encoding="utf-8"))
        for entry in data.get("docs", []):
            for field in ("title", "text"):
                if isinstance(entry.get(field), str):
                    entry[field] = entry[field].replace("\u200b", "")
        index.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    replacements = {}
    for worker in (site / "assets/javascripts/workers").glob("search.*.min.js"):
        if worker.name.startswith("search.ja."):
            continue
        content = JAPANESE_SEARCH_QUERIES + worker.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        name = f"search.ja.{digest}.min.js"
        (worker.parent / name).write_text(content, encoding="utf-8")
        replacements[worker.name] = name
    replace_html_references(site, replacements)
