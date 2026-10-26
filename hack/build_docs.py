#!/usr/bin/env python3
"""Build and preview documentation languages from shared English sources."""

import argparse
import gzip
import importlib.util
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import yaml
from yaml.nodes import MappingNode

try:
    from .mkdocs_i18n import (
        LANGUAGES,
        is_translated_source,
        language_links,
        language_prefix,
    )
except ImportError:
    from mkdocs_i18n import (
        LANGUAGES,
        is_translated_source,
        language_links,
        language_prefix,
    )


ROOT = Path(__file__).resolve().parents[1]


def config_field(text: str, name: str):
    """Read a plain root field without evaluating the config's Python YAML tags."""
    document = yaml.compose(text)
    if not isinstance(document, MappingNode):
        raise ValueError("MkDocs configuration must be a mapping")
    for key, value in document.value:
        if key.value == name:
            return yaml.safe_load(yaml.serialize(value))
    return None


def translate_nav(nav, translations: Dict[str, str]):
    """Translate labels without changing the page paths shared by both languages."""
    if isinstance(nav, list):
        return [translate_nav(item, translations) for item in nav]
    if isinstance(nav, dict):
        return {
            translations.get(label, label): translate_nav(value, translations)
            for label, value in nav.items()
        }
    return nav


def overlay_tree(source: Path, destination: Path, translations: bool = False) -> None:
    """Copy shared resources, skipping translations marked as pending."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if translations and path.suffix == ".md" and not is_translated_source(path):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _html_attributes(tag: str, attributes: Dict[str, str]) -> str:
    for name, value in attributes.items():
        pattern = rf"\s{re.escape(name)}\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)"
        replacement = f' {name}="{escape(value, quote=True)}"'
        if re.search(pattern, tag):
            tag = re.sub(pattern, lambda _, value=replacement: value, tag)
        else:
            tag = tag[:-1] + replacement + ">"
    return tag


def configure_image_selector(docs: Path, language: str) -> None:
    """Set the standalone image selector's language and local language links."""
    page = docs / "image-selector/index.html"
    if not page.is_file():
        return
    text = page.read_text(encoding="utf-8")
    text = re.sub(
        r"<html\b[^>]*>",
        lambda match: _html_attributes(
            match.group(), {"lang": language, "data-docs-language": language}
        ),
        text,
        count=1,
    )
    links = {
        entry["lang"]: entry["link"]
        for entry in language_links("image-selector/", language)
    }
    for target_language, link in links.items():
        text = re.sub(
            rf"<a\b[^>]*\bdata-lang=[\"']{target_language}[\"'][^>]*>",
            lambda match, href=link, lang=target_language: _html_attributes(
                match.group(), {"href": href, "hreflang": lang}
            ),
            text,
        )
    page.write_text(text, encoding="utf-8")


def configure_translation(child: dict, config_text: str, translation: dict) -> None:
    """Localize navigation and theme text while inheriting all other settings."""
    for field in ("site_name", "site_description"):
        if field in translation:
            child[field] = translation[field]
    child["nav"] = translate_nav(
        config_field(config_text, "nav"), translation.get("nav_translations", {})
    )
    theme = config_field(config_text, "theme") or {}
    for palette in theme.get("palette", []):
        if "toggle" in palette:
            toggle = palette["toggle"]
            toggle["name"] = translation.get("theme_translations", {}).get(
                toggle["name"], toggle["name"]
            )
    if "palette" in theme:
        child["theme"]["palette"] = theme["palette"]


def language_site_url(base: str, language: str, offline: bool) -> str:
    """Keep offline URLs relative and public canonical URLs in their version root."""
    if offline or not base:
        return ""
    return base.rstrip("/") + "/" + language_prefix(language)


def prepare_language(
    root: Path,
    stage: Path,
    language: str,
    offline: bool = False,
    stamp: Optional[str] = None,
    site_url: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Path:
    """Assemble one language without editing source files or shared configuration."""
    if language not in LANGUAGES:
        raise ValueError(f"unsupported documentation language: {language}")
    root, stage = root.resolve(), stage.resolve()
    validate_destination(root, stage)
    translation_dir = root / "docs-i18n" / language
    sources = [root / "docs"]
    if language != "en":
        sources.append(translation_dir)
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"documentation source not found: {source}")
    stage.mkdir(parents=True, exist_ok=True)
    docs = stage / "docs"
    if docs.is_symlink():
        raise ValueError("documentation staging must not use a symlink for its sources")
    if docs.exists():
        shutil.rmtree(docs)
    overlay_tree(root / "docs", docs)
    translated = sorted(
        path.relative_to(translation_dir).as_posix()
        for path in translation_dir.rglob("*.md")
        if is_translated_source(path)
    )
    if language != "en":
        overlay_tree(translation_dir, docs, translations=True)
    configure_image_selector(docs, language)

    config_text = (root / "mkdocs.yml").read_text(encoding="utf-8")
    shutil.copy2(root / "mkdocs.yml", stage / "mkdocs.yml")
    hooks = config_field(config_text, "hooks") or []
    if offline:
        offline_text = (root / "mkdocs.offline.yml").read_text(encoding="utf-8")
        shutil.copy2(root / "mkdocs.offline.yml", stage / "mkdocs.offline.yml")
        hooks = list(dict.fromkeys(hooks + (config_field(offline_text, "hooks") or [])))
    hooks = list(dict.fromkeys(hooks + ["hack/mkdocs_i18n.py"]))
    for hook in hooks:
        source = root / hook
        target = stage / hook
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    settings = {
        "language": language,
        "offline": offline,
        "translated_sources": translated,
        "prepared": True,
    }
    child = {
        "INHERIT": "mkdocs.offline.yml" if offline else "mkdocs.yml",
        "theme": {"language": language},
        "hooks": hooks,
        "site_url": language_site_url(
            site_url if site_url is not None else config_field(config_text, "site_url"),
            language,
            offline,
        ),
        "extra": {
            "alternate": language_links("", language),
            "docs_i18n": settings,
        },
    }
    if offline and stamp:
        child["extra"]["docs_stamp"] = stamp
    if strict is not None:
        child["strict"] = strict
    if language != "en":
        translation = yaml.safe_load(
            (root / "docs-i18n/mkdocs" / f"{language}.yml").read_text(encoding="utf-8")
        )
        configure_translation(child, config_text, translation)
    config_path = stage / "mkdocs.language.yml"
    config_path.write_text(
        yaml.safe_dump(child, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    if offline:
        cache = root / ".cache/plugin/privacy"
        cache.mkdir(parents=True, exist_ok=True)
        link = stage / ".cache/plugin/privacy"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(cache, target_is_directory=True)
    return config_path


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_language(config_path: Path, output: Path) -> None:
    """Build a staged language through MkDocs' configuration and build APIs."""
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    with working_directory(config_path.parent):
        config = load_config(config_file=str(config_path), site_dir=str(output))
        config.plugins.on_startup(command="build", dirty=False)
        try:
            build(config)
        finally:
            config.plugins.on_shutdown()


def documentation_stamp(root: Path) -> str:
    """Identify an offline build by its UTC date and source commit."""
    commit = os.environ.get("DOCS_COMMIT", "")
    if not commit or commit == "unknown":
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else ""
    if not commit:
        raise ValueError(
            "cannot determine documentation commit; pass DOCS_COMMIT=<sha>"
        )
    return f"{datetime.now(timezone.utc):%Y-%m-%d} ({commit[:7]})"


def check_offline_site(root: Path, output: Path) -> None:
    """Reuse the offline asset gate for each language before publishing the site."""
    index = output / "search/search_index.json"
    with (
        index.open("rb") as source,
        gzip.open(str(index) + ".gz", "wb", compresslevel=9) as target,
    ):
        shutil.copyfileobj(source, target)
    spec = importlib.util.spec_from_file_location(
        "check_offline_docs", root / "hack/check_offline_docs.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    if checker.main(output):
        raise ValueError(f"offline documentation contains external assets: {output}")


def publish_site(source: Path, destination: Path, backup: Path) -> None:
    """Replace a complete site, restoring its previous contents if publication fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists()
    if exists:
        destination.rename(backup)
    try:
        source.rename(destination)
    except OSError:
        if exists:
            backup.rename(destination)
        raise


def validate_destination(root: Path, destination: Path) -> None:
    """Reject output paths that could replace documentation or builder sources."""
    protected = [root / name for name in ("docs", "docs-i18n", "hack", ".git")]
    for source in protected:
        if (
            destination == source
            or destination in source.parents
            or source in destination.parents
        ):
            raise ValueError(
                f"documentation output must be outside source trees: {destination}"
            )


def build_docs(
    root: Path,
    offline: bool = False,
    stamp: Optional[str] = None,
    destination: Optional[Path] = None,
    site_url: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Path:
    """Publish the combined site only after every language passes its build."""
    root = root.resolve()
    destination = destination or root / ("gpustack/help" if offline else "site")
    destination = destination.resolve()
    validate_destination(root, destination)
    if offline and not stamp:
        stamp = documentation_stamp(root)
    with TemporaryDirectory(prefix=".docs-build-", dir=root) as temporary:
        workspace = Path(temporary)
        combined = workspace / "combined"
        combined.mkdir()
        for language in LANGUAGES:
            stage = workspace / language
            config = prepare_language(
                root, stage, language, offline, stamp, site_url, strict
            )
            output = stage / "output"
            build_language(config, output)
            if offline:
                check_offline_site(root, output)
            target = combined / language_prefix(language)
            shutil.copytree(output, target, dirs_exist_ok=True)
        publish_site(combined, destination, workspace / "previous")
    return destination


def serve_docs(root: Path, address: str) -> None:
    """Rebuild and live reload all languages when sources or configuration change."""
    from mkdocs.livereload import LiveReloadServer

    host, port = address.rsplit(":", 1)
    with TemporaryDirectory(prefix=".docs-preview-", dir=root) as temporary:
        preview = Path(temporary) / "site"

        def rebuild() -> None:
            build_docs(root, destination=preview, site_url=f"http://{address}/")

        rebuild()
        server = LiveReloadServer(
            builder=rebuild, host=host, port=int(port), root=str(preview)
        )
        server.error_handler = lambda code: (
            (preview / "404.html").read_bytes() if code == 404 else None
        )
        watched = ["docs", "docs-i18n", "mkdocs.yml", "mkdocs.offline.yml", "hack"]
        for name in watched:
            server.watch(str(root / name))
        for language, name in LANGUAGES.items():
            print(f"{name}: http://{address}/{language_prefix(language)}", flush=True)
        try:
            server.serve()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()


def main() -> None:
    """Select a public build, offline build, or multilingual development preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--serve", action="store_true")
    parser.add_argument("--stamp")
    args = parser.parse_args()
    if args.serve:
        serve_docs(ROOT, os.environ.get("DOCS_DEV_ADDR", "127.0.0.1:8000"))
    else:
        output = build_docs(ROOT, args.offline, args.stamp)
        print(f"Multilingual documentation built to {output}")


if __name__ == "__main__":
    main()
