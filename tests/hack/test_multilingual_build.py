"""Ordinary MkDocs, versioned publication, and Japanese search keep all languages."""

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import pytest
import yaml

from hack import build_docs as builder
from hack import mkdocs_i18n as i18n
from hack import mkdocs_offline as offline_hook


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def multilingual_source(tmp_path):
    root = tmp_path / "documentation"
    for language in i18n.LANGUAGES:
        source = root / ("docs" if language == "en" else f"docs-{language}")
        (source / "user-guide").mkdir(parents=True)
        titles = {"en": "Cluster management", "zh": "集群管理", "ja": "クラスター管理"}
        bodies = {
            "en": "Deploy language models safely with GPUStack.",
            "zh": "使用集群管理功能安全部署语言模型。",
            "ja": "GPUStackは言語モデルを安全にデプロイします。クラスター管理を試します。",
        }
        (source / "user-guide/guide.md").write_text(
            f"# {titles[language]}\n\n{bodies[language]}\n", encoding="utf-8"
        )
        (source / "index.md").write_text(f"# {titles[language]}\n", encoding="utf-8")
        if language != "en":
            (root / f"mkdocs.{language}.yml").write_text(
                yaml.safe_dump({"nav_translations": {"Guide": titles[language]}}),
                encoding="utf-8",
            )
    (root / "hack").mkdir()
    for name in (
        "build_docs.py",
        "mkdocs_i18n.py",
        "mkdocs_multilingual.py",
        "mkdocs_offline.py",
        "check_offline_docs.py",
    ):
        shutil.copy2(ROOT / "hack" / name, root / "hack" / name)
    (root / "mkdocs.yml").write_text(
        yaml.safe_dump(
            {
                "site_name": "GPUStack",
                "site_url": "https://docs.example.test/docs/",
                "theme": {"name": "material", "font": False},
                "plugins": {"search": {}},
                "extra": {"version": {"provider": "mike"}},
                "nav": [{"Home": "index.md"}, {"Guide": "user-guide/guide.md"}],
                "hooks": ["hack/mkdocs_i18n.py", "hack/mkdocs_multilingual.py"],
            }
        ),
        encoding="utf-8",
    )
    return root


class HeadLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.languages = {}
        self.canonical = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link" and attrs.get("rel") == "alternate" and attrs.get("hreflang"):
            self.languages[attrs["hreflang"]] = attrs["href"]
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs["href"]


def test_ordinary_mkdocs_build_publishes_every_language_and_native_search(
    multilingual_source,
):
    output = multilingual_source / "site"

    builder.build_language(multilingual_source / "mkdocs.yml", output)

    for language in i18n.LANGUAGES:
        language_site = output / i18n.language_prefix(language)
        assert (language_site / "user-guide/guide/index.html").is_file()
        index = json.loads((language_site / "search/search_index.json").read_text())
        assert index["config"]["lang"] == (["ja"] if language == "ja" else ["en"])
        if language == "ja":
            assert all("\u200b" not in entry["text"] for entry in index["docs"])
            assert any("言語モデル" in entry["text"] for entry in index["docs"])
            assets = language_site / "assets/javascripts/lunr"
            assert (assets / "tinyseg.js").is_file()
            assert (assets / "min/lunr.ja.min.js").is_file()
        elif language == "zh":
            assert any("\u200b" in entry["text"] for entry in index["docs"])
        if language != "en":
            bundles = list(
                (language_site / "assets/javascripts").glob("bundle.docs-*.min.js")
            )
            assert len(bundles) == 1
            original = bundles[0].read_bytes()
            html = (language_site / "user-guide/guide/index.html").read_text()
            assert bundles[0].name in html
            i18n.prepare_version_navigation(language_site, language)
            assert bundles[0].read_bytes() == original
            assert (language_site / "user-guide/guide/index.html").read_text() == html
    assert not list(multilingual_source.glob(".docs-build-*"))


def test_mike_publishes_version_and_latest_with_every_language(multilingual_source):
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "GIT_AUTHOR_NAME": "Documentation test",
        "GIT_AUTHOR_EMAIL": "docs@example.test",
        "GIT_COMMITTER_NAME": "Documentation test",
        "GIT_COMMITTER_EMAIL": "docs@example.test",
    }

    def command(*arguments):
        return subprocess.run(
            arguments,
            cwd=multilingual_source,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    command("git", "init", "-q")
    command("git", "config", "commit.gpgsign", "false")
    command("git", "add", ".")
    command(
        "git", "-c", "commit.gpgsign=false", "commit", "-qm", "documentation fixture"
    )
    command(
        str(Path(sys.executable).parent / "mike"),
        "deploy",
        "2.2",
        "--ignore-remote-status",
    )
    command(
        str(Path(sys.executable).parent / "mike"),
        "deploy",
        "2.3",
        "latest",
        "--alias-type",
        "copy",
        "--ignore-remote-status",
    )

    versions = json.loads(command("git", "show", "gh-pages:versions.json"))
    assert versions[0]["version"] == "2.3"
    assert versions[0]["aliases"] == ["latest"]
    for version in ("2.3", "latest"):
        for language in i18n.LANGUAGES:
            relative = i18n.language_prefix(language) + "user-guide/guide/"
            html = command("git", "show", f"gh-pages:{version}/{relative}index.html")
            links = HeadLinks()
            links.feed(html)
            assert links.canonical == ("https://docs.example.test/docs/2.3/" + relative)
            assert set(links.languages) == set(i18n.LANGUAGES)
            current = f"https://docs.example.test/docs/{version}/{relative}"
            for target, href in links.languages.items():
                assert urljoin(current, href) == (
                    f"https://docs.example.test/docs/{version}/"
                    + i18n.language_prefix(target)
                    + "user-guide/guide/"
                )
    assert command("git", "remote") == ""
    published = multilingual_source.parent / "published"
    command("git", "worktree", "add", "--detach", str(published), "gh-pages")
    node = shutil.which("node")
    if node:
        for language in i18n.LANGUAGES:
            result = json.loads(
                command(
                    node,
                    str(ROOT / "tests/fixtures/docs_version_menu.cjs"),
                    str(published),
                    language,
                )
            )
            assert (
                result["requests"][0] == "https://docs.example.test/docs/versions.json"
            )
            assert result["selected"] == "2.3"
            target = "https://docs.example.test/docs/2.2/" + i18n.language_prefix(
                language
            )
            assert result["targetHref"] == target
            assert (
                result["navigated"]
                == target + "user-guide/guide/?q=language#cluster-management"
            )
            assert result["requests"][-1] == target + "sitemap.xml"


def test_real_japanese_worker_finds_mixed_script_queries(multilingual_source):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the bundled browser worker")
    output = multilingual_source / "site"
    builder.build_language(multilingual_source / "mkdocs.yml", output)
    japanese = output / "ja"
    worker = next((japanese / "assets/javascripts/workers").glob("search.ja.*.min.js"))
    html = (japanese / "user-guide/guide/index.html").read_text()
    assert worker.name in html
    original_worker = worker.read_bytes()
    i18n.prepare_japanese_search(japanese)
    assert worker.read_bytes() == original_worker
    assert (japanese / "user-guide/guide/index.html").read_text() == html
    assert len(list(worker.parent.glob("search.ja.*.min.js"))) == 1
    for language in ("en", "zh"):
        workers = output / i18n.language_prefix(language) / "assets/javascripts/workers"
        assert not list(workers.glob("search.ja.*.min.js"))

    script = r"""
const fs = require("fs"), path = require("path"), vm = require("vm");
const root = process.argv[1], worker = process.argv[2];
let listener, reply;
const context = vm.createContext({
  console,
  addEventListener: (type, handler) => { listener = handler; },
  postMessage: value => { reply = value; }
});
context.self = context;
context.importScripts = (...urls) => {
  for (const url of urls)
    vm.runInContext(fs.readFileSync(path.resolve(path.dirname(worker), url), "utf8"), context);
};
vm.runInContext(fs.readFileSync(worker, "utf8"), context);
(async () => {
  const data = JSON.parse(fs.readFileSync(path.join(root, "search/search_index.json")));
  data.options = { suggest: false };
  await listener({ data: { type: 0, data } });
  const results = {};
  for (const query of ["言語モデル", "クラスター管理", "デプロイ", "GPUStack", "title:管理"]) {
    await listener({ data: { type: 2, data: query } });
    results[query] = reply.data.items.flat().map(item => item.location);
  }
  process.stdout.write(JSON.stringify(results));
})();
"""
    result = subprocess.run(
        [node, "-e", script, str(japanese), str(worker)],
        check=True,
        capture_output=True,
        text=True,
    )
    for query, matches in json.loads(result.stdout).items():
        assert any(
            location.startswith("user-guide/guide/") for location in matches
        ), query


def test_direct_offline_config_keeps_every_language_local(
    multilingual_source, monkeypatch
):
    monkeypatch.setenv("DOCS_COMMIT", "1234567890abcdef")
    configuration = multilingual_source / "mkdocs.offline.yml"
    configuration.write_text(
        yaml.safe_dump(
            {
                "INHERIT": "mkdocs.yml",
                "site_url": "",
                "extra": {"offline": True, "version": None},
                "hooks": [
                    "hack/mkdocs_offline.py",
                    "hack/mkdocs_i18n.py",
                    "hack/mkdocs_multilingual.py",
                ],
            }
        ),
        encoding="utf-8",
    )
    output = multilingual_source / "offline-site"

    builder.build_language(configuration, output)

    for language in i18n.LANGUAGES:
        site = output / i18n.language_prefix(language)
        html = (site / "user-guide/guide/index.html").read_text()
        links = HeadLinks()
        links.feed(html)
        assert links.canonical is None
        assert (
            gzip.decompress((site / "search/search_index.json.gz").read_bytes())
            == (site / "search/search_index.json").read_bytes()
        )
        error = (site / "404.html").read_text()
        assert 'href="/help/ja/"' in error
        assert not list((site / "assets/javascripts").glob("bundle.docs-*.min.js"))
        match = re.search(
            r'<script id="__config"[^>]*>(.*?)</script>', error, re.DOTALL
        )
        settings = json.loads(match[1])
        base = "/help/" + i18n.language_prefix(language)
        assert settings["base"] == base
        worker = settings["search"]
        assert worker.startswith(base + "assets/javascripts/workers/")
        assert (output / worker.removeprefix("/help/")).is_file()
        assert (
            output
            / settings["base"].removeprefix("/help/")
            / "search/search_index.json"
        ).is_file()
        resources = re.findall(r'(?:href|src)="([^"]+)"', error)
        for resource in resources:
            if resource.startswith(base + "assets/"):
                assert (output / resource.removeprefix("/help/")).is_file(), resource
    assert not list(multilingual_source.glob(".docs-root-*"))


@pytest.mark.parametrize("language", i18n.LANGUAGES)
def test_offline_error_page_privacy_assets_use_current_language(language):
    config = {"theme": {"language": language}}
    html = '<script src="/assets/external/vendor/example.js"></script>'
    html += '<link href="/help/assets/existing.css" rel="stylesheet">'

    rendered = offline_hook.on_post_template(
        html, template_name="404.html", config=config
    )

    assert (
        f'src="/help/{i18n.language_prefix(language)}assets/external/vendor/example.js"'
        in rendered
    )
    assert 'href="/help/assets/existing.css"' in rendered


def test_native_failure_preserves_published_output(multilingual_source):
    output = multilingual_source / "site"
    output.mkdir()
    (output / "previous.html").write_text("Previously published languages")
    (multilingual_source / "mkdocs.ja.yml").unlink()

    with pytest.raises(FileNotFoundError):
        builder.build_language(multilingual_source / "mkdocs.yml", output)

    assert sorted(path.name for path in output.iterdir()) == ["previous.html"]
    assert not list(multilingual_source.glob(".docs-root-*"))


def test_native_strict_translation_warning_preserves_published_output(
    multilingual_source,
):
    output = multilingual_source / "site"
    output.mkdir()
    (output / "previous.html").write_text("Previously published languages")
    japanese = multilingual_source / "docs-ja/user-guide/guide.md"
    japanese.write_text(japanese.read_text() + "\n[Missing](missing.md)\n")

    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=multilingual_source,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "strict mode" in result.stdout + result.stderr
    assert sorted(path.name for path in output.iterdir()) == ["previous.html"]
    assert not list(multilingual_source.glob(".docs-root-*"))
    assert not list(multilingual_source.glob(".docs-build-*"))


@pytest.mark.parametrize("destination", [".", "docs", "docs-ja/nested", "hack", ".git"])
def test_builder_refuses_output_inside_sources(multilingual_source, destination):
    with pytest.raises(ValueError, match="outside source trees"):
        builder.build_docs(
            multilingual_source,
            destination=multilingual_source / destination,
        )
    assert (multilingual_source / "docs/user-guide/guide.md").is_file()


def test_version_navigation_refuses_unsupported_material_bundle(tmp_path):
    assets = tmp_path / "assets/javascripts"
    assets.mkdir(parents=True)
    (assets / "bundle.example.min.js").write_text("unsupported vendor layout")

    with pytest.raises(ValueError, match="unsupported Material"):
        i18n.prepare_version_navigation(tmp_path, "ja")

    assert not list(assets.glob("bundle.docs-*.min.js"))
