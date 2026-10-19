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
    (root / "docs-i18n/mkdocs").mkdir(parents=True)
    for language in i18n.LANGUAGES:
        source = root / ("docs" if language == "en" else f"docs-i18n/{language}")
        (source / "user-guide").mkdir(parents=True)
        titles = {
            "en": "Cluster management",
            "zh": "集群管理",
            "ja": "クラスター管理",
            "ru": "Управление кластерами",
            "tr": "Küme yönetimi",
        }
        bodies = {
            "en": "Deploy language models safely with GPUStack.",
            "zh": "使用集群管理功能安全部署语言模型。",
            "ja": "GPUStackは言語モデルを安全にデプロイします。クラスター管理を試します。",
            "ru": "Управление кластерами: безопасное развертывание языковых моделей GPUStack.",
            "tr": "Küme yönetimi ile GPUStack dil modellerini güvenle dağıtın.",
        }
        (source / "user-guide/guide.md").write_text(
            f"# {titles[language]}\n\n{bodies[language]}\n", encoding="utf-8"
        )
        (source / "index.md").write_text(f"# {titles[language]}\n", encoding="utf-8")
        if language != "en":
            (root / f"docs-i18n/mkdocs/{language}.yml").write_text(
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
        assert index["config"]["lang"] == (
            [language] if language in ("ja", "ru", "tr") else ["en"]
        )
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


@pytest.fixture
def mike_publication(multilingual_source):
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
    legacy = yaml.safe_load((multilingual_source / "mkdocs.yml").read_text())
    legacy["hooks"] = []
    legacy_config = multilingual_source / "mkdocs.legacy.yml"
    legacy_config.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    command(
        str(Path(sys.executable).parent / "mike"),
        "deploy",
        "2.0",
        "--config-file",
        str(legacy_config),
        "--ignore-remote-status",
    )
    legacy_config.unlink()
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
    published = multilingual_source.parent / "published"
    command("git", "worktree", "add", "--detach", str(published), "gh-pages")
    return command, published


def test_mike_publishes_version_and_latest_with_every_language(mike_publication):
    command, published = mike_publication

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


def test_version_menu_falls_back_to_real_english_only_versions(mike_publication):
    command, published = mike_publication
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise Material's version menu")

    def navigate(language, version, options=None):
        return json.loads(
            command(
                node,
                str(ROOT / "tests/fixtures/docs_version_menu.cjs"),
                str(published),
                language,
                version,
                json.dumps(options or {}),
            )
        )

    english = "https://docs.example.test/docs/2.0/"
    for language in i18n.LANGUAGES:
        if language == "en":
            continue
        assert not (published / "2.0" / language).exists()
        result = navigate(language, "2.0")
        assert result["targetHref"] == english
        assert result["navigated"] == (
            english + "user-guide/guide/?q=language#cluster-management"
        )
        assert (
            result["requests"].count(
                f"https://docs.example.test/docs/2.0/{language}/sitemap.xml"
            )
            == 1
        )
        assert result["requests"][-1] == english + "sitemap.xml"
        assert result["prevented"]
        for modifier in ("ctrl", "meta"):
            result = navigate(language, "2.0", {"modifier": modifier})
            assert result["targetHref"] == english
            assert not result["prevented"]
            assert result["navigated"].startswith(
                f"https://docs.example.test/docs/latest/{language}/"
            )

        localized = f"https://docs.example.test/docs/2.2/{language}/"
        result = navigate(
            language,
            "2.2",
            {"omitCurrentPageFrom": [f"2.2/{language}/sitemap.xml"]},
        )
        assert result["targetHref"] == localized
        assert result["navigated"] == localized
        result = navigate(
            language,
            "2.0",
            {"omitCurrentPageFrom": ["2.0/sitemap.xml"]},
        )
        assert result["targetHref"] == english
        assert result["navigated"] == english

        english_newer = "https://docs.example.test/docs/2.2/"
        result = navigate(
            language,
            "2.2",
            {"failedRequests": [f"2.2/{language}/sitemap.xml"]},
        )
        assert result["targetHref"] == english_newer
        assert result["navigated"] == (
            english_newer + "user-guide/guide/?q=language#cluster-management"
        )
        result = navigate(
            language,
            "2.2",
            {"failedRequests": [f"2.2/{language}/sitemap.xml", "2.2/sitemap.xml"]},
        )
        assert result["targetHref"] == english_newer
        assert result["navigated"] == english_newer


def test_version_menu_bounds_sitemap_waits_and_ignores_late_responses(mike_publication):
    command, published = mike_publication
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise Material's version menu")

    def navigate(language, version, options):
        return json.loads(
            command(
                node,
                str(ROOT / "tests/fixtures/docs_version_menu.cjs"),
                str(published),
                language,
                version,
                json.dumps({"timerScale": 0.003, "waitMs": 40, **options}),
            )
        )

    base = "https://docs.example.test/docs/"
    english = navigate("en", "2.0", {"hangRequests": ["2.2/sitemap.xml"]})
    assert english["targetHref"] == base + "2.0/"
    assert english["navigated"] == (
        base + "2.0/user-guide/guide/?q=language#cluster-management"
    )
    assert base + "2.2/sitemap.xml" not in english["requests"]
    assert english["expiredTimers"] == 0

    for language in i18n.LANGUAGES:
        if language == "en":
            continue
        probe = f"2.2/{language}/sitemap.xml"
        result = navigate(language, "2.0", {"hangRequests": [probe]})
        assert result["targetHref"] == base + "2.0/"
        assert base + "2.2/" in result["targets"]
        assert result["navigated"] == (
            base + "2.0/user-guide/guide/?q=language#cluster-management"
        )
        assert result["abortedRequests"] == [base + probe]
        assert result["activeRequests"] == []
        assert result["activeTimers"] == 0
        assert result["expiredTimers"] == 1
        assert result["clearedTimers"] >= 1

        result = navigate(language, "2.2", {"hangRequests": [probe, "2.2/sitemap.xml"]})
        assert result["targetHref"] == base + "2.2/"
        assert result["navigated"] == base + "2.2/"
        assert result["abortedRequests"] == [base + probe, base + "2.2/sitemap.xml"]
        assert result["activeRequests"] == []
        assert result["activeTimers"] == 0
        assert result["expiredTimers"] == 2
        assert result["clearedTimers"] >= 1

        for modifier in (None, "ctrl", "meta"):
            result = navigate(
                language,
                "2.2",
                {
                    "delayedRequests": {probe: 100},
                    "waitAfterMenu": 120,
                    "modifier": modifier,
                },
            )
            assert result["targetHref"] == base + "2.2/"
            assert base + "2.2/" in result["targets"]
            assert base + f"2.2/{language}/" not in result["targets"]
            assert result["lateLoads"] == [base + probe]
            assert result["abortedRequests"] == [base + probe]
            assert result["activeRequests"] == []
            assert result["activeTimers"] == 0
            assert result["expiredTimers"] == 1
            assert result["clearedTimers"] >= 1
            if modifier:
                assert not result["prevented"]
                assert result["navigated"].startswith(base + f"latest/{language}/")
            else:
                assert result["prevented"]
                assert result["navigated"] == (
                    base + "2.2/user-guide/guide/?q=language#cluster-management"
                )


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
    (multilingual_source / "docs-i18n/mkdocs/ja.yml").unlink()

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
    japanese = multilingual_source / "docs-i18n/ja/user-guide/guide.md"
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


@pytest.mark.parametrize(
    "destination",
    [
        ".",
        "docs",
        "docs-i18n",
        "docs-i18n/ja/nested",
        "docs-i18n/mkdocs",
        "docs-i18n/mkdocs/nested",
        "docs-i18n/future",
        "hack",
        ".git",
    ],
)
def test_builder_refuses_output_inside_sources(multilingual_source, destination):
    originals = {
        path.relative_to(multilingual_source): path.read_bytes()
        for path in multilingual_source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="outside source trees"):
        builder.build_docs(
            multilingual_source,
            destination=multilingual_source / destination,
        )
    assert {
        path.relative_to(multilingual_source): path.read_bytes()
        for path in multilingual_source.rglob("*")
        if path.is_file()
    } == originals


def test_version_navigation_refuses_unsupported_material_bundle(tmp_path):
    assets = tmp_path / "assets/javascripts"
    assets.mkdir(parents=True)
    (assets / "bundle.example.min.js").write_text("unsupported vendor layout")

    with pytest.raises(ValueError, match="unsupported Material"):
        i18n.prepare_version_navigation(tmp_path, "ja")

    assert not list(assets.glob("bundle.docs-*.min.js"))


@pytest.mark.parametrize(
    "language,queries",
    [
        ("ru", ["управление", "кластерами", "языковых", "GPUStack"]),
        ("tr", ["küme", "yönetimi", "modellerini", "GPUStack"]),
    ],
)
def test_new_language_workers_find_native_queries(
    multilingual_source, language, queries
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the bundled browser worker")
    output = multilingual_source / "site"
    builder.build_language(multilingual_source / "mkdocs.yml", output)
    site = output / language
    worker = next((site / "assets/javascripts/workers").glob("search.*.min.js"))
    index = json.loads((site / "search/search_index.json").read_text())
    assert index["config"]["lang"] == [language]
    assert (site / f"assets/javascripts/lunr/min/lunr.{language}.min.js").is_file()
    script = (ROOT / "tests/fixtures/docs_search_worker.cjs").read_text()
    result = subprocess.run(
        [node, "-e", script, str(site), str(worker), json.dumps(queries)],
        check=True,
        capture_output=True,
        text=True,
    )
    for query, matches in json.loads(result.stdout).items():
        assert any(
            location.startswith("user-guide/guide/") for location in matches
        ), query
