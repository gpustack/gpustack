"""Build documentation languages without changing sources or published sites."""

import shutil
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin

import pytest
import yaml

from hack import build_docs as builder
from hack import mkdocs_i18n as i18n


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    files = {
        "docs/overview.md": "# Overview\n\nEnglish overview.\n",
        "docs/user-guide/guide.md": "# Guide\n\nEnglish guide.\n",
        "docs/user-guide/missing.md": "# Missing\n\nEnglish fallback.\n",
        "docs/user-guide/pending.md": "# Pending\n\nPublished English.\n",
        "docs/assets/chart.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
        "docs-i18n/zh/overview.md": "# 概览\n\n中文概览。\n",
        "docs-i18n/zh/user-guide/guide.md": "# 指南\n\n中文指南。\n",
        "docs-i18n/zh/user-guide/pending.md": (
            "---\ntranslation_status: pending\n---\n\nDraft translation.\n"
        ),
        "docs-i18n/ja/overview.md": "# 概要\n\n日本語の概要。\n",
        "docs-i18n/ja/user-guide/guide.md": "# ガイド\n\n日本語のガイド。\n",
        "docs-i18n/ja/user-guide/pending.md": (
            "---\ntranslation_status: pending\n---\n\nDraft translation.\n"
        ),
        "hack/existing_hook.py": "# Shared documentation hook.\n",
        "hack/offline_hook.py": "# Offline documentation hook.\n",
    }
    for code, title, guide in [
        ("ru", "Обзор", "Русское руководство."),
        ("tr", "Genel bakış", "Türkçe kılavuz."),
    ]:
        files[f"docs-i18n/{code}/overview.md"] = f"# {title}\n"
        files[f"docs-i18n/{code}/user-guide/guide.md"] = f"# {title}\n\n{guide}\n"
        files[f"docs-i18n/{code}/user-guide/pending.md"] = (
            "---\ntranslation_status: pending\n---\n\nDraft translation.\n"
        )
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    shutil.copy2(REPOSITORY_ROOT / "hack/mkdocs_i18n.py", root / "hack/mkdocs_i18n.py")
    configs = {
        "mkdocs.yml": {
            "site_name": "GPUStack",
            "site_url": "https://docs.example.test/latest/",
            "theme": {
                "language": "en",
                "palette": [
                    {
                        "toggle": {
                            "name": "Switch to dark mode",
                            "icon": "material/weather-night",
                        }
                    }
                ],
            },
            "nav": [
                {"Overview": "overview.md"},
                {"User Guide": [{"Guide": "user-guide/guide.md"}]},
            ],
            "hooks": ["hack/existing_hook.py"],
            "extra": {"version": {"provider": "mike"}},
        },
        "docs-i18n/mkdocs/zh.yml": {
            "site_description": "中文文档",
            "nav_translations": {
                "Overview": "概览",
                "User Guide": "用户指南",
                "Guide": "指南",
            },
            "theme_translations": {"Switch to dark mode": "切换到深色模式"},
        },
        "docs-i18n/mkdocs/ja.yml": {
            "site_description": "日本語ドキュメント",
            "nav_translations": {
                "Overview": "概要",
                "User Guide": "ユーザーガイド",
                "Guide": "ガイド",
            },
            "theme_translations": {"Switch to dark mode": "ダークモードに切り替え"},
        },
        "mkdocs.offline.yml": {
            "INHERIT": "mkdocs.yml",
            "site_url": "",
            "repo_url": None,
            "edit_uri": None,
            "hooks": ["hack/offline_hook.py"],
            "theme": {"font": False},
            "extra": {"offline": True, "version": None},
        },
    }
    for code, title in [("ru", "Обзор"), ("tr", "Genel bakış")]:
        configs[f"docs-i18n/mkdocs/{code}.yml"] = {
            "nav_translations": {"Overview": title, "Guide": title}
        }
    for filename, config in configs.items():
        config_path = root / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )
    return root


def read_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_contents(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


class LanguageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {"a": {}, "link": {}}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in self.links and attributes.get("hreflang"):
            self.links[tag][attributes["hreflang"]] = attributes["href"]


def rendered_language_links(path):
    parser = LanguageLinks()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.links


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
def test_language_overlay_preserves_shared_assets_and_english_sources(
    source_tree, tmp_path, language
):
    original_sources = file_contents(source_tree)
    stage = tmp_path / language
    config = read_config(builder.prepare_language(source_tree, stage, language))

    assert file_contents(source_tree) == original_sources
    expected = {
        "en": "English guide.",
        "zh": "中文指南。",
        "ja": "日本語のガイド。",
        "ru": "Русское руководство.",
        "tr": "Türkçe kılavuz.",
    }[language]
    assert expected in (stage / "docs/user-guide/guide.md").read_text()
    assert (stage / "docs/assets/chart.svg").read_bytes() == (
        source_tree / "docs/assets/chart.svg"
    ).read_bytes()
    assert "English fallback." in (stage / "docs/user-guide/missing.md").read_text()
    assert "Published English." in (stage / "docs/user-guide/pending.md").read_text()
    assert (
        "Draft translation." not in (stage / "docs/user-guide/pending.md").read_text()
    )
    settings = config["extra"]["docs_i18n"]
    assert settings["language"] == language
    assert set(settings["translated_sources"]) == (
        {
            "overview.md",
            "user-guide/guide.md",
        }
        if language != "en"
        else set()
    )
    assert "hack/existing_hook.py" in config["hooks"]
    assert (stage / "hack/mkdocs_i18n.py").is_file()
    if language == "zh":
        assert config["nav"] == [
            {"概览": "overview.md"},
            {"用户指南": [{"指南": "user-guide/guide.md"}]},
        ]
        assert config["theme"]["palette"][0]["toggle"]["name"] == "切换到深色模式"
        assert config["site_url"] == "https://docs.example.test/latest/zh/"
    elif language == "ja":
        assert config["nav"] == [
            {"概要": "overview.md"},
            {"ユーザーガイド": [{"ガイド": "user-guide/guide.md"}]},
        ]
        assert (
            config["theme"]["palette"][0]["toggle"]["name"] == "ダークモードに切り替え"
        )
        assert config["site_url"] == "https://docs.example.test/latest/ja/"


@pytest.mark.parametrize("page_name", ["missing", "pending"])
@pytest.mark.parametrize("language", ["zh", "ja"])
def test_missing_or_pending_translation_shows_english_with_local_fallback_link(
    source_tree, tmp_path, page_name, language
):
    stage = tmp_path / "chinese"
    config = read_config(builder.prepare_language(source_tree, stage, language))
    source = f"user-guide/{page_name}.md"
    page = SimpleNamespace(
        file=SimpleNamespace(src_uri=source),
        url=f"user-guide/{page_name}/",
        meta={},
    )
    markdown = (stage / "docs" / source).read_text(encoding="utf-8")

    output = i18n.on_page_markdown(markdown, page=page, config=config, files=None)

    assert output.endswith(markdown)
    assert i18n.FALLBACK_NOTICES[language][0] in output
    assert f"[{i18n.FALLBACK_NOTICES[language][2]}]" in output
    link = i18n.language_links(page.url, language)[0]["link"]
    assert f"]({link})" in output
    assert urljoin(f"https://docs.example.test/2.3/{language}/{page.url}", link) == (
        f"https://docs.example.test/2.3/{page.url}"
    )


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_completed_translation_has_no_fallback_notice(source_tree, tmp_path, language):
    stage = tmp_path / "chinese"
    config = read_config(builder.prepare_language(source_tree, stage, language))
    page = SimpleNamespace(
        file=SimpleNamespace(src_uri="user-guide/guide.md"),
        url="user-guide/guide/",
        meta={},
    )
    markdown = (stage / "docs/user-guide/guide.md").read_text(encoding="utf-8")

    assert (
        i18n.on_page_markdown(markdown, page=page, config=config, files=None)
        == markdown
    )


@pytest.mark.parametrize(
    "language, translated, metadata, has_edit_url, expected_directory",
    [
        ("en", True, {}, True, "docs"),
        ("zh", True, {}, True, "docs-i18n/zh"),
        ("zh", False, {}, True, "docs"),
        ("zh", True, {"translation_status": "pending"}, True, "docs"),
        ("zh", True, {}, False, None),
        ("ja", True, {}, True, "docs-i18n/ja"),
        ("ja", False, {}, True, "docs"),
        ("ja", True, {"translation_status": "pending"}, True, "docs"),
        ("ja", True, {}, False, None),
        ("ru", True, {}, True, "docs-i18n/ru"),
        ("ru", False, {}, True, "docs"),
        ("tr", True, {}, True, "docs-i18n/tr"),
        ("tr", True, {"translation_status": "pending"}, True, "docs"),
    ],
)
def test_source_action_follows_the_displayed_translation(
    language, translated, metadata, has_edit_url, expected_directory
):
    source = "user-guide/guide.md"
    page = SimpleNamespace(
        file=SimpleNamespace(src_uri=source),
        url="user-guide/guide/",
        meta=metadata,
        edit_url=(
            "https://github.com/example/gpustack/edit/main/docs/user-guide/guide.md"
            if has_edit_url
            else None
        ),
    )
    config = {
        "extra": {
            "docs_i18n": {
                "language": language,
                "translated_sources": [source] if translated else [],
            }
        }
    }
    context = {}

    assert i18n.on_page_context(context, page=page, config=config, nav=None) is context

    expected = (
        "https://github.com/example/gpustack/edit/main/"
        f"{expected_directory}/user-guide/guide.md"
        if expected_directory
        else None
    )
    assert page.edit_url == expected


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ("# 中文内容\n", True),
        ("---\ntranslation_status: translated\n---\n中文内容\n", True),
        ("---\ntranslation_status: pending\n---\nDraft\n", False),
        ("\ufeff---\ntranslation_status: pending\n...\nDraft\n", False),
    ],
)
def test_translation_readiness(tmp_path, metadata, expected):
    source = tmp_path / "translation.md"
    source.write_text(metadata, encoding="utf-8")
    assert i18n.is_translated_source(source) is expected
    assert not i18n.is_translated_source(tmp_path / "missing.md")


@pytest.mark.parametrize("prefix", ["/", "/latest/", "/docs/2.3/"])
@pytest.mark.parametrize("page_url", ["", "overview/", "user-guide/deep/guide/"])
@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
def test_public_language_links_preserve_page_and_deployment_prefix(
    prefix, page_url, language
):
    current = prefix + i18n.language_prefix(language) + page_url
    links = i18n.language_links(page_url, language)

    assert {link["lang"] for link in links} == set(i18n.LANGUAGES)
    for link in links:
        assert not link["link"].startswith(("/", "http:", "https:"))
        expected = prefix + i18n.language_prefix(link["lang"]) + page_url
        assert urljoin("https://docs.example.test" + current, link["link"]) == (
            "https://docs.example.test" + expected
        )


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
@pytest.mark.parametrize("offline", [False, True])
def test_material_renders_page_relative_language_links_without_normalizing_again(
    source_tree, tmp_path, language, offline
):
    config_path = source_tree / "mkdocs.yml"
    config = read_config(config_path)
    config["theme"].update({"name": "material", "font": False})
    config["plugins"] = []
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    stage = tmp_path / language
    output = tmp_path / f"output-{language}"
    staged_config = builder.prepare_language(
        source_tree, stage, language, offline=offline
    )

    builder.build_language(staged_config, output)

    links = rendered_language_links(output / "user-guide/guide/index.html")
    page_url = "user-guide/guide/"
    base = "https://docs.example.test" + ("/help/" if offline else "/latest/")
    current = base
    current += i18n.language_prefix(language) + page_url
    for tag in ("a", "link"):
        assert set(links[tag]) == set(i18n.LANGUAGES)
        for target_language, link in links[tag].items():
            expected = base
            expected += i18n.language_prefix(target_language) + page_url
            assert urljoin(current, link) == expected

    error_links = rendered_language_links(output / "404.html")
    for tag in ("a", "link"):
        assert set(error_links[tag]) == set(i18n.LANGUAGES)
        for target_language, link in error_links[tag].items():
            for missing_url in (
                base + "missing/deep/page/",
                base + "zh/missing/deep/page/",
            ):
                expected = base + i18n.language_prefix(target_language)
                assert urljoin(missing_url, link) == expected


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
def test_material_edit_and_view_actions_point_to_the_displayed_source(
    source_tree, tmp_path, language
):
    config_path = source_tree / "mkdocs.yml"
    config = read_config(config_path)
    config["theme"].update(
        {
            "name": "material",
            "font": False,
            "features": ["content.action.edit", "content.action.view"],
        }
    )
    repository = "https://github.com/example/gpustack"
    config.update(
        {"repo_url": repository, "edit_uri": "edit/main/docs/", "plugins": []}
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    staged_config = builder.prepare_language(source_tree, tmp_path / language, language)

    builder.build_language(staged_config, output)

    class SourceActions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = set()

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            href = attributes.get("href", "")
            if tag == "a" and href.startswith(repository + "/"):
                if "/edit/" in href or "/raw/" in href:
                    self.links.add(href)

    for page_name in ("guide", "missing", "pending"):
        parser = SourceActions()
        page = output / f"user-guide/{page_name}/index.html"
        parser.feed(page.read_text(encoding="utf-8"))
        directory = (
            f"docs-i18n/{language}"
            if language != "en" and page_name == "guide"
            else "docs"
        )
        assert parser.links == {
            f"{repository}/edit/main/{directory}/user-guide/{page_name}.md",
            f"{repository}/raw/main/{directory}/user-guide/{page_name}.md",
        }


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
def test_offline_configuration_and_error_links_stay_under_help(
    source_tree, tmp_path, language
):
    stage = tmp_path / language
    stamp = "2026-10-19 (abc1234)"
    config = read_config(
        builder.prepare_language(
            source_tree, stage, language, offline=True, stamp=stamp
        )
    )

    assert config["INHERIT"] == "mkdocs.offline.yml"
    assert config["extra"]["docs_stamp"] == stamp
    assert config["extra"]["docs_i18n"]["offline"] is True
    offline_config = read_config(stage / "mkdocs.offline.yml")
    assert offline_config["extra"]["version"] is None
    assert offline_config["theme"]["font"] is False
    assert offline_config["repo_url"] is None
    assert offline_config["edit_uri"] is None
    assert "hack/offline_hook.py" in config["hooks"]
    assert (stage / "hack/offline_hook.py").is_file()
    assert config.get("site_url", offline_config["site_url"]) == ""

    context = {}
    assert (
        i18n.on_template_context(context, template_name="404.html", config=config)
        is context
    )
    links = config["extra"]["alternate"]
    assert {entry["lang"]: entry["link"] for entry in links} == {
        "en": "/help/",
        "zh": "/help/zh/",
        "ja": "/help/ja/",
        "ru": "/help/ru/",
        "tr": "/help/tr/",
    }
    for missing_page in ("/help/missing/nested/page/", "/help/zh/missing/page/"):
        for link in links:
            assert urljoin(missing_page, link["link"]) == link["link"]


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
@pytest.mark.parametrize("prefix", ["/", "/latest/", "/docs/2.3/"])
def test_public_error_language_links_ignore_unknown_request_depth(language, prefix):
    config = {
        "site_url": (
            "https://docs.example.test/"
            + prefix.lstrip("/")
            + i18n.language_prefix(language)
        ),
        "extra": {"docs_i18n": {"language": language, "offline": False}},
    }

    i18n.on_template_context({}, template_name="404.html", config=config)

    links = config["extra"]["alternate"]
    assert {entry["lang"]: entry["link"] for entry in links} == {
        "en": prefix,
        "zh": prefix + "zh/",
        "ja": prefix + "ja/",
        "ru": prefix + "ru/",
        "tr": prefix + "tr/",
    }
    for missing_url in (
        "https://docs.example.test" + prefix + "missing/nested/page/",
        "https://docs.example.test" + prefix + "zh/missing/nested/page/",
    ):
        for entry in links:
            expected = "https://docs.example.test" + prefix
            expected += i18n.language_prefix(entry["lang"])
            assert urljoin(missing_url, entry["link"]) == expected


def test_preview_build_and_live_reload_use_the_local_server_root(
    source_tree, monkeypatch
):
    import mkdocs.livereload

    original_sources = file_contents(source_tree)
    site_urls = []
    servers = []

    def fake_build(config_path, output):
        config = read_config(config_path)
        site_urls.append(config["site_url"])
        output.mkdir()
        (output / "index.html").write_text("preview")
        (output / "404.html").write_text("preview error page")

    class PreviewServer:
        def __init__(self, builder, host, port, root):
            self.builder = builder
            self.host = host
            self.port = port
            self.stopped = False
            self.watched = []
            servers.append(self)

        def watch(self, path):
            self.watched.append(Path(path).relative_to(source_tree).as_posix())

        def serve(self):
            self.builder()
            assert self.error_handler(404) == b"preview error page"

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr(builder, "build_language", fake_build)
    monkeypatch.setattr(mkdocs.livereload, "LiveReloadServer", PreviewServer)

    builder.serve_docs(source_tree, "127.0.0.1:8123")

    assert (
        site_urls
        == [
            f"http://127.0.0.1:8123/{i18n.language_prefix(language)}"
            for language in i18n.LANGUAGES
        ]
        * 2
    )
    assert len(servers) == 1
    assert (servers[0].host, servers[0].port) == ("127.0.0.1", 8123)
    assert set(servers[0].watched) == {
        "docs",
        "docs-i18n",
        "mkdocs.yml",
        "mkdocs.offline.yml",
        "hack",
    }
    assert servers[0].stopped
    assert file_contents(source_tree) == original_sources


@pytest.mark.parametrize("offline", [False, True])
def test_publish_replaces_old_site_after_all_languages_pass(
    source_tree, tmp_path, monkeypatch, offline
):
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "index.html").write_text("old documentation")
    (destination / "obsolete.html").write_text("obsolete page")
    previous = file_contents(destination)
    events = []

    def fake_build(config_path, output):
        language = read_config(config_path)["extra"]["docs_i18n"]["language"]
        assert file_contents(destination) == previous
        output.mkdir()
        (output / "index.html").write_text(f"new {language} documentation")
        events.append(("build", language))

    def fake_check(root, output):
        assert root == source_tree
        assert file_contents(destination) == previous
        events.append(("check", output.parent.name))

    monkeypatch.setattr(builder, "build_language", fake_build)
    monkeypatch.setattr(builder, "check_offline_site", fake_check)

    result = builder.build_docs(
        source_tree,
        offline=offline,
        stamp="2026-10-19 (abc1234)",
        destination=destination,
    )

    assert result == destination
    assert (destination / "index.html").read_text() == "new en documentation"
    assert (destination / "zh/index.html").read_text() == "new zh documentation"
    assert (destination / "ja/index.html").read_text() == "new ja documentation"
    assert not (destination / "obsolete.html").exists()
    expected = [("build", language) for language in i18n.LANGUAGES]
    if offline:
        expected = [
            (event, language)
            for language in i18n.LANGUAGES
            for event in ("build", "check")
        ]
    assert events == expected


@pytest.mark.parametrize(
    "failure", ["en-build", "zh-build", "zh-check", "ja-build", "ja-check"]
)
def test_build_or_validation_failure_preserves_previous_bilingual_site(
    source_tree, tmp_path, monkeypatch, failure
):
    destination = tmp_path / "published"
    (destination / "zh").mkdir(parents=True)
    (destination / "index.html").write_text("old English")
    (destination / "zh/index.html").write_text("旧中文")
    previous = file_contents(destination)

    def fake_build(config_path, output):
        language = read_config(config_path)["extra"]["docs_i18n"]["language"]
        output.mkdir()
        (output / "index.html").write_text(f"partial {language} build")
        if failure == f"{language}-build":
            raise RuntimeError("documentation build failed")

    def fake_check(root, output):
        if failure == f"{output.parent.name}-check":
            raise RuntimeError("offline asset validation failed")

    monkeypatch.setattr(builder, "build_language", fake_build)
    monkeypatch.setattr(builder, "check_offline_site", fake_check)

    with pytest.raises(RuntimeError, match="failed"):
        builder.build_docs(
            source_tree,
            offline=failure.endswith("-check"),
            stamp="2026-10-19 (abc1234)",
            destination=destination,
        )

    assert file_contents(destination) == previous
