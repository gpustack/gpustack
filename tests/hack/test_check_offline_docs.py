import pytest

from hack.check_offline_docs import ExternalAssets, main


@pytest.mark.parametrize(
    "html",
    [
        '<pre><code>curl -sSLO https://raw.githubusercontent.com/'
        'gpustack/gpustack-operator/main/docs/migration/cleanup-v0.5-orphans.sh'
        '</code></pre>',
        '<a href="https://raw.githubusercontent.com/example/script.sh">Download</a>',
        '<pre>&lt;script src="https://unpkg.com/example.js"&gt;&lt;/script&gt;</pre>',
        '<script src="/help/assets/external/unpkg.com/example.js"></script>',
    ],
)
def test_documented_urls_and_vendored_assets_are_allowed(html):
    parser = ExternalAssets()
    parser.feed(html)
    assert not parser.references


@pytest.mark.parametrize(
    "html",
    [
        '<script src="https://unpkg.com/example.js"></script>',
        '<img src="//raw.githubusercontent.com/example/logo.png">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto">',
        '<link rel="preconnect" href="https://fonts.gstatic.com">',
        '<img srcset="local.png 1x, https://img.shields.io/badge/test 2x">',
        '<style>@import "https://fonts.googleapis.com/css";</style>',
        '<script>fetch("https://api.github.com/repos/gpustack/gpustack")</script>',
        '<script>fetch("https://api.github.com")</script>',
        '<div style="background: url(https://raw.githubusercontent.com/example/bg.png)"></div>',
    ],
)
def test_external_asset_requests_are_rejected(html):
    parser = ExternalAssets()
    parser.feed(html)
    assert parser.references


def test_gate_reports_the_failing_page(tmp_path, capsys):
    page = tmp_path / "index.html"
    page.write_text('<script src="https://unpkg.com/example.js"></script>')
    assert main(tmp_path) == 1
    assert str(page) in capsys.readouterr().err
    page.write_text('<code>curl https://unpkg.com/example.js</code>')
    assert main(tmp_path) == 0


@pytest.mark.parametrize(
    "reference",
    [
        'url(fonts/KaTeX_Main-Regular.woff2)',
        'url("fonts/KaTeX_Main-Regular.woff2?v=1#font")',
        "url('fonts/KaTeX_Main-Regular.woff2')",
        '@import "fonts/missing.css";',
        '@import url("fonts/missing.css");',
    ],
)
def test_gate_rejects_missing_css_dependencies(tmp_path, capsys, reference):
    stylesheet = tmp_path / "katex.css"
    stylesheet.write_text(f"@font-face {{ src: {reference}; }}")

    assert main(tmp_path) == 1
    assert str(stylesheet) in capsys.readouterr().err


@pytest.mark.parametrize(
    "reference",
    [
        'url(fonts/font.woff2)',
        'url("fonts/%66ont.woff2?v=1#font")',
        'url(/fonts/font.woff2)',
        'url(data:font/woff2;base64,AAAA)',
        'url("data:image/svg+xml,<svg>(icon)</svg>")',
        'url(#font)',
        'url("")',
        'local("System font")',
        '/* url(fonts/missing.woff2) */',
    ],
)
def test_gate_allows_local_and_embedded_css_resources(tmp_path, reference):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts/font.woff2").write_bytes(b"font")
    (tmp_path / "katex.css").write_text(f"@font-face {{ src: {reference}; }}")

    assert main(tmp_path) == 0


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.test/font.woff2",
        "//example.test/font.woff2",
        "file:///fonts/font.woff2",
    ],
)
def test_gate_rejects_external_css_resources(tmp_path, reference):
    (tmp_path / "katex.css").write_text(f"@font-face {{ src: url({reference}); }}")

    assert main(tmp_path) == 1


@pytest.mark.parametrize("reference", ["../outside.woff2", "%2e%2e/outside.woff2"])
def test_gate_rejects_css_paths_outside_site(tmp_path, reference):
    site = tmp_path / "site"
    site.mkdir()
    (tmp_path / "outside.woff2").write_bytes(b"outside")
    (site / "katex.css").write_text(f"@font-face {{ src: url({reference}); }}")

    assert main(site) == 1


def test_gate_rejects_css_symlinks_outside_site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    outside = tmp_path / "outside.woff2"
    outside.write_bytes(b"outside")
    (site / "font.woff2").symlink_to(outside)
    (site / "katex.css").write_text("@font-face { src: url(font.woff2); }")

    assert main(site) == 1


def test_gate_checks_dependencies_of_imported_stylesheets(tmp_path):
    (tmp_path / "base.css").write_text('@import "imported.css";')
    (tmp_path / "imported.css").write_text("@font-face { src: url(missing.woff2); }")

    assert main(tmp_path) == 1
