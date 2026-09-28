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
