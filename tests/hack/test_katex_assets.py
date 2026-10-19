"""The shared math renderer ships its scripts, stylesheet, and every font."""

import hashlib
from pathlib import Path

from hack.build_docs import config_field
from hack.check_offline_docs import css_references, main


ROOT = Path(__file__).resolve().parents[2]


def test_math_renderer_loads_local_assets_before_initializing():
    configuration = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    scripts = config_field(configuration, "extra_javascript")
    stylesheet = "assets/katex/katex.min.css"
    engine = "assets/katex/katex.min.js"
    auto_render = "assets/katex/contrib/auto-render.min.js"
    initializer = "javascripts/katex.js"

    assert stylesheet in config_field(configuration, "extra_css")
    assert (
        scripts.index(engine) < scripts.index(auto_render) < scripts.index(initializer)
    )
    for asset in (stylesheet, engine, auto_render):
        assert (ROOT / "docs" / asset).is_file()
    assert not any("unpkg.com/katex" in script for script in scripts)


def test_every_vendored_math_font_is_available():
    assets = ROOT / "docs/assets/katex"
    stylesheet = (assets / "katex.min.css").read_text(encoding="utf-8")
    fonts = set(css_references(stylesheet))

    assert len(fonts) == 60
    assert {Path(font).suffix for font in fonts} == {".woff2", ".woff", ".ttf"}
    assert all((assets / font).is_file() for font in fonts)
    assert (assets / "LICENSE").is_file()
    assert main(assets) == 0


def test_vendored_math_assets_match_the_verified_upstream_package():
    assets = ROOT / "docs/assets/katex"
    manifest = ROOT / "tests/fixtures/katex-0.19.0.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        expected[path] = digest
    actual = {
        path.relative_to(assets)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in assets.rglob("*")
        if path.is_file() and path.name != "README.txt"
    }

    assert actual == expected
