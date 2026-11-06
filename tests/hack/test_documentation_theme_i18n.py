"""Shared error and outdated-version pages follow the selected language."""

import re
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import pytest
from jinja2 import DictLoader, Environment


OVERRIDES = Path(__file__).resolve().parents[2] / "docs/overrides"


@pytest.mark.parametrize(
    "language,prefix,banner,error",
    [
        ("en", "", "You're not viewing the latest version.", "404 - Not found"),
        ("zh", "zh/", "您正在查看的不是最新版本。", "404 - 页面未找到"),
        (
            "ja",
            "ja/",
            "最新バージョンのドキュメントではありません。",
            "404 - ページが見つかりません",
        ),
        (
            "ru",
            "ru/",
            "Вы просматриваете не последнюю версию документации.",
            "404 — Страница не найдена",
        ),
        (
            "tr",
            "tr/",
            "Belgelerin en son sürümünü görüntülemiyorsunuz.",
            "404 — Sayfa bulunamadı",
        ),
    ],
)
def test_localized_error_and_latest_link(language, prefix, banner, error):
    templates = {
        "base.html": (
            "{% block content %}{% endblock %}" "{% block outdated %}{% endblock %}"
        ),
        "main.html": (OVERRIDES / "main.html").read_text(encoding="utf-8"),
        "404.html": (OVERRIDES / "404.html").read_text(encoding="utf-8"),
    }
    environment = Environment(loader=DictLoader(templates), autoescape=True)

    output = environment.get_template("404.html").render(
        config={"theme": {"language": language}}, base_url="../.."
    )

    assert error in unescape(output)
    assert banner in unescape(output)
    href = re.search(r'href="([^"]+)"', output)[1]
    current = f"https://docs.example/docs/2.2/{prefix}user-guide/guide/"
    assert urljoin(current, href) == f"https://docs.example/docs/latest/{prefix}"
