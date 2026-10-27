"""Reject external HTML assets and missing or external CSS dependencies."""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlsplit

EXTERNAL_URL = re.compile(
    r"(?:https?:)?//(?:unpkg\.com|fonts\.googleapis\.com|fonts\.gstatic\.com|"
    r"buttons\.github\.io|img\.shields\.io|raw\.githubusercontent\.com|"
    r"api\.github\.com)(?=[:/?#\s\"'<>)]|$)",
    re.IGNORECASE,
)
RESOURCE_LINKS = {
    "stylesheet",
    "icon",
    "preload",
    "modulepreload",
    "prefetch",
    "preconnect",
    "dns-prefetch",
}
CSS_REFERENCE = re.compile(
    r"url\(\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|"
    r"(?P<bare>[^)\s]*))\s*\)|"
    r"@import\s+(?:\"(?P<import_double>[^\"]*)\"|'(?P<import_single>[^']*)')",
    re.IGNORECASE,
)


def css_references(stylesheet: str) -> Iterable[str]:
    """Extract resource URLs, excluding comments and CSS local() font names."""
    content = re.sub(r"/\*.*?\*/", "", stylesheet, flags=re.DOTALL)
    for match in CSS_REFERENCE.finditer(content):
        yield next(value for value in match.groups() if value is not None).strip()


def css_dependency_error(reference: str, stylesheet: Path, site: Path) -> Optional[str]:
    """Require CSS URLs to resolve to files within the generated site."""
    url = urlsplit(reference)
    if url.scheme == "data" or (not url.scheme and not url.netloc and not url.path):
        return None
    if url.scheme or url.netloc:
        return "external asset reference"
    path = unquote(url.path)
    target = (
        site / path.lstrip("/") if path.startswith("/") else stylesheet.parent / path
    ).resolve()
    if not target.is_relative_to(site):
        return "asset path outside site"
    if not target.is_file():
        return "missing local asset"
    return None


class ExternalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.inline_tag = None

    def check(self, value):
        if EXTERNAL_URL.search(value):
            self.references.append(value)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ("src", "srcset", "poster", "style"):
            self.check(attrs.get(name) or "")
        if tag == "object":
            self.check(attrs.get("data") or "")
        if tag == "link" and RESOURCE_LINKS.intersection(
            (attrs.get("rel") or "").lower().split()
        ):
            self.check(attrs.get("href") or "")
        if tag in ("image", "use"):
            self.check(attrs.get("href") or attrs.get("xlink:href") or "")
        if tag in ("script", "style"):
            self.inline_tag = tag

    def handle_endtag(self, tag):
        if tag == self.inline_tag:
            self.inline_tag = None

    def handle_data(self, data):
        if self.inline_tag:
            self.check(data)


def main(site_dir):
    failed = False
    site = Path(site_dir).resolve()
    for page in sorted(site.rglob("*.html")):
        parser = ExternalAssets()
        parser.feed(page.read_text(encoding="utf-8"))
        parser.close()
        for reference in parser.references:
            print(f"{page}: external asset reference: {reference}", file=sys.stderr)
            failed = True
    for stylesheet in sorted(site.rglob("*.css")):
        for reference in css_references(stylesheet.read_text(encoding="utf-8")):
            error = css_dependency_error(reference, stylesheet, site)
            if error:
                print(f"{stylesheet}: {error}: {reference}", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
