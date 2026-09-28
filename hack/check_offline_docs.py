"""Reject external asset requests in HTML, allowing links and code examples."""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

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
    for page in sorted(Path(site_dir).rglob("*.html")):
        parser = ExternalAssets()
        parser.feed(page.read_text(encoding="utf-8"))
        parser.close()
        for reference in parser.references:
            print(f"{page}: external asset reference: {reference}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
