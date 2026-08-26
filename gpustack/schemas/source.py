"""The content-source data model: what a source *is*, shared by the three tables
that hold one (catalog / community backend / runner).

Lives here rather than beside the source layer's behaviour
(``server/sources/core.py``) so the dependency runs one way: every consumer of a
source is a ``schemas`` model, and a model must never import from ``server``.
Fetching, ordering and materializing stay on the ``server`` side and import this.
"""

import enum
from typing import NamedTuple, Optional

from sqlalchemy import Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlmodel import SQLModel, Field as SQLField


class SourceTypeEnum(str, enum.Enum):
    """Where a source's content comes from.

    - FILE: inline
    - URL: fetched at PUT time
    - BUILTIN: packaged baseline, seeded by a controller
    - OFFICIAL: fetched by the probe — a layer of its own, not a BUILTIN rewrite

    Runner has no BUILTIN row; its baseline is the packaged catalog.
    """

    BUILTIN = "builtin"
    OFFICIAL = "official"
    FILE = "file"
    URL = "url"


class SourceMixin(SQLModel):
    """Shared columns for a content source of a materialized catalog: inline
    ``content`` (FILE) or a ``url`` fetched at PUT time (URL). ``id``/timestamps
    come from ``BaseModelMixin``.
    """

    # The natural key, unique-indexed by the migration: writers check-then-write,
    # which two leaders can interleave.
    name: str = SQLField(index=True, unique=True)
    source_type: SourceTypeEnum = SQLField(default=SourceTypeEnum.FILE)
    # ``sa_type`` (not ``sa_column``): each inheriting table needs its own Column.
    # LONGTEXT on MySQL, where ``TEXT`` caps at 64 KiB — every published document
    # is past it (the community-backend one by four times), and the refresh dies
    # on "Data too long for column 'content'". PostgreSQL and SQLite are unbounded.
    content: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(LONGTEXT(), "mysql")
    )
    url: Optional[str] = SQLField(default=None)
    # sha256 of ``content``: decides whether to write.
    content_hash: Optional[str] = SQLField(default=None)
    # sha256 of the *raw* remote document: decides whether to download.
    remote_hash: Optional[str] = SQLField(default=None)
    enabled: bool = SQLField(default=True)
    # Refresh cadence in hours, 0 = off. On for OFFICIAL, off for a user URL.
    auto_update_hours: int = SQLField(default=0)
    # Reserved for future multi-tenancy; excluded from serialized responses.
    owner_principal_id: Optional[int] = SQLField(default=None, exclude=True)


class SourceContent(NamedTuple):
    """One enabled source's normalized content, tagged with its identity —
    reconcile functions stamp it onto the rows they produce (origin display)."""

    name: str
    source_type: SourceTypeEnum
    content: str


# --- Icon validation -------------------------------------------------------
#
# An ``icon`` is an opaque URL (Helm ``Chart.yaml`` semantics): stored untouched,
# scheme-checked only, so active content is rejected at import time.

# SVG can carry script; the two script schemes run in an href — stored XSS.
_BLOCKED_PREFIXES = ("javascript:", "vbscript:", "data:image/svg+xml")

# Chars a browser drops before resolving a scheme (so ``java\tscript:`` runs).
_IGNORED_CHARACTERS = frozenset(chr(code) for code in range(0x21)) | {"\x7f"}


def validate_icon(icon: Optional[str]) -> Optional[str]:
    """Accept an absolute URL, a ``/``-rooted path or a raster ``data:`` URI;
    anything else raises ``ValueError`` (→ HTTP 400)."""
    if not icon:
        return icon
    probe = "".join(
        character for character in icon if character not in _IGNORED_CHARACTERS
    ).lower()
    if probe.startswith(_BLOCKED_PREFIXES):
        raise ValueError(f"icon '{icon}' uses a scheme that can carry active content")
    # An allowlist, not the inverse of _BLOCKED_PREFIXES: a bare ``data:`` accepts
    # text/html and friends, which the block list can't chase.
    # ``//host/x`` is protocol-relative — another origin, not a rooted path.
    rooted_path = probe.startswith("/") and not probe.startswith("//")
    if probe.startswith("data:image/") or "://" in probe or rooted_path:
        return icon
    raise ValueError(
        f"icon '{icon}' must be an absolute URL, a '/'-rooted path, "
        "or a data:image/ URI"
    )
