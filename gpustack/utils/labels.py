"""Label searches shared by database queries and watch streams."""

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from sqlalchemy import ColumnElement, and_, func, literal_column, or_, select, true


def _split_label_query(query: str) -> Tuple[str, Optional[str]]:
    parts = re.split(r"[=:]", query, maxsplit=1)
    return parts[0], parts[1] if len(parts) == 2 else None


def _mysql_contains_term(column: ColumnElement, term: str) -> ColumnElement[bool]:
    # JSON_SEARCH requires a compile-time constant escape character.
    pattern = term
    for special in ("/", "%", "_"):
        pattern = pattern.replace(special, "/" + special)
    return or_(
        *[
            func.json_search(
                func.lower(document), "one", f"%{pattern}%", literal_column("'/'")
            ).is_not(None)
            for document in (func.json_keys(column), column)
        ]
    )


@dataclass(frozen=True)
class LabelSearch:
    """Match all whitespace-separated terms, or an exact key=value/key:value pair."""

    query: str

    def matches(self, labels: Optional[Dict[str, str]]) -> bool:
        """Return whether a label map satisfies this search."""
        labels = labels or {}
        key, value = _split_label_query(self.query)
        if value is not None:
            return key in labels and labels[key] == value

        return all(
            any(
                term in key.lower() or term in value.lower()
                for key, value in labels.items()
            )
            for term in self.query.lower().split()
        )

    def condition(self, column: ColumnElement, dialect: str) -> ColumnElement[bool]:
        """Build a predicate applied before pagination and to the total count.

        PostgreSQL and openGauss expose object entries through
        ``json_each_text``. MySQL and OceanBase search JSON strings through
        ``JSON_SEARCH``; searching ``JSON_KEYS`` covers the keys as well.
        """
        key, value = _split_label_query(self.query)
        terms = self.query.lower().split()
        if value is None and not terms:
            return true()

        if dialect == "mysql":
            if value is not None:
                return func.json_contains(column, json.dumps({key: value})) == 1
            return and_(*[_mysql_contains_term(column, term) for term in terms])

        if dialect == "postgresql":
            entries = func.json_each_text(column).table_valued("key", "value")
        elif dialect == "sqlite":
            entries = func.json_each(column).table_valued("key", "value")
        else:
            raise ValueError(f"Unsupported label search dialect: {dialect}")

        if value is not None:
            condition = (entries.c.key == key) & (entries.c.value == value)
            return select(1).select_from(entries).where(condition).exists()

        return and_(
            *[
                select(1)
                .select_from(entries)
                .where(
                    or_(
                        func.lower(entries.c.key).contains(term, autoescape=True),
                        func.lower(entries.c.value).contains(term, autoescape=True),
                    )
                )
                .exists()
                for term in terms
            ]
        )
