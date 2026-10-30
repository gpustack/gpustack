"""Exact worker label filtering shared by list queries and watch streams."""

import json
from dataclasses import dataclass
from typing import Dict, Optional

from sqlalchemy import ColumnElement, func, select


@dataclass(frozen=True)
class LabelFilter:
    """Match an exact, case-sensitive label key and value."""

    key: str
    value: str

    @classmethod
    def parse(cls, query: str) -> "LabelFilter":
        """Split on the first equals sign, preserving the key and value."""
        key, separator, value = query.partition("=")
        if not separator or not key:
            raise ValueError("Label filter must use key=value with a non-empty key")
        return cls(key=key, value=value)

    def matches(self, labels: Optional[Dict[str, str]]) -> bool:
        """Return whether the label map contains this exact pair."""
        return (labels or {}).get(self.key) == self.value

    def condition(self, column: ColumnElement, dialect: str) -> ColumnElement[bool]:
        """Build a bound predicate applied before pagination and counting."""
        if dialect == "mysql":
            return func.json_contains(column, json.dumps({self.key: self.value})) == 1
        if dialect == "postgresql":
            return column[self.key].as_string() == self.value
        if dialect == "sqlite":
            entries = func.json_each(column).table_valued("key", "value")
            return (
                select(1)
                .select_from(entries)
                .where((entries.c.key == self.key) & (entries.c.value == self.value))
                .exists()
            )
        raise ValueError(f"Unsupported label filter dialect: {dialect}")
