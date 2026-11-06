import json

import pytest
from sqlalchemy import JSON, column, select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from gpustack.utils.labels import LabelFilter


@pytest.mark.parametrize(
    "labels,query,expected",
    [
        ({"region": "beijing"}, "region=beijing", True),
        ({"region": "beijing-east"}, "region=beijing", False),
        ({"Region": "Beijing"}, "region=beijing", False),
        ({"Region": "Beijing"}, "Region=Beijing", True),
        ({"region": "other", "city": "beijing"}, "region=beijing", False),
        ({"region": ""}, "region=", True),
        ({}, "region=", False),
        (None, "region=beijing", False),
        ({"city": "New York"}, "city=New York", True),
        ({"city": " New York "}, "city= New York ", True),
        ({"region": "beijing"}, " region=beijing", False),
        ({"region": "beijing"}, "region=beijing ", False),
        (
            {"region": "beijing", "env": "production"},
            "region=beijing production",
            False,
        ),
        ({"url": "http://host:8000?a=b"}, "url=http://host:8000?a=b", True),
        ({"region": "bei_jing%"}, "region=bei_jing%", True),
        ({"region": "beijing"}, "region=bei_jing%", False),
        ({"a.b/c": "北京=x=y"}, "a.b/c=北京=x=y", True),
        ({'a"b': 'c\\d'}, 'a"b=c\\d', True),
    ],
)
def test_label_filter_matches_exact_pairs(labels, query, expected):
    assert LabelFilter.parse(query).matches(labels) is expected


@pytest.mark.parametrize(
    "query",
    ["region", "beijing", "region beijing", "region:beijing", "=beijing", " ", ""],
)
def test_label_filter_rejects_invalid_pairs(query):
    with pytest.raises(ValueError, match="key=value"):
        LabelFilter.parse(query)


@pytest.mark.parametrize(
    "dialect", [postgresql.dialect(), mysql.dialect(), sqlite.dialect()]
)
@pytest.mark.parametrize(
    "query,pair",
    [
        ("region=bei'jing", ("region", "bei'jing")),
        ('a"b=x=y', ('a"b', 'x=y')),
        ("url=http://host:8000?a=b", ("url", "http://host:8000?a=b")),
        ("region=", ("region", "")),
        ("a.b/c=北京=x=y", ("a.b/c", "北京=x=y")),
    ],
)
def test_label_filter_sql_uses_bound_parameters(dialect, query, pair):
    condition = LabelFilter.parse(query).condition(column("labels", JSON), dialect.name)
    compiled = select(condition).compile(dialect=dialect)
    sql = str(compiled).lower()
    assert query not in sql
    key, value = pair
    if dialect.name == "mysql":
        assert "json_contains(labels," in sql
        assert json.dumps({key: value}) in compiled.params.values()
    else:
        assert key in compiled.params.values()
        assert value in compiled.params.values()
        if dialect.name == "postgresql":
            assert "->>" in sql
        else:
            assert "json_each(labels)" in sql
