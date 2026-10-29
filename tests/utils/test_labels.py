import json

import pytest
from sqlalchemy import JSON, column, select
from sqlalchemy.dialects import mysql, postgresql

from gpustack.utils.labels import LabelSearch


@pytest.mark.parametrize(
    "labels,query,expected",
    [
        ({"region": "beijing"}, "region", True),
        ({"region": "beijing"}, "beijing", True),
        ({"region": "beijing"}, "BEI", True),
        ({"Region": "Beijing"}, "REGION", True),
        ({"region": "beijing"}, "region beijing", True),
        ({"Region": "Beijing-east"}, "BEI REGION", True),
        ({"region": "beijing"}, "  region  \tbeijing\n", True),
        ({"region": "beijing"}, "region\u3000beijing", True),
        ({"region": "beijing"}, "region region", True),
        ({"region": "shanghai"}, "region beijing", False),
        ({"city": "beijing"}, "region beijing", False),
        ({"region": "other", "city": "beijing"}, "region beijing", True),
        ({"region": "beijing", "env": "prod"}, "region prod", True),
        ({"region": "beijing"}, "region missing", False),
        ({"region": "beijing"}, "region bei jing", True),
        ({"rack_name": "bei_jing%"}, "rack _jing%", True),
        ({"region": "北京"}, "region 北京", True),
        ({}, "region beijing", False),
        (None, "region beijing", False),
        ({}, " \t\n", True),
        (None, "", True),
        ({"region": "beijing"}, "region=beijing", True),
        ({"region": "beijing"}, "region:beijing", True),
        ({"region": "beijing-east"}, "region:beijing", False),
        ({"Region": "Beijing"}, "region:beijing", False),
        ({"region": "other", "city": "beijing"}, "region:beijing", False),
        ({"region": ""}, "region:", True),
        ({"city": "New York"}, "city:New York", True),
        ({"city": "New York"}, "city=New York", True),
        ({"city": "New York"}, "city:New", False),
        ({"city": " New York "}, "city= New York ", True),
        ({}, "region:", False),
        ({"url": "http://host:8000?a=b"}, "url:http://host:8000?a=b", True),
        ({"url": "http://host:8000?a=b"}, "url=http://host:8000?a=b", True),
        ({"region": "beijing-east"}, "region=beijing", False),
        ({"Region": "Beijing"}, "region=beijing", False),
        ({"region": "beijing", "env": "prod"}, "region=beijing", True),
        ({"region": "other", "city": "beijing"}, "region=beijing", False),
        ({"region": ""}, "region=", True),
        ({}, "region=", False),
        ({}, "region", False),
        (None, "region", False),
        ({"region": "beijing"}, 'region": "beijing', False),
        ({"region": "beijing"}, "%", False),
        ({"region": "beijing"}, "_", False),
        ({"region": "bei_jing%"}, "_jing%", True),
        ({"topology.gpustack.ai/region": "北京"}, "北京", True),
        ({"topology.gpustack.ai/region": "北京"}, "gpustack.ai/", True),
        ({"a.b/c": "x=y"}, "a.b/c=x=y", True),
        ({'a"b': 'c\\d'}, 'a"b=c\\d', True),
    ],
)
def test_label_search_matches_keys_values_and_exact_pairs(labels, query, expected):
    assert LabelSearch(query).matches(labels) is expected


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
@pytest.mark.parametrize(
    "query,pair",
    [
        ("bei_jing%/", None),
        ("region=bei'jing", ("region", "bei'jing")),
        ("region:bei'jing", ("region", "bei'jing")),
        ('a"b=x=y', ('a"b', 'x=y')),
        ("url:http://host:8000?a=b", ("url", "http://host:8000?a=b")),
        ("url=http://host:8000?a=b", ("url", "http://host:8000?a=b")),
    ],
)
def test_label_sql_uses_bound_parameters(dialect, query, pair):
    condition = LabelSearch(query).condition(column("labels", JSON), dialect.name)
    compiled = select(condition).compile(dialect=dialect)
    sql = str(compiled).lower()
    assert query not in sql

    if dialect.name == "postgresql":
        assert "json_each_text(labels)" in sql
        if pair is not None:
            key, value = pair
            assert key in compiled.params.values()
            assert value in compiled.params.values()
        else:
            assert ".key) like" in sql
            assert ".value) like" in sql
            assert "bei/_jing/%//" in compiled.params.values()
    elif pair is not None:
        assert "json_contains(labels," in sql
        key, value = pair
        assert json.dumps({key: value}) in compiled.params.values()
    else:
        assert "json_search(lower(json_keys(labels))," in sql
        assert "json_search(lower(labels)," in sql
        assert "%bei/_jing/%//%" in compiled.params.values()
        # JSON_SEARCH's escape must be a SQL constant for prepared statements.
        assert ", '/')" in sql


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_label_sql_requires_every_term_across_labels(dialect):
    condition = LabelSearch("  REGION\tbei_jing%/  ").condition(
        column("labels", JSON), dialect.name
    )
    compiled = select(condition).compile(dialect=dialect)
    sql = str(compiled).lower()
    assert " and " in sql
    assert " or " in sql
    assert "region" not in sql
    assert "bei_jing" not in sql
    if dialect.name == "postgresql":
        assert sql.count("exists") == 2
        assert "region" in compiled.params.values()
        assert "bei/_jing/%//" in compiled.params.values()
    else:
        assert sql.count("json_search") == 4
        assert "%region%" in compiled.params.values()
        assert "%bei/_jing/%//%" in compiled.params.values()
