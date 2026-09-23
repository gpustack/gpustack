"""`GET /v2/benchmarks?search=` matches ANY of several names, not one substring.

Comparing runs means putting a specific handful of them side by side, and the
names of that handful rarely share a substring that excludes everything else.

The assertions below cover both consumers of the filter -- `extra_conditions`
for the paginated poll and `filter_func` for the watch stream -- because the
list subscribes to the stream right after it renders a page: a fix that reached
only one of the two would show rows the other then contradicts.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.routes import benchmarks as route


def _params(watch=False):
    return SimpleNamespace(watch=watch, page=1, perPage=100, order_by=None)


def _ctx():
    return SimpleNamespace(current_principal_id=None)


async def _conditions(**kwargs):
    """Run `_get_benchmarks` and hand back the `extra_conditions` it built."""
    captured = {}

    async def _fake_paginated(**call_kwargs):
        captured.update(call_kwargs)
        return SimpleNamespace(items=[])

    with (
        patch.object(
            route.Benchmark,
            "paginated_by_query",
            AsyncMock(side_effect=_fake_paginated),
        ),
        patch.object(route, "tenant_list_conditions", lambda *_a, **_k: []),
        patch.object(route, "async_session"),
    ):
        await route._get_benchmarks(ctx=_ctx(), params=_params(), **kwargs)

    return captured.get("extra_conditions", [])


async def _stream_filter(**kwargs):
    """Run the watch branch and hand back the `filter_func` it built."""
    captured = {}

    def _fake_streaming(**call_kwargs):
        captured.update(call_kwargs)
        return iter(())

    with (
        patch.object(route.Benchmark, "streaming", _fake_streaming),
        patch.object(route, "tenant_list_conditions", lambda *_a, **_k: []),
        patch.object(
            route, "_make_benchmark_visibility_filter", lambda _ctx: lambda _d: True
        ),
    ):
        await route._get_benchmarks(ctx=_ctx(), params=_params(watch=True), **kwargs)

    return captured["filter_func"]


@pytest.mark.parametrize(
    "search, expected",
    [
        ("pd-4k", ["pd-4k"]),
        ("pd-4k,nonpd-4k", ["pd-4k", "nonpd-4k"]),
        ("pd-4k nonpd-4k", ["pd-4k", "nonpd-4k"]),
        ("pd-4k, nonpd-4k,  b3", ["pd-4k", "nonpd-4k", "b3"]),
        # A trailing separator is what a half-typed list looks like; an empty
        # needle would match every row.
        ("pd-4k,", ["pd-4k"]),
        ("  pd-4k  ", ["pd-4k"]),
        ("pd-4k,pd-4k", ["pd-4k"]),
        ("", []),
        (None, []),
        (" , ", []),
    ],
)
def test_split_search_terms(search, expected):
    assert route.split_search_terms(search) == expected


@pytest.mark.parametrize(
    "name, terms, matches",
    [
        ("b-pd-4k", ["pd"], True),
        ("b-pd-4k", ["nonpd"], False),
        ("b-pd-4k", ["nonpd", "pd"], True),
        ("b-PD-4k", ["pd"], True),
        ("b-pd-4k", ["pd", "nonpd"], True),
        ("b-pd-4k", [], True),
        (None, ["pd"], False),
        (None, [], True),
    ],
)
def test_name_search_filter(name, terms, matches):
    assert route.name_search_filter(SimpleNamespace(name=name), terms) is matches


@pytest.mark.asyncio
async def test_several_names_or_together():
    conditions = await _conditions(search="pd-4k,nonpd-4k")

    sql = " ".join(str(c) for c in conditions)
    assert sql.count("lower(benchmarks.name) LIKE") == 2
    assert " OR " in sql


@pytest.mark.asyncio
async def test_one_name_is_a_plain_substring_match():
    conditions = await _conditions(search="pd-4k")

    sql = " ".join(str(c) for c in conditions)
    assert sql.count("lower(benchmarks.name) LIKE") == 1
    assert " OR " not in sql


@pytest.mark.asyncio
async def test_wildcards_in_a_name_stay_literal():
    """`_` is a LIKE wildcard; unescaped, `a_b` would also match `axb`."""
    conditions = await _conditions(search="a_b,c%d")

    params = [c.compile().params for c in conditions]
    values = [v for p in params for v in p.values()]
    assert values == ["%a\\_b%", "%c\\%d%"]


@pytest.mark.asyncio
async def test_no_search_leaves_the_name_unfiltered():
    conditions = await _conditions()

    assert not [c for c in conditions if "benchmarks.name" in str(c)]


def _row(name: str) -> SimpleNamespace:
    """A streamed row carrying only what the stream's filter reads."""
    return SimpleNamespace(
        name=name,
        gpu_summary=None,
        profile=None,
        model_name=None,
        load_type=None,
        target_mode=None,
    )


@pytest.mark.asyncio
async def test_the_stream_matches_the_same_rows_as_the_page():
    filter_func = await _stream_filter(search="pd-4k,nonpd-4k")

    assert filter_func(_row("b-pd-4k-1"))
    assert filter_func(_row("b-nonpd-4k-1"))
    assert not filter_func(_row("b-pd-8k-1"))


@pytest.mark.asyncio
async def test_the_stream_is_unfiltered_without_a_search():
    filter_func = await _stream_filter()

    assert filter_func(_row("anything"))
