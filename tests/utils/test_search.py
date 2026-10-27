import pytest

from gpustack.utils.search import rank_matches

# A slice of real catalog names, in catalog order, so ranking assertions
# reflect what the /model-sets endpoint actually has to sort through.
CATALOG = [
    "Qwen3-0.6B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3.6-35B-A3B",
    "Qwen3.5-0.8B",
    "Qwen3.5-9B",
    "Qwen3.5-27B",
    "Qwen3.5-397B-A17B",
    "GLM-4.6",
    "GLM-4.7",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Deepseek-V3.2",
    "Deepseek-V3.2-Speciale",
    "MiniMax-M2.5",
    "FLUX.2-klein-9B",
]


def search(query, catalog=CATALOG):
    return rank_matches(catalog, query, key=lambda name: name)


@pytest.mark.parametrize(
    "query",
    [
        # Every spelling a user might reach for to mean one specific model.
        "qwen3.5    ",
        "   qwen3.5",
        "qwen3.5 9b",
        "qwen 3.5 9b",
        "9b qwen3.5",
        "qwen3.5-9b",
        "qwen3.5_9b",
        "qwen3.5   9b",
        "QWEN3.5 9B",
    ],
)
def test_separator_and_order_insensitive(query):
    assert "Qwen3.5-9B" in search(query)


def test_exact_name_ranks_first():
    # `9b` alone also matches FLUX.2-klein-9B, and `qwen3.5` alone matches the
    # whole Qwen3.5 family; together they must put the exact model on top.
    assert search("qwen3.5 9b")[0] == "Qwen3.5-9B"
    assert search("9b qwen3.5")[0] == "Qwen3.5-9B"
    assert search("Qwen3.5-9B")[0] == "Qwen3.5-9B"


def test_typing_order_does_not_change_the_result():
    # The decoy holds both tokens but with something between them, so it has to
    # rank below the name where they are adjacent -- whichever order they were
    # typed in. It is listed first, so falling back to catalog order on a score
    # tie would put it on top.
    catalog = ["Foo3.5-VL-9B-Instruct", "Foo3.5-9B"]
    assert search("foo3.5 9b", catalog) == search("9b foo3.5", catalog)
    assert search("9b foo3.5", catalog)[0] == "Foo3.5-9B"


def test_contiguous_match_outranks_scattered_tokens():
    # Ignoring separators, `35` also appears inside `Qwen3-235B-A22B`, which is
    # the kind of incidental match the relevance floor is there to remove.
    results = search("qwen 3.5")
    assert results[:3] == ["Qwen3.5-0.8B", "Qwen3.5-9B", "Qwen3.5-27B"]
    assert "Qwen3-235B-A22B-Instruct-2507" not in results


def test_ties_keep_catalog_order():
    # Nothing distinguishes these beyond the curated order they came in.
    assert search("gpt oss") == ["gpt-oss-20b", "gpt-oss-120b"]
    assert search("qwen3")[:4] == [
        "Qwen3-0.6B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]


def test_unit_match_outranks_mid_unit_match():
    # `20b` is a whole unit of gpt-oss-20b but only part of `120b`, far enough
    # below to be dropped rather than merely ranked lower.
    assert search("gpt oss 20b") == ["gpt-oss-20b"]
    # `9b` is a whole unit of both, so neither is promoted over the other.
    assert search("9b") == ["Qwen3.5-9B", "FLUX.2-klein-9B"]


def test_version_aligns_across_a_letter_digit_transition():
    # `Qwen3.5` has no separator between `3` and `5` once squashed, yet `3.5`
    # must still align there and outrank the `35` buried inside `235B`.
    results = search("3.5")
    assert results.index("Qwen3.5-9B") < results.index("Qwen3-235B-A22B-Instruct-2507")
    # `M2.5` does not contain a `3.5` at all, however it is squashed.
    assert "MiniMax-M2.5" not in results


def test_relevance_floor_drops_incidental_matches():
    # Combining tokens is what starts matching across unrelated parts of a
    # name, so the count -- not just the first page -- has to shed those.
    assert "Qwen3-235B-A22B-Instruct-2507" not in search("qwen 3.5")
    assert search("gpt oss 20b") == ["gpt-oss-20b"]


def test_single_token_search_keeps_partial_matches():
    # A lone token is an ordinary contains-search: a model whose family name
    # embeds the query is a real hit, even though the query lands mid-unit
    # there, so the floor must not reach it.
    catalog = ["DeepSeek-OCR", "PaddleOCR-VL-1.5", "LightOnOCR-2-1B"]
    assert search("ocr", catalog) == catalog


def test_dotted_version_is_not_a_wildcard():
    assert search("glm 4.6") == ["GLM-4.6"]
    assert search("deepseek v3.2") == ["Deepseek-V3.2", "Deepseek-V3.2-Speciale"]


def test_every_token_must_match():
    assert search("qwen3.5 405b") == []
    assert search("qwen3.5 glm") == []


def test_blank_query_does_not_filter():
    # A user who typed only spaces should see the full catalog, not nothing.
    for query in [None, "", "   ", "\t\n", "---", "  _.-  "]:
        assert search(query) == CATALOG


def test_unmatchable_query_returns_nothing():
    # Non-alphanumeric scripts survive tokenization, so they filter normally
    # instead of degrading into an empty (match-everything) query.
    assert search("通义千问") == []
    assert search("nonexistent-model") == []
