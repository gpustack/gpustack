import pytest
from gpustack.routes.draft_models import get_draft_models
from gpustack.routes.model_sets import filter_specs_by_gpu, get_model_sets
from gpustack.schemas.common import ListParams
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.model_sets import (
    DraftModel,
    GPUFilters,
    ModelSetPublic,
    ModelSpec,
)
from gpustack.schemas.models import SourceEnum


def make_model_spec(**kwargs):
    return ModelSpec(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="Qwen/Qwen3-0.6B", **kwargs
    )


@pytest.mark.parametrize(
    "case_name, gpus, model_specs, filtered_specs_expected",
    [
        (
            "filter by gpu vendor",
            [
                GPUDevice(vendor="nvidia", compute_capability="8.0"),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["amd"])
                ),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
            ],
        ),
        (
            "filter by gpu vendor ascend",
            [
                GPUDevice(vendor="ascend"),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
            ],
            [
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
            ],
        ),
        (
            "filter by gpu vendor and compute capability",
            [
                GPUDevice(vendor="nvidia", compute_capability="7.0"),
            ],
            [
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0"
                    ),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["amd"], compute_capability=">=7.0"),
                ),
                make_model_spec(
                    mode="throughput",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=8.0"
                    ),
                ),
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0,<=9.0"
                    ),
                ),
            ],
            [
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0,<=9.0"
                    ),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0"
                    ),
                ),
            ],
        ),
        (
            "filter by gpu vendor and CANN variant",
            [
                GPUDevice(vendor="ascend", arch_family="Ascend910B2"),
            ],
            [
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["nvidia"]),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="310p"),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="910b"),
                ),
                make_model_spec(
                    mode="throughput",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="310p"),
                ),
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="910b"),
                ),
                make_model_spec(
                    mode="any-ascend",
                    gpu_filters=GPUFilters(vendor=["ascend"]),
                ),
            ],
            [
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="910b"),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="910b"),
                ),
                make_model_spec(
                    mode="any-ascend",
                    gpu_filters=GPUFilters(vendor=["ascend"]),
                ),
            ],
        ),
        (
            "unmapped ascend SoC falls back to vendor-only",
            [
                GPUDevice(vendor="ascend", arch_family="AscendUnknownSoC"),
            ],
            [
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["ascend"], vendor_variant="910b"),
                ),
                make_model_spec(
                    mode="any-ascend",
                    gpu_filters=GPUFilters(vendor=["ascend"]),
                ),
            ],
            [
                make_model_spec(
                    mode="any-ascend",
                    gpu_filters=GPUFilters(vendor=["ascend"]),
                ),
            ],
        ),
        (
            "no gpu filters",
            [
                GPUDevice(vendor="amd", compute_capability=None),
            ],
            [
                make_model_spec(mode="standard", gpu_filters=None),
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
            ],
            [
                make_model_spec(mode="standard", gpu_filters=None),
            ],
        ),
    ],
)
def test_filter_specs_by_gpu(
    config, case_name, gpus, model_specs, filtered_specs_expected
):
    try:
        actual_specs = filter_specs_by_gpu(gpus, model_specs)
        assert actual_specs == filtered_specs_expected
    except AssertionError as e:
        print(f"Test case '{case_name}' failed.")
        raise e


# --- GET /model-sets and GET /draft-models: search, filters, pagination -------
# Both handlers read the catalog from the database and then search, filter and
# paginate it in Python; stubbing that read drives them without one.

CATALOG = [
    ModelSetPublic(id=1, name="Qwen3-235B-A22B-Instruct-2507", categories=["llm"]),
    ModelSetPublic(id=2, name="Qwen3.5-0.8B", categories=["llm"]),
    ModelSetPublic(id=3, name="Qwen3.5-9B", categories=["llm"]),
    ModelSetPublic(id=4, name="Qwen3.5-27B", categories=["llm"]),
    ModelSetPublic(id=5, name="Qwen3-Embedding-8B", categories=["embedding"]),
    ModelSetPublic(id=6, name="FLUX.2-klein-9B", categories=["image"]),
]


@pytest.fixture(autouse=True)
def stub_catalog(monkeypatch):
    async def model_sets(_session):
        return CATALOG

    async def draft_models(_session):
        return DRAFT_MODELS

    monkeypatch.setattr(
        "gpustack.routes.model_sets.list_catalog_model_sets", model_sets
    )
    monkeypatch.setattr(
        "gpustack.routes.draft_models.get_catalog_draft_models", draft_models
    )


def params(page=1, perPage=100):
    return ListParams(page=page, perPage=perPage, watch=False, sort_by=None)


async def call(**kwargs):
    kwargs.setdefault("params", params())
    kwargs.setdefault("session", None)
    # Both are declared with FastAPI defaults, which are marker objects rather
    # than None when the handler is called directly.
    kwargs.setdefault("search", None)
    kwargs.setdefault("categories", None)
    return await get_model_sets(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search",
    ["qwen3.5 9b", "9b qwen3.5", "qwen3.5_9b", "qwen 3.5 9b", "  QWEN3.5-9B  "],
)
async def test_search_finds_the_model_however_it_is_typed(search):
    result = await call(search=search)
    assert [item.name for item in result.items] == ["Qwen3.5-9B"]
    assert result.pagination.total == 1


@pytest.mark.asyncio
async def test_search_ranks_and_excludes_incidental_matches():
    result = await call(search="qwen 3.5")
    names = [item.name for item in result.items]
    # `35` also sits inside `235B`, and the total is what feeds pagination, so
    # it has to count only the relevant ones.
    assert names == ["Qwen3.5-0.8B", "Qwen3.5-9B", "Qwen3.5-27B"]
    assert result.pagination.total == 3


@pytest.mark.asyncio
async def test_blank_search_returns_the_whole_catalog():
    # Whitespace-only input reaches the route as a truthy string; it has to
    # behave as no query rather than matching nothing.
    for search in ["   ", "\t", None]:
        result = await call(search=search)
        assert result.pagination.total == len(CATALOG)


@pytest.mark.asyncio
async def test_search_combines_with_categories():
    result = await call(search="9b", categories=["image"])
    assert [item.name for item in result.items] == ["FLUX.2-klein-9B"]
    assert result.pagination.total == 1


@pytest.mark.asyncio
async def test_search_survives_pagination():
    first = await call(search="qwen 3.5", params=params(page=1, perPage=2))
    second = await call(search="qwen 3.5", params=params(page=2, perPage=2))

    assert [item.name for item in first.items] == ["Qwen3.5-0.8B", "Qwen3.5-9B"]
    assert [item.name for item in second.items] == ["Qwen3.5-27B"]
    assert first.pagination.total == second.pagination.total == 3
    assert first.pagination.totalPage == 2


@pytest.mark.asyncio
async def test_search_with_no_match_is_empty():
    result = await call(search="nonexistent-model")
    assert result.items == []
    assert result.pagination.total == 0


# --- /draft-models: same catalog, same matching --------------------------------


def draft(name, repo):
    return DraftModel(
        name=name,
        algorithm="eagle3",
        source="huggingface",
        huggingface_repo_id=repo,
    )


DRAFT_MODELS = [
    draft("Qwen3-8B-EAGLE3", "Tengyunw/qwen3_8b_eagle3"),
    draft("Qwen3-30B-A3B-EAGLE3", "Tengyunw/qwen3_30b_moe_eagle3"),
    draft("Qwen3-235B-A22B-EAGLE3", "lmsys/Qwen3-235B-A22B-EAGLE3"),
    draft("gpt-oss-120b-EAGLE3", "lmsys/EAGLE3-gpt-oss-120b-bf16"),
]


async def call_drafts(**kwargs):
    kwargs.setdefault("params", params())
    kwargs.setdefault("session", None)
    kwargs.setdefault("search", None)
    kwargs.setdefault("algorithm", None)
    return await get_draft_models(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search",
    ["qwen3 235b eagle3", "eagle3 qwen3-235b", "qwen3_235b_a22b", "  QWEN3 235B  "],
)
async def test_draft_model_search_matches_like_model_sets(search):
    result = await call_drafts(search=search)
    assert [item.name for item in result.items] == ["Qwen3-235B-A22B-EAGLE3"]


@pytest.mark.asyncio
async def test_draft_model_blank_search_returns_everything():
    result = await call_drafts(search="   ")
    assert result.pagination.total == len(DRAFT_MODELS)


@pytest.mark.asyncio
async def test_draft_model_search_combines_with_algorithm():
    result = await call_drafts(search="gpt oss", algorithm="eagle3")
    assert [item.name for item in result.items] == ["gpt-oss-120b-EAGLE3"]
    result = await call_drafts(search="gpt oss", algorithm="medusa")
    assert result.items == []
