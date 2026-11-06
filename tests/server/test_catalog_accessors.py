import pytest
import yaml
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.catalog_source import (
    CatalogModelEntry,
    normalize_catalog_yaml,
    reconcile_catalog,
)
from gpustack.server import catalog
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.schemas.workers import GPUDeviceStatus

CATALOG = {
    "model_sets": [
        {
            "name": "Qwen3",
            "order": 2,
            "categories": ["llm"],
            "specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "Qwen/Qwen3-8B",
                    "mode": "standard",
                }
            ],
        },
        {
            "name": "Llama",
            "order": 1,
            "specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "meta/Llama-8B",
                    "mode": "standard",
                }
            ],
        },
    ],
    "draft_models": [
        {
            "name": "eagle",
            "algorithm": "eagle",
            "source": "huggingface",
            "huggingface_repo_id": "draft/eagle",
        }
    ],
}


@pytest.mark.asyncio
async def test_db_backed_catalog_accessors():
    """The read path (routes + scheduler) reads model sets / specs / drafts from
    the materialized CatalogModelEntry table via these accessors."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all, tables=[CatalogModelEntry.__table__]
        )

    async with AsyncSession(engine) as session:
        content = normalize_catalog_yaml(yaml.safe_dump(CATALOG))
        await reconcile_catalog(
            session, [SourceContent("file-a", SourceTypeEnum.FILE, content)]
        )

        # get_model_sets: sorted by order asc, source-stamped, ids populated.
        model_sets = await catalog.get_model_sets(session)
        assert [m.name for m in model_sets] == ["Llama", "Qwen3"]
        assert all(
            m.source_name == "file-a" and m.source_type == SourceTypeEnum.FILE
            for m in model_sets
        )
        qwen = next(m for m in model_sets if m.name == "Qwen3")
        assert qwen.id is not None

        # specs by set id; unknown id yields [].
        specs = await catalog.get_model_set_specs(session, qwen.id)
        assert [s.huggingface_repo_id for s in specs] == ["Qwen/Qwen3-8B"]
        assert await catalog.get_model_set_specs(session, 99999) == []

        # spec lookup by model_source_key (scheduler set_default_spec).
        spec = await catalog.get_catalog_spec_by_source_key(session, "Qwen/Qwen3-8B")
        assert spec is not None and spec.huggingface_repo_id == "Qwen/Qwen3-8B"
        assert await catalog.get_catalog_spec_by_source_key(session, "nope") is None

        # draft models.
        drafts = await catalog.get_catalog_draft_models(session)
        assert [d.name for d in drafts] == ["eagle"]
        assert drafts[0].huggingface_repo_id == "draft/eagle"

    await engine.dispose()


@pytest.mark.asyncio
async def test_default_recipe_filters_source_before_selecting_mode(monkeypatch):
    def spec(repo, mode, parameters, gpu_filters=None):
        return dict(
            source="huggingface",
            huggingface_repo_id=repo,
            backend="vLLM",
            mode=mode,
            backend_parameters=parameters,
            gpu_filters=gpu_filters,
        )

    entry = CatalogModelEntry(
        kind="model_set",
        name="model",
        payload={
            "name": "model",
            "specs": [
                spec("org/other", "standard", ["--other"]),
                spec("org/model", "throughput", ["--throughput"]),
                spec(
                    "org/model",
                    "standard",
                    ["--hardware"],
                    {"vendor": "nvidia"},
                ),
                spec("org/model", "standard", ["--fallback"]),
            ],
        },
    )

    async def entries(_session):
        return [entry]

    monkeypatch.setattr(catalog, "_model_set_entries", entries)
    gpus = [GPUDeviceStatus(vendor="nvidia", compute_capability="9.0")]
    selected = await catalog.get_catalog_spec_by_source_key(
        None, "org/model", gpus=gpus
    )
    assert selected.backend_parameters == ["--hardware"]
    selected = await catalog.get_catalog_spec_by_source_key(None, "org/model")
    assert selected.backend_parameters == ["--fallback"]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_env", [True, False])
async def test_mixed_gpu_defaults_require_equivalent_settings(monkeypatch, same_env):
    specs = [
        dict(
            source="huggingface",
            huggingface_repo_id="org/model",
            backend="vLLM",
            mode="standard",
            gpu_filters={"vendor": vendor},
            backend_parameters=["--max-model-len=8192"],
            env={"SETTING": "common" if same_env else vendor},
        )
        for vendor in ["nvidia", "amd"]
    ]

    async def entries(_session):
        return [
            CatalogModelEntry(
                kind="model_set",
                name="model",
                payload={"name": "model", "specs": specs},
            )
        ]

    monkeypatch.setattr(catalog, "_model_set_entries", entries)
    gpus = [GPUDeviceStatus(vendor=vendor) for vendor in ["nvidia", "amd"]]
    selected = await catalog.get_catalog_spec_by_source_key(
        None, "org/model", gpus=gpus
    )
    assert (selected is not None) is same_env
