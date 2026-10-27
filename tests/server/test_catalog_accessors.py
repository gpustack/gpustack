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
