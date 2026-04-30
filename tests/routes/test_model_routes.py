from contextlib import asynccontextmanager

import pytest
from sqlalchemy import true

from gpustack.routes import model_routes
from gpustack.schemas.common import Pagination
from gpustack.schemas.model_routes import (
    ModelRouteListParams,
    ModelRoutesPublic,
    MyModel,
)


@pytest.mark.asyncio
async def test_get_model_routes_filters_categories_on_target_class(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    def fake_build_category_conditions(session, target_class, categories):
        captured["target_class"] = target_class
        captured["categories"] = categories
        return [true()]

    async def fake_paginated_by_query(**kwargs):
        captured["fields"] = kwargs["fields"]
        captured["extra_conditions"] = kwargs["extra_conditions"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(
        model_routes, "build_category_conditions", fake_build_category_conditions
    )
    monkeypatch.setattr(MyModel, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        categories=["image"],
        target_class=MyModel,
        user_id=123,
    )

    assert captured["target_class"] is MyModel
    assert captured["categories"] == ["image"]
    assert captured["fields"]["user_id"] == 123
    assert captured["extra_conditions"]
