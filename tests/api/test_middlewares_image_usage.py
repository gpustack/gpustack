"""Image generation metering on the direct (non-gateway) inference path."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from starlette.responses import Response

from gpustack.api import middlewares
from gpustack.schemas.images import ImageRequestParams, ImagesResponse
from gpustack.schemas.model_usage import OperationEnum


def _request(params):
    state = SimpleNamespace(
        user=SimpleNamespace(id=5),
        model=SimpleNamespace(id=9, name="sd-turbo", cluster_id=1),
        model_route_id=21,
        start_time=datetime.now(timezone.utc),
        stream=False,
        image_request_params=params,
    )
    return SimpleNamespace(state=state, headers={})


@pytest.fixture
def captured_metrics(monkeypatch):
    captured = []

    async def _capture(metrics):
        captured.extend(metrics)

    monkeypatch.setattr(middlewares, "accumulate_gateway_metrics", _capture)
    return captured


class _JSONStream:
    """Stands in for the downstream ``StreamingResponse`` the middleware reads."""

    def __init__(self, payload):
        self.headers = {"content-type": "application/json"}
        self.status_code = 200
        body = json.dumps(payload).encode()

        async def _iter():
            yield body

        self.body_iterator = _iter()


@pytest.mark.asyncio
async def test_non_streaming_response_meters_the_images_returned(captured_metrics):
    request = _request(ImageRequestParams(n=4, size="512x512"))
    # The backend returned 2 of the 4 requested images.
    response = _JSONStream(
        {"created": 1, "data": [{"b64_json": "a"}, {"b64_json": "b"}]}
    )

    result = await middlewares.process_request(
        request, response, ImagesResponse, OperationEnum.IMAGE_GENERATION
    )

    assert isinstance(result, Response)
    (metric,) = captured_metrics
    assert metric.operation == OperationEnum.IMAGE_GENERATION
    assert metric.image_count == 2
    assert metric.image_pixel_count == 2 * 512 * 512
    assert metric.request_count == 1
    assert metric.completed is True


@pytest.mark.asyncio
async def test_without_a_response_count_the_requested_n_is_metered(
    captured_metrics,
):
    request = _request(ImageRequestParams(n=3, size="1024x1024"))

    await middlewares.record_model_usage(
        request, {"total_tokens": 0}, OperationEnum.IMAGE_GENERATION
    )

    (metric,) = captured_metrics
    assert metric.image_count == 3
    assert metric.image_pixel_count == 3 * 1024 * 1024


@pytest.mark.asyncio
async def test_text_operations_carry_no_image_units(captured_metrics):
    request = _request(ImageRequestParams(n=3, size="1024x1024"))

    await middlewares.record_model_usage(request, None, OperationEnum.CHAT_COMPLETION)

    (metric,) = captured_metrics
    assert metric.image_count == 0
    assert metric.image_pixel_count == 0
