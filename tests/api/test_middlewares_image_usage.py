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


class _SSEStream:
    """Stands in for a downstream SSE ``StreamingResponse``."""

    def __init__(self, events):
        self.headers = {"content-type": "text/event-stream"}
        self.status_code = 200

        async def _iter():
            for event in events:
                payload = event if isinstance(event, str) else json.dumps(event)
                yield f"data: {payload}\n\n".encode()

        self.body_iterator = _iter()


def _image_event(index, finish_reason=None, progress=50.0):
    return {
        "created": 1,
        "object": "image.chunk",
        "data": [
            {
                "index": index,
                "object": "image",
                "progress": progress,
                "b64_json": "abc" if finish_reason else None,
                "finish_reason": finish_reason,
            }
        ],
    }


def _streaming_request(params):
    request = _request(params)
    request.state.stream = True
    return request


async def _drain(response):
    return [chunk async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_image_stream_without_usage_event_meters_finished_images(
    captured_metrics,
):
    request = _streaming_request(ImageRequestParams(n=3, size="512x512"))
    upstream = _SSEStream(
        [
            _image_event(0),
            _image_event(0, "stop", 100.0),
            _image_event(1),
            _image_event(1, "stop", 100.0),
            "[DONE]",
        ]
    )

    response = await middlewares.process_request(
        request, upstream, ImagesResponse, OperationEnum.IMAGE_GENERATION
    )
    await _drain(response)

    # Two of the three requested images finished.
    (metric,) = captured_metrics
    assert metric.image_count == 2
    assert metric.image_pixel_count == 2 * 512 * 512


@pytest.mark.asyncio
async def test_image_stream_with_usage_event_is_metered_once(captured_metrics):
    request = _streaming_request(ImageRequestParams(n=1, size="1024x1024"))
    usage_event = {**_image_event(0, "stop", 100.0), "usage": {"total_tokens": 0}}
    upstream = _SSEStream([_image_event(0), usage_event, "[DONE]"])

    response = await middlewares.process_request(
        request, upstream, ImagesResponse, OperationEnum.IMAGE_GENERATION
    )
    await _drain(response)

    (metric,) = captured_metrics
    assert metric.image_count == 1
    assert metric.image_pixel_count == 1024 * 1024


@pytest.mark.asyncio
async def test_failed_usage_event_recording_falls_back_at_stream_end(monkeypatch):
    captured = []
    attempts = 0

    async def _flaky_accumulate(metrics):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("buffer unavailable")
        captured.extend(metrics)

    monkeypatch.setattr(middlewares, "accumulate_gateway_metrics", _flaky_accumulate)
    request = _streaming_request(ImageRequestParams(n=1, size="512x512"))
    usage_event = {**_image_event(0, "stop", 100.0), "usage": {"total_tokens": 0}}
    upstream = _SSEStream([usage_event, "[DONE]"])

    response = await middlewares.process_request(
        request, upstream, ImagesResponse, OperationEnum.IMAGE_GENERATION
    )
    await _drain(response)

    (metric,) = captured
    assert metric.image_count == 1
    assert attempts == 2


@pytest.mark.asyncio
async def test_abandoned_image_stream_is_not_metered(captured_metrics):
    request = _streaming_request(ImageRequestParams(n=2, size="512x512"))
    upstream = _SSEStream([_image_event(0), _image_event(0, "stop"), "[DONE]"])

    response = await middlewares.process_request(
        request, upstream, ImagesResponse, OperationEnum.IMAGE_GENERATION
    )
    body = response.body_iterator
    await body.__anext__()
    # The client goes away before the stream ends.
    await body.aclose()

    assert captured_metrics == []
