import json
from urllib.parse import unquote

import aiohttp
import pytest

from gpustack.routes.video_parameters import adapt_h3_parameters, adapt_video_request
from tests.routes.test_openai_multipart import _request, _WireWriter
from tests.utils.model import new_model


def h3(backend="SGLang"):
    return new_model(
        1, "renamed-h3", huggingface_repo_id="MiniMaxAI/MiniMax-H3", backend=backend
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["SGLang", "vLLM"])
@pytest.mark.parametrize("multipart", [False, True])
async def test_common_video_request_is_encoded_for_the_selected_engine(
    backend, multipart
):
    parameters = dict(
        model="renamed-h3",
        prompt="A city at night",
        size="720x1280",
        seconds="4",
        seed="0",
    )
    form = None
    if multipart:
        form = aiohttp.FormData(default_to_multipart=True)
        for key, value in parameters.items():
            form.add_field(key, value)
    body, forwarded = adapt_video_request(
        h3(backend), None if multipart else parameters, form
    )
    if forwarded is not None:
        payload = forwarded()
        writer = _WireWriter()
        await payload.write(writer)
        downstream = await _request(
            b"".join(writer.chunks), payload.headers["Content-Type"]
        ).form()
        body = dict(downstream)
    assert body["prompt"] == parameters["prompt"]
    assert body["seed"] == "0"
    if backend == "SGLang":
        target = json.loads(body["target"]) if multipart else body["target"]
        assert body["task"] == "t2va"
        assert target == dict(short_edge=720, aspect_ratio="9:16", duration_seconds=4)
    else:
        assert forwarded is not None
        extra = json.loads(body["extra_params"])
        assert extra == dict(task="t2va", aspect_ratio="9:16", duration=4)
        assert "extra_body" not in body


@pytest.mark.parametrize("backend", ["SGLang", "vLLM"])
def test_explicit_target_is_preserved_and_invalid_json_is_rejected(backend):
    target = dict(short_edge=480, aspect_ratio="16:9", duration_seconds=5)
    result = adapt_h3_parameters(
        h3(backend),
        dict(
            size="720x1280",
            seconds=4,
            extra_body=json.dumps(dict(task="t2va", target=target)),
        ),
    )
    native = result if backend == "SGLang" else result["extra_params"]
    assert native["target"] == target
    for value in ("[1]", "null", "{"):
        with pytest.raises(ValueError, match="JSON object"):
            adapt_h3_parameters(h3(backend), {"extra_body": value})


@pytest.mark.parametrize("seconds", [0, 3, 4.5, 16, True, "NaN", "invalid"])
def test_invalid_common_duration_fails_before_submission(seconds):
    with pytest.raises(ValueError, match="seconds must be between"):
        adapt_h3_parameters(h3(), {"seconds": seconds})


def test_native_conflicts_and_text_only_reference_inputs_are_not_silently_ignored():
    with pytest.raises(ValueError, match="task must be one of"):
        adapt_h3_parameters(h3(), {"task": ""})
    with pytest.raises(ValueError, match="Conflicting"):
        adapt_h3_parameters(h3(), {"task": "t2va", "extra_body": '{"task":"fl2va"}'})
    with pytest.raises(ValueError, match="explicit task"):
        adapt_h3_parameters(h3(), {"reference_url": "https://example.com/image.png"})
    with pytest.raises(ValueError, match="text only"):
        adapt_h3_parameters(h3(), {"task": "t2va", "input_reference": b"image"})


def test_other_video_models_keep_their_original_request():
    model = new_model(1, "wan", huggingface_repo_id="org/wan", backend="vLLM")
    body = {"model": "wan", "extra_body": "engine-specific-value"}
    assert adapt_video_request(model, body, None) == (body, None)


def test_vllm_native_frame_count_and_reference_task_inference_are_preserved():
    result = adapt_h3_parameters(
        h3("vLLM"),
        {
            "num_frames": 124,
            "fps": 24,
            "extra_params": {"task": "t2va", "aspect_ratio": "16:9"},
        },
    )
    assert "duration" not in result["extra_params"]
    result = adapt_h3_parameters(
        h3("vLLM"), {"input_reference": b"image", "seconds": 5}
    )
    assert "task" not in result["extra_params"]
    assert result["input_reference"] == b"image"


def test_sglang_native_reference_target_can_derive_duration_from_audio():
    model = h3()
    model.backend_parameters = ["--model-variant=ref2va"]
    target = {"short_edge": 768, "aspect_ratio": "auto"}
    result = adapt_h3_parameters(
        model,
        {
            "task": "ref2va",
            "target": target,
            "conditions": [
                {
                    "type": "audio",
                    "uri": "https://example.com/audio.wav",
                    "role": "reference",
                }
            ],
        },
    )
    assert result["target"] == target
    assert "duration_seconds" not in result["target"]


@pytest.mark.asyncio
async def test_native_reference_request_preserves_file_bytes_and_filename():
    form = aiohttp.FormData(default_to_multipart=True)
    form.add_field("task", "fl2va")
    form.add_field(
        "input_reference",
        b"reference-image",
        filename="frame, 1.png",
        content_type="image/png",
    )
    _, forwarded = adapt_video_request(h3(), None, form)
    payload = forwarded()
    writer = _WireWriter()
    await payload.write(writer)
    downstream = await _request(
        b"".join(writer.chunks), payload.headers["Content-Type"]
    ).form()
    image = downstream["input_reference"]
    assert unquote(image.filename) == "frame, 1.png"
    assert image.content_type == "image/png"
    assert await image.read() == b"reference-image"
