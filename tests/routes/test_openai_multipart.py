from typing import List

import pytest
from starlette.requests import Request

from gpustack.routes import openai


def _request(body: bytes, content_type: str) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/videos",
            "query_string": b"",
            "headers": [
                (b"content-type", content_type.encode()),
                (b"content-length", str(len(body)).encode()),
            ],
        },
        receive,
    )


class _WireWriter:
    def __init__(self):
        self.chunks: List[bytes] = []

    async def write(self, value: bytes):
        self.chunks.append(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [None, "url", "upload"])
async def test_video_form_stays_multipart_and_preserves_prompt_and_parameters(
    reference,
):
    fields = {
        "size": "720x1280",
        "seconds": "4",
        "seed": "2040801485",
        "model": "wan2.1-t2v-1.3b-diffusers",
        "prompt": (
            "A medieval knight standing on a mountain cliff during a storm, "
            "cape blowing in the wind, lightning in the background, "
            "epic cinematic lighting."
        ),
    }
    if reference == "url":
        fields["reference_url"] = "https://example.com/reference.png"
    boundary = "browser-video-boundary"
    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()
        for key, value in fields.items()
    ]
    image = b"\x89PNG\r\n\x1a\nreference-image"
    if reference == "upload":
        parts.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="input_reference"; '
                'filename="reference.png"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode()
            + image
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    request = _request(b"".join(parts), f"multipart/form-data; boundary={boundary}")

    form, model, stream = await openai.parse_form_data(request)
    assert model == fields["model"]
    assert stream is False
    headers, data = openai._prepare_proxy_request(request, None, form)
    assert all(key.lower() != "content-type" for key in headers)
    assert all(key.lower() != "content-length" for key in headers)

    payload = data()
    content_type = payload.headers["Content-Type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    writer = _WireWriter()
    await payload.write(writer)

    # The downstream parses the actual serialized bytes and generated boundary.
    forwarded = _request(b"".join(writer.chunks), content_type)
    received = await forwarded.form()
    assert {key: received[key] for key in fields} == fields
    if reference == "upload":
        uploaded = received["input_reference"]
        assert uploaded.filename == "reference.png"
        assert uploaded.content_type == "image/png"
        assert await uploaded.read() == image


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/"])
@pytest.mark.parametrize(
    "stream", ["false", "0", "no", "off", "FALSE", "true", "1", "yes", "on"]
)
async def test_video_multipart_stream_flag_is_parsed_as_a_boolean(path, stream):
    from gpustack.api.exceptions import BadRequestException

    boundary = "stream-flag-boundary"
    body = (
        "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            for key, value in {"model": "wan", "stream": stream}.items()
        )
        + f"--{boundary}--\r\n"
    )
    request = _request(body.encode(), f"multipart/form-data; boundary={boundary}")
    request.scope["path"] = path
    if stream.lower() in ("true", "1", "yes", "on"):
        with pytest.raises(BadRequestException) as error:
            await openai.parse_request_body(request)
        assert error.value.status_code == 400
        assert "do not support streaming" in error.value.message
    else:
        _, parsed_stream, _, form = await openai.parse_request_body(request)
        assert parsed_stream is False
        assert form is not None
