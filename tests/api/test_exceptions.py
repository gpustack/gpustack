import logging

from fastapi import FastAPI
import httpx
import pytest

from gpustack.api.exceptions import (
    ErrorResponse,
    HTTPException,
    OpenAIAPIException,
    http_exception_factory,
    register_handlers,
    raise_if_response_error,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)


@pytest.mark.parametrize(
    "name, given, expected",
    [
        (
            "valid case",
            {"code": 404, "reason": "NotFound", "message": "Resource not found"},
            True,
        ),
        (
            "valid case with type",
            {
                "code": 404,
                "reason": "NotFound",
                "type": "NotFound",
                "message": "Resource not found",
            },
            True,
        ),
        (
            "invalid type key",
            {"code": 404, "type": "NotFound", "message": "Resource not found"},
            False,
        ),
        (
            "invalid code type",
            {"code": "404", "reason": "NotFound", "message": "Resource not found"},
            False,
        ),
        (
            "invalid reason type",
            {"code": 404, "reason": 123, "message": "Resource not found"},
            False,
        ),
        ("missing message", {"code": 404, "reason": "NotFound"}, False),
        ("missing reason and message", {"code": 404}, False),
        (
            "missing code",
            {"reason": "NotFound", "message": "Resource not found"},
            False,
        ),
    ],
)
def test_error_response_model_validate(name, given, expected):
    try:
        _ = ErrorResponse.model_validate(given)
        assert (
            expected is True
        ), f"Case {name} expected validation to fail but succeeded"
    except Exception as e:
        assert (
            expected is False
        ), f"Case {name} expected validation to succeed but failed: {e}"


@pytest.mark.parametrize(
    "name, given, expected",
    [
        ("valid response", httpx.Response(status_code=200, content="..."), None),
        ("valid response without content", httpx.Response(status_code=204), None),
        (
            "client error response",
            httpx.Response(
                status_code=400,
                json={
                    "code": 400,
                    "reason": "BadRequest",
                    "message": "Invalid request",
                },
            ),
            BadRequestException("Invalid request"),
        ),
        (
            "server error response",
            httpx.Response(
                status_code=500,
                json={
                    "code": 500,
                    "reason": "InternalServerError",
                    "message": "Server error",
                },
            ),
            InternalServerErrorException("Server error"),
        ),
        (
            "not found response",
            httpx.Response(
                status_code=404,
                json={
                    "code": 404,
                    "reason": "NotFound",
                    "message": "Resource not found",
                },
            ),
            NotFoundException("Resource not found"),
        ),
        (
            "client error openai response",
            httpx.Response(
                status_code=400,
                json={
                    "error": {
                        "code": 400,
                        "type": "NotFound",
                        "message": "Invalid request",
                    }
                },
            ),
            BadRequestException("Invalid request"),
        ),
        (
            "server error openai response",
            httpx.Response(
                status_code=500,
                json={
                    "error": {
                        "code": 500,
                        "type": "InternalServerError",
                        "message": "Server error",
                    }
                },
            ),
            InternalServerErrorException("Server error"),
        ),
        (
            "not found openai response",
            httpx.Response(
                status_code=404,
                json={
                    "error": {
                        "code": 404,
                        "type": "NotFound",
                        "message": "Resource not found",
                    }
                },
            ),
            NotFoundException("Resource not found"),
        ),
    ],
)
def test_raise_if_response_error(name, given, expected):
    try:
        raise_if_response_error(given)
        assert expected is None, f"Case {name} expected get exception but none"
    except Exception as e:
        assert str(e) == str(
            expected
        ), f"Case {name} expected exception {expected} but got {e}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "paths",
    [
        ["/openai", "/plain"],
        ["/plain", "/openai", "/plain"],
        ["/openai", "/plain", "/openai", "/plain"],
    ],
)
async def test_exception_response_format_is_local_to_each_request(paths):
    error_type = http_exception_factory(404, "NotFound", "Not found")
    app = FastAPI()
    register_handlers(app)

    @app.get("/plain")
    async def plain():
        raise error_type("missing", details={"resource": "rank"})

    @app.get("/openai")
    async def openai():
        raise error_type("missing", is_openai_exception=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in paths:
            response = await client.get(path)
            assert response.status_code == 404
            if path == "/openai":
                assert response.json() == {
                    "error": {"message": "missing", "code": 404, "type": "NotFound"}
                }
            else:
                body = response.json()
                assert "error" not in body
                assert body["reason"] == "NotFound"
                assert body["details"] == {"resource": "rank"}


@pytest.mark.parametrize("form", ["positional", "positional_flag", "keywords"])
def test_openai_exception_preserves_catch_type_without_changing_other_instances(
    form,
):
    error_type = http_exception_factory(503, "Unavailable", "Unavailable")
    plain_before = error_type("plain before")
    details = {"worker_id": 3}
    args = () if form == "keywords" else ("upstream",)
    kwargs = {"is_openai_exception": True, "details": details}
    if form == "positional_flag":
        args, kwargs = ("upstream", True, details), {}
    elif form == "keywords":
        kwargs["message"] = "upstream"
        plain = error_type(message="plain keyword", details=details)
        assert type(plain) is error_type
        assert plain.message == "plain keyword"
        assert plain.details == details
    upstream = error_type(*args, **kwargs, log_level=logging.WARNING)
    plain_after = error_type("plain after")

    assert upstream.args == args
    assert str(plain_before) == "plain before"
    assert isinstance(upstream, error_type)
    assert isinstance(upstream, OpenAIAPIException)
    assert isinstance(upstream, HTTPException)
    assert not isinstance(plain_before, OpenAIAPIException)
    assert not isinstance(plain_after, OpenAIAPIException)
    assert not issubclass(error_type, OpenAIAPIException)
    assert upstream.status_code == 503
    assert upstream.message == "upstream"
    assert upstream.details == details
    assert upstream.log_level == logging.WARNING
    with pytest.raises(error_type):
        raise upstream
