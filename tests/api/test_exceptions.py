import httpx
import pytest

from gpustack.api.exceptions import (
    ErrorResponse,
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
