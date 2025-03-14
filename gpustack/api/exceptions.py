from typing import Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel


class HTTPException(Exception):
    def __init__(self, status_code: int, reason: str, message: str):
        self.status_code = status_code
        self.reason = reason
        self.message = message


class OpenAIAPIException(HTTPException):
    pass


def http_exception_factory(
    status_code: int,
    reason: str,
    default_message: str,
):
    class_name = reason + "Exception"

    def init(self, message=default_message, is_openai_exception=False):
        if is_openai_exception:
            self.__class__.__bases__ = (OpenAIAPIException,)
        super(self.__class__, self).__init__(status_code, reason, message)

    return type(
        class_name,
        (HTTPException,),
        {"__init__": init},
    )


AlreadyExistsException = http_exception_factory(
    status.HTTP_409_CONFLICT, "AlreadyExists", "Already exists"
)
NotFoundException = http_exception_factory(
    status.HTTP_404_NOT_FOUND, "NotFound", "Not found"
)
UnauthorizedException = http_exception_factory(
    status.HTTP_401_UNAUTHORIZED, "Unauthorized", "Unauthorized"
)
ForbiddenException = http_exception_factory(
    status.HTTP_403_FORBIDDEN, "Forbidden", "Forbidden"
)
InvalidException = http_exception_factory(
    status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid", "Invalid input"
)
BadRequestException = http_exception_factory(
    status.HTTP_400_BAD_REQUEST, "BadRequest", "Bad request"
)
InternalServerErrorException = http_exception_factory(
    status.HTTP_500_INTERNAL_SERVER_ERROR,
    "InternalServerError",
    "Internal server error",
)
ServiceUnavailableException = http_exception_factory(
    status.HTTP_503_SERVICE_UNAVAILABLE, "ServiceUnavailable", "Service unavailable"
)
GatewayTimeoutException = http_exception_factory(
    status.HTTP_504_GATEWAY_TIMEOUT, "GatewayTimeout", "Gateway timeout"
)


def raise_if_response_error(response: httpx.Response):  # noqa: C901
    if response.status_code < status.HTTP_400_BAD_REQUEST:
        return

    try:
        error = ErrorResponse.model_validate(response.json())
    except Exception:
        raise HTTPException(response.status_code, "Unknown", response.text)

    if response.status_code == status.HTTP_404_NOT_FOUND:
        raise NotFoundException(error.message)

    if (
        response.status_code == status.HTTP_409_CONFLICT
        and error.reason == "AlreadyExists"
    ):
        raise AlreadyExistsException(error.message)

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise UnauthorizedException(error.message)

    if response.status_code == status.HTTP_403_FORBIDDEN:
        raise ForbiddenException(error.message)

    if response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        raise InvalidException(error.message)

    if response.status_code == status.HTTP_400_BAD_REQUEST:
        raise BadRequestException(error.message)

    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise InternalServerErrorException(error.message)

    if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        raise ServiceUnavailableException(error.message)

    if response.status_code == status.HTTP_504_GATEWAY_TIMEOUT:
        raise GatewayTimeoutException(error.message)

    raise HTTPException(error.code, error.reason, error.message)


class ErrorResponse(BaseModel):
    code: int
    reason: str
    message: str


error_responses = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


class OpenAIAPIError(BaseModel):
    message: str
    type: Optional[str] = None
    code: Optional[int] = None
    param: Optional[str] = None


class OpenAIAPIErrorResponse(BaseModel):
    error: OpenAIAPIError


openai_api_error_responses = {
    404: {"model": OpenAIAPIErrorResponse},
    409: {"model": OpenAIAPIErrorResponse},
    401: {"model": OpenAIAPIErrorResponse},
    403: {"model": OpenAIAPIErrorResponse},
    422: {"model": OpenAIAPIErrorResponse},
    400: {"model": OpenAIAPIErrorResponse},
    500: {"model": OpenAIAPIErrorResponse},
    503: {"model": OpenAIAPIErrorResponse},
}


def register_handlers(app: FastAPI):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.status_code,
                reason=exc.reason,
                message=exc.message,
            ).model_dump(),
        )

    @app.exception_handler(OpenAIAPIException)
    async def openai_api_exception_handler(request: Request, exc: OpenAIAPIException):
        """
        This handler is used to return error response in OpenAI API format.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "code": exc.status_code,
                    "type": exc.reason,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        message = f"{len(exc.errors())} validation errors:\n"
        for err in exc.errors():
            message += f"  {err}\n"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                reason="Invalid",
                message=message,
            ).model_dump(),
        )
