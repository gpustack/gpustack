from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class HTTPException(Exception):
    def __init__(self, status_code: int, reason: str, message: str):
        self.status_code = status_code
        self.reason = reason
        self.message = message


def http_exception_factory(status_code: int, reason: str, default_message: str):
    class_name = reason + "Exception"
    return type(
        class_name,
        (HTTPException,),
        {
            "__init__": lambda self, message=default_message: super(
                self.__class__, self
            ).__init__(status_code, reason, message)
        },
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


def register_handlers(app: FastAPI):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.status_code, reason=exc.reason, message=exc.message
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
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


def is_error_response(e):
    if hasattr(e, "code"):
        code_value = getattr(e, "code")
        if isinstance(code_value, int) and code_value >= 400:
            return True
    return False

    # TODO unify api and gen_client schemas so that we can check by isinstance(e, ErrorResponse)


def is_already_exists(e):
    reason = getattr(e, "reason")
    if reason == "AlreadyExists":
        return True

    return False


def is_not_found(e):
    reason = getattr(e, "reason")
    if reason == "NotFound":
        return True

    return False


def is_unauthorized(e):
    reason = getattr(e, "reason")
    if reason == "Unauthorized":
        return True

    return False


def is_forbidden(e):
    reason = getattr(e, "reason")
    if reason == "Forbidden":
        return True

    return False


def is_invalid(e):
    reason = getattr(e, "reason")
    if reason == "Invalid":
        return True

    return False


def is_bad_request(e):
    reason = getattr(e, "reason")
    if reason == "BadRequest":
        return True

    return False


def is_internal_server_error(e):
    reason = getattr(e, "reason")
    if reason == "InternalServerError":
        return True

    return False


def is_service_unavailable(e):
    reason = getattr(e, "reason")
    if reason == "ServiceUnavailable":
        return True

    return False
