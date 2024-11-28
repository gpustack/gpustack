from typing import List, Tuple
from fastapi.responses import StreamingResponse
from fastapi import status
from starlette.types import Send

from gpustack.api.exceptions import (
    OpenAIAPIError,
    OpenAIAPIErrorResponse,
)


class StreamingResponseWithStatusCode(StreamingResponse):
    '''
    Variation of StreamingResponse that can dynamically decide the HTTP status code, based on the returns from the content iterator (parameter 'content').
    Expects the content to yield tuples of (content: str, status_code: int), instead of just content as it was in the original StreamingResponse.
    The parameter status_code in the constructor is ignored, but kept for compatibility with StreamingResponse.
    '''

    async def stream_response(self, send: Send) -> None:
        try:
            first_chunk_content, headers, self.status_code = (
                await self.body_iterator.__anext__()
            )

            if not isinstance(first_chunk_content, bytes):
                first_chunk_content = first_chunk_content.encode(self.charset)

            asgi_headers: List[Tuple[bytes, bytes]] = [
                (key.encode("latin-1"), value.encode("latin-1"))
                for key, value in headers.items()
            ]
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": asgi_headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": first_chunk_content,
                    "more_body": True,
                }
            )

            async for chunk_content, _, _ in self.body_iterator:
                if not isinstance(chunk_content, bytes):
                    chunk_content = chunk_content.encode(self.charset)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk_content,
                        "more_body": True,
                    }
                )

            await send({"type": "http.response.body", "body": b"", "more_body": False})

        except StopAsyncIteration:
            self.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )

            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message="Service unavailable. Please retry your requests after a brief wait.",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    type="ServiceUnavailable",
                ),
            )

            await send(
                {
                    "type": "http.response.body",
                    "body": error_response.model_dump_json().encode(),
                    "more_body": False,
                }
            )
        except Exception as e:
            self.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )

            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=str(e),
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    type="InternalServerError",
                ),
            )

            await send(
                {
                    "type": "http.response.body",
                    "body": error_response.model_dump_json().encode(),
                    "more_body": False,
                }
            )
