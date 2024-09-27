from fastapi.responses import StreamingResponse
from starlette.types import Send


class StreamingResponseWithStatusCode(StreamingResponse):
    '''
    Variation of StreamingResponse that can dynamically decide the HTTP status code, based on the returns from the content iterator (parameter 'content').
    Expects the content to yield tuples of (content: str, status_code: int), instead of just content as it was in the original StreamingResponse.
    The parameter status_code in the constructor is ignored, but kept for compatibility with StreamingResponse.
    '''

    async def stream_response(self, send: Send) -> None:
        first_chunk_content, self.status_code = await self.body_iterator.__anext__()

        if not isinstance(first_chunk_content, bytes):
            first_chunk_content = first_chunk_content.encode(self.charset)

        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": first_chunk_content,
                "more_body": True,
            }
        )

        async for chunk_content, chunk_status in self.body_iterator:
            if chunk_status // 100 != 2:
                self.status_code = chunk_status
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
                return
            if not isinstance(chunk_content, bytes):
                chunk_content = chunk_content.encode(self.charset)
            await send(
                {"type": "http.response.body", "body": chunk_content, "more_body": True}
            )

        await send({"type": "http.response.body", "body": b"", "more_body": False})
