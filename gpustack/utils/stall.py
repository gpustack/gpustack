"""Terminal error payloads for an upstream that stopped responding (#6012).

A stall is reported differently depending on whether the downstream response
headers have already been committed:

- Before: as a real ``504``, which the official OpenAI and Anthropic SDKs retry
  automatically.
- After: as an in-stream terminal error event, because the status code is
  already fixed and simply dropping the connection would produce the silently
  truncated stream this exists to avoid.

The JSON shape matches ``openai_api_exception_handler`` in
``gpustack.api.exceptions`` so a stall detected mid-stream is indistinguishable
from the same stall reported as a status code.
"""

import json

from fastapi import status

STALL_ERROR_MESSAGE = "Upstream stalled: no response data within the proxy timeout"


def stall_error_json(message: str = STALL_ERROR_MESSAGE) -> bytes:
    """OpenAI-shaped error body for a stalled upstream."""
    return json.dumps(
        {
            "error": {
                "message": message,
                "code": status.HTTP_504_GATEWAY_TIMEOUT,
                "type": "GatewayTimeout",
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")


def stall_error_sse(message: str = STALL_ERROR_MESSAGE) -> bytes:
    """Terminal SSE frames for a stalled upstream: the error event, then [DONE].

    Ending the stream properly matters as much as the error itself -- a client
    that only ever sees the connection go away cannot tell a stall from a
    completed generation.
    """
    return b"data: " + stall_error_json(message) + b"\n\ndata: [DONE]\n\n"
