import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

import aiohttp

from gpustack.schemas.workers import Worker
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)

_TIMEOUT = 15


async def is_worker_reachable(
    worker: Worker,
    proxy_client: Optional[aiohttp.ClientSession] = None,
    no_proxy_client: Optional[aiohttp.ClientSession] = None,
    timeout_in_second: int = 10,
    retry_interval_in_second: int = 3,
) -> bool:
    """
    Check if a worker is reachable via a lightweight health check.

    Args:
        worker: Target worker.
        proxy_client: HTTP client with proxy.
        no_proxy_client: HTTP client without proxy.
        timeout_in_second: Timeout in seconds. Defaults to 10.
        retry_interval_in_second: Retry interval in seconds. Defaults to 3.

    Returns:
        True if worker responds with status < 500, False otherwise.
    """
    end_time = time.time() + timeout_in_second
    while time.time() < end_time:
        try:
            async with _request_to_worker(
                worker=worker,
                method="GET",
                path="healthz",
                proxy_client=proxy_client,
                no_proxy_client=no_proxy_client,
                timeout=aiohttp.ClientTimeout(total=2),
                raise_on_error=False,
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(retry_interval_in_second)
    return False


def _build_url(worker: Worker, path: str) -> str:
    """Build URL for a worker request."""
    hostname = (
        worker.advertise_address
        if worker.advertise_address and not worker.get_proxy_address()
        else worker.ip
    )
    return f"http://{hostname}:{worker.port}/{path.lstrip('/')}"


def _convert_params(params: Optional[Dict]) -> Optional[Dict]:
    """Convert bool params to str for aiohttp compatibility."""
    if params:
        return {
            k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()
        }
    return params


@asynccontextmanager
async def _request_to_worker(
    worker: Worker,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    path: str,
    proxy_client: Optional[aiohttp.ClientSession] = None,
    no_proxy_client: Optional[aiohttp.ClientSession] = None,
    params: Optional[Dict] = None,
    data: Optional[Union[bytes, AsyncIterator[bytes], aiohttp.FormData]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    raise_on_error: bool = True,
):
    """
    Async context manager for worker requests. Yields resp and auto-closes on exit.

    Raises:
        aiohttp.ClientError: If raise_on_error=True and response is non-2xx.
    """
    url = _build_url(worker, path)
    params = _convert_params(params)

    use_env_proxy = use_proxy_env_for_url(url)
    client = (
        proxy_client
        if use_env_proxy and worker.get_proxy_address() is None
        else no_proxy_client
    )

    if client is None:
        raise ValueError(
            f"No http client available: proxy_client={proxy_client}, no_proxy_client={no_proxy_client}"
        )

    req_headers = {"Authorization": f"Bearer {worker.token}"}
    if headers:
        req_headers.update(headers)

    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5)

    try:
        resp = await client.request(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=req_headers,
            timeout=timeout,
            proxy=worker.get_proxy_address(),
        )
        if resp.status >= 400 and raise_on_error:
            error_text = await resp.text()
            raise aiohttp.ClientError(
                f"Worker request failed: {worker.id} {method} {url} "
                f"status={resp.status}, error={error_text}"
            )
        yield resp
    except aiohttp.ClientError:
        raise
    except Exception as e:
        logger.error(f"Worker request failed: {worker.id} {method} {url}: {e}")
        raise
    finally:
        resp.close()


async def request_to_worker(
    worker: Worker,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    path: str,
    proxy_client: Optional[aiohttp.ClientSession] = None,
    no_proxy_client: Optional[aiohttp.ClientSession] = None,
    params: Optional[Dict] = None,
    data: Optional[Union[bytes, AsyncIterator[bytes], aiohttp.FormData]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    raise_on_error: bool = True,
) -> Tuple[aiohttp.ClientResponse, Optional[bytes]]:
    """
    Send a request to a worker.

    Returns:
        Tuple of (response, body_bytes). Body is None if no content.

    Raises:
        aiohttp.ClientError: If raise_on_error=True and response is non-2xx, or on other errors.
    """
    async with _request_to_worker(
        worker=worker,
        method=method,
        path=path,
        proxy_client=proxy_client,
        no_proxy_client=no_proxy_client,
        params=params,
        data=data,
        headers=headers,
        timeout=timeout,
        raise_on_error=raise_on_error,
    ) as resp:
        body = await resp.read()
        return resp, body if body else None


def _process_stream_line(line_bytes: bytes) -> str:
    """Process a line of bytes to ensure it is properly formatted for streaming."""
    line = line_bytes.decode("utf-8").strip()
    return line + "\n\n" if line else ""


async def _stream_response_chunks(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[str, None]:
    """Stream the response content in chunks, processing each line for SSE format."""
    chunk_size = 4096  # 4KB
    chunk_buffer = b""
    async for data in resp.content.iter_chunked(chunk_size):
        lines = (chunk_buffer + data).split(b'\n')
        chunk_buffer = lines.pop(-1)

        for line_bytes in lines:
            if line_bytes:
                yield _process_stream_line(line_bytes)

    if chunk_buffer:
        yield _process_stream_line(chunk_buffer)


async def stream_to_worker(
    worker: Worker,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    path: str,
    proxy_client: Optional[aiohttp.ClientSession] = None,
    no_proxy_client: Optional[aiohttp.ClientSession] = None,
    params: Optional[Dict] = None,
    data: Optional[Union[bytes, AsyncIterator[bytes], aiohttp.FormData]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    on_exception: Optional[
        Callable[[Exception, aiohttp.ClientTimeout], Tuple[str, int]]
    ] = None,
) -> AsyncGenerator[Tuple[Union[bytes, str], dict, int], None]:
    """
    Stream a request to a worker and yield response chunks.

    Yields tuples of (chunk, headers, status).

    Automatically handles:
    - URL construction (advertise_address or ip + port)
    - Proxy selection (based on use_proxy_env_for_url)
    - Authorization header

    Args:
        worker: Target worker
        method: HTTP method
        path: API path
        proxy_client: HTTP client with proxy
        no_proxy_client: HTTP client without proxy
        params: Query parameters
        data: Bytes, async iterator of bytes, or FormData
        headers: Additional headers
        timeout: Request timeout
        on_exception: Optional callback(exception, timeout) -> (error_msg, status_code).
            Called when an exception occurs during streaming. If not provided,
            the exception is raised.
    """
    try:
        async with _request_to_worker(
            worker=worker,
            method=method,
            path=path,
            proxy_client=proxy_client,
            no_proxy_client=no_proxy_client,
            params=params,
            data=data,
            headers=headers,
            timeout=timeout,
            raise_on_error=False,
        ) as resp:
            if resp.status >= 400:
                body = await resp.read()
                yield body, dict(resp.headers), resp.status
                return

            async for chunk in _stream_response_chunks(resp):
                yield chunk, dict(resp.headers), resp.status
    except Exception as e:
        logger.error(
            f"Worker stream failed: {worker.id} {method} {path}: {e}", exc_info=True
        )
        if on_exception is not None:
            error_msg, status_code = on_exception(e, timeout)
            yield error_msg, {}, status_code
        else:
            raise
