import logging
from typing import List, Optional

import aiohttp
from gpustack.schemas.filesystem import FileEntry, FileListResponse, FileExistsResponse
from gpustack.schemas.workers import Worker
from gpustack.utils.network import use_proxy_env_for_url
from gpustack import envs

logger = logging.getLogger(__name__)


class WorkerFilesystemClient:
    """Client for interacting with worker filesystem APIs."""

    def __init__(self):
        """Initialize the client and create HTTP clients."""
        self._connector = aiohttp.TCPConnector(
            limit=envs.TCP_CONNECTOR_LIMIT,
            force_close=True,
        )
        self._http_client = aiohttp.ClientSession(
            connector=self._connector, trust_env=True
        )
        self._http_client_no_proxy = aiohttp.ClientSession(connector=self._connector)

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all HTTP clients."""
        await self.close()

    async def close(self):
        """Close all HTTP clients and connector."""
        if self._http_client:
            await self._http_client.close()
        if self._http_client_no_proxy:
            await self._http_client_no_proxy.close()
        if self._connector:
            await self._connector.close()

    async def list_files(
        self,
        worker: Worker,
        path: str,
    ) -> List[FileEntry]:
        """
        List files and directories in the specified path on a worker.

        Args:
            worker: The worker to query
            path: The directory path to list

        Returns:
            List of file entries

        Raises:
            aiohttp.ClientError: If the request fails
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/list"
        params = {"path": path}
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

        try:
            async with client.get(
                url, params=params, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to list files on worker {worker.id}: "
                        f"status={resp.status}, error={error_text}"
                    )
                    raise aiohttp.ClientError(
                        f"Failed to list files: status={resp.status}, error={error_text}"
                    )

                data = await resp.json()
                response = FileListResponse.model_validate(data)
                return response.files

        except aiohttp.ClientError as e:
            logger.error(f"Error listing files on worker {worker.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing files on worker {worker.id}: {e}")
            raise aiohttp.ClientError(f"Unexpected error: {str(e)}")

    async def read_file(
        self,
        worker: Worker,
        path: str,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> bytes:
        """
        Read content of a file on a worker.

        Args:
            worker: The worker to query
            path: The file path to read
            offset: Offset in bytes
            length: Length in bytes to read

        Returns:
            File content as bytes

        Raises:
            aiohttp.ClientError: If the request fails
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/read"
        params = {"path": path, "offset": offset}
        if length is not None:
            params["length"] = length
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

        try:
            async with client.get(
                url, params=params, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to read file on worker {worker.id}: "
                        f"status={resp.status}, error={error_text}"
                    )
                    raise aiohttp.ClientError(
                        f"Failed to read file: status={resp.status}, error={error_text}"
                    )

                data = await resp.json()
                return data["content"]

        except aiohttp.ClientError as e:
            logger.error(f"Error reading file on worker {worker.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading file on worker {worker.id}: {e}")
            raise aiohttp.ClientError(f"Unexpected error: {str(e)}")

    async def path_exists(
        self,
        worker: Worker,
        path: str,
    ) -> FileExistsResponse:
        """
        Check if a path exists on a worker.

        Args:
            worker: The worker to query
            path: The path to check

        Returns:
            FileExistsResponse indicating if the path exists

        Raises:
            aiohttp.ClientError: If the request fails
        """
        url = (
            f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/exists"
        )
        params = {"path": path}
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

        try:
            async with client.get(
                url, params=params, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to check path on worker {worker.id}: "
                        f"status={resp.status}, error={error_text}"
                    )
                    raise aiohttp.ClientError(
                        f"Failed to check path: status={resp.status}, error={error_text}"
                    )

                data = await resp.json()
                return FileExistsResponse.model_validate(data)

        except aiohttp.ClientError as e:
            logger.error(f"Error checking path on worker {worker.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking path on worker {worker.id}: {e}")
            raise aiohttp.ClientError(f"Unexpected error: {str(e)}")
