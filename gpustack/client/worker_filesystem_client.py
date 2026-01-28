import asyncio
import json
import logging
from typing import Dict

import aiohttp
from gpustack.schemas.filesystem import FileExistsResponse
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.utils.network import use_proxy_env_for_url
from gpustack import envs

logger = logging.getLogger(__name__)

_TIMEOUT = 15
_GGUF_PARSE_TIMEOUT = 90


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
        await self._http_client.close()
        await self._http_client_no_proxy.close()
        await self._connector.close()

    async def read_model_config(
        self,
        worker: Worker,
        path: str,
    ) -> Dict:
        """
        Read and parse a config file on a worker.

        Args:
            worker: The worker to query
            path: The file path to read

        Returns:
            Parsed config as dict
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/model-config"
        params = {"path": path}
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5)

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

                config_data = await resp.json()
                return config_data
        except Exception as e:
            logger.error(f"Error reading config file on worker {worker.name}: {e}")
            raise

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
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/file-exists"
        params = {"path": path}
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5)

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
        except Exception as e:
            logger.error(f"Error checking path on worker {worker.name}: {e}")
            raise

    async def get_model_weight_size(
        self,
        worker: Worker,
        path: str,
    ) -> int:
        """
        Get the size of model weight files in a directory on a worker.

        Args:
            worker: The worker to query
            path: The directory path to scan

        Returns:
            The total size in bytes
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/model-weight-size"
        params = {"path": path}
        headers = {"Authorization": f"Bearer {worker.token}"}

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5)

        try:
            async with client.get(
                url, params=params, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to get model weight size on worker {worker.id}: "
                        f"status={resp.status}, error={error_text}"
                    )
                    raise aiohttp.ClientError(
                        f"Failed to get model weight size: status={resp.status}, error={error_text}"
                    )

                data = await resp.json()
                return data.get("size", 0)
        except Exception as e:
            logger.error(
                f"Error getting model weight size on worker {worker.name}: {e}"
            )
            raise

    async def parse_gguf(
        self,
        worker: Worker,
        model: Model,
        offload: str = "full",
        **kwargs,  # tensor_split, rpc override parameters
    ) -> Dict:
        """
        Parse a GGUF file on a worker using gguf-parser.

        Args:
            worker: The worker to query
            model: The Model object to parse
            offload: GPU offload strategy (full, partial, disable)
            **kwargs: Optional override parameters (tensor_split, rpc)

        Returns:
            Parsed GGUF output as dict (GGUFParserOutput structure)

        Raises:
            aiohttp.ClientError: If the request fails
        """
        url = f"http://{worker.advertise_address or worker.ip}:{worker.port}/files/parse-gguf"
        headers = {"Authorization": f"Bearer {worker.token}"}

        # Build request payload
        payload = {
            "model_dict": model.model_dump(),  # Serialize Model object
            "offload": offload,
        }

        # Add override parameters
        for key in ("tensor_split", "rpc"):
            if key in kwargs:
                payload[key] = kwargs[key]

        use_proxy_env = use_proxy_env_for_url(url)
        client = self._http_client if use_proxy_env else self._http_client_no_proxy
        timeout = aiohttp.ClientTimeout(total=_GGUF_PARSE_TIMEOUT, sock_connect=10)

        try:
            async with client.post(
                url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to parse GGUF on worker {worker.id}: "
                        f"status={resp.status}, error={error_text}"
                    )
                    raise aiohttp.ClientError(f"Failed to parse GGUF: {error_text}")

                response_data = await resp.json()

                if not response_data.get("success", False):
                    error = response_data.get("error", "Unknown error")
                    raise aiohttp.ClientError(f"GGUF parsing failed: {error}")

                # Parse JSON output
                output_str = response_data.get("output", "{}")
                return json.loads(output_str)

        except asyncio.TimeoutError:
            logger.error(f"Timeout parsing GGUF on worker {worker.name}")
            raise aiohttp.ClientError("GGUF parsing timed out")
        except aiohttp.ClientError:
            raise
        except Exception as e:
            logger.error(f"Error parsing GGUF on worker {worker.name}: {e}")
            raise aiohttp.ClientError(f"Error parsing GGUF: {str(e)}")
