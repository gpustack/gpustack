import json
import logging
from typing import Dict

import aiohttp
from gpustack.schemas.filesystem import FileExistsResponse
from gpustack.schemas.models import Model
from gpustack.schemas.workers import Worker
from gpustack.server.worker_request import request_to_worker
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
        _, body = await request_to_worker(
            worker=worker,
            method="GET",
            path="files/model-config",
            proxy_client=self._http_client,
            no_proxy_client=self._http_client_no_proxy,
            params={"path": path},
            timeout=aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5),
        )
        return json.loads(body)

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
        _, body = await request_to_worker(
            worker=worker,
            method="GET",
            path="files/file-exists",
            proxy_client=self._http_client,
            no_proxy_client=self._http_client_no_proxy,
            params={"path": path},
            timeout=aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5),
        )
        return FileExistsResponse.model_validate_json(body)

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
        _, body = await request_to_worker(
            worker=worker,
            method="GET",
            path="files/model-weight-size",
            proxy_client=self._http_client,
            no_proxy_client=self._http_client_no_proxy,
            params={"path": path},
            timeout=aiohttp.ClientTimeout(total=_TIMEOUT, sock_connect=5),
        )
        return json.loads(body).get("size", 0)

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
        # Build request payload
        payload = {
            "model_dict": model.model_dump(),  # Serialize Model object
            "offload": offload,
        }

        # Add override parameters
        for key in ("tensor_split", "rpc"):
            if key in kwargs:
                payload[key] = kwargs[key]

        _, body = await request_to_worker(
            worker=worker,
            method="POST",
            path="files/parse-gguf",
            proxy_client=self._http_client,
            no_proxy_client=self._http_client_no_proxy,
            data=json.dumps(payload).encode(),
            timeout=aiohttp.ClientTimeout(total=_GGUF_PARSE_TIMEOUT, sock_connect=10),
        )

        response_data = json.loads(body)

        if not response_data.get("success", False):
            error = response_data.get("error", "Unknown error")
            raise aiohttp.ClientError(f"GGUF parsing failed: {error}")

        # Parse JSON output
        output_str = response_data.get("output", "{}")
        return json.loads(output_str)
