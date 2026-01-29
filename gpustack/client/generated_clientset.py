import base64
from urllib.parse import urlparse
from .generated_http_client import HTTPClient
from typing import Optional

from .generated_worker_client import WorkerClient
from .generated_model_client import ModelClient
from .generated_model_instance_client import ModelInstanceClient
from .generated_model_file_client import ModelFileClient
from .generated_user_client import UserClient
from .generated_inference_backend_client import InferenceBackendClient
from .generated_benchmark_client import BenchmarkClient
from .generated_model_route_target_client import ModelRouteTargetClient

from gpustack.utils.network import use_proxy_env_for_url


class ClientSet:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = 60.0,
        enable_cache: bool = True,
    ):
        if headers is None:
            headers = {}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            base64_credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {base64_credentials}"

        self.base_url = base_url
        self.headers = headers

        verify = None
        parsed_url = urlparse(base_url)
        if parsed_url.hostname == "127.0.0.1" and parsed_url.scheme == "https":
            verify = False

        use_proxy_env = use_proxy_env_for_url(base_url)
        http_client = (
            HTTPClient(
                base_url=base_url,
                verify_ssl=verify,
                httpx_args={"trust_env": use_proxy_env},
            )
            .with_headers(headers)
            .with_timeout(timeout)
        )
        self.http_client = http_client

        self.workers = WorkerClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.models = ModelClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.model_instances = ModelInstanceClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.model_files = ModelFileClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.users = UserClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.inference_backends = InferenceBackendClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.benchmarks = BenchmarkClient(
            http_client,
            enable_cache=enable_cache,
        )
        self.model_route_targets = ModelRouteTargetClient(
            http_client,
            enable_cache=enable_cache,
        )
