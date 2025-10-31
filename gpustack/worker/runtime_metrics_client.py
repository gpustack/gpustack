import logging
from typing import Optional
import requests
from prometheus_client.parser import text_string_to_metric_families
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpustack.schemas.models import BackendEnum


logger = logging.getLogger(__name__)

BackendVersionAPI = {
    BackendEnum.VLLM.value: "version",
    BackendEnum.SGLANG.value: "get_server_info",
    BackendEnum.ASCEND_MINDIE.value: "info",
}


class Config:
    def __init__(
        self, timeout=3, max_retries=2, base_delay=1, max_delay=3, insecure_tls=True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.insecure_tls = insecure_tls


class Client:
    def __init__(self, config=None):
        self.config = config or Config()

    def fetch_metrics_from_endpoint(self, endpoint):
        url = f"http://{endpoint}/metrics"

        logger.trace(f"Fetching metrics from {url}")

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    timeout=self.config.timeout,
                    verify=not self.config.insecure_tls,
                )
                if resp.status_code == 200:
                    metrics = {}
                    for family in text_string_to_metric_families(resp.text):
                        metrics[family.name] = family
                    return metrics
                else:
                    logger.warning(
                        f"[{endpoint}] Attempt {attempt + 1}: Bad status {resp.status_code}"
                    )
            except Exception as e:
                logger.error(f"[{endpoint}] Attempt {attempt + 1}: Error {e}")
            # Exponential backoff
            if attempt < self.config.max_retries:
                delay = min(
                    self.config.base_delay * (2**attempt), self.config.max_delay
                )
                import time

                time.sleep(delay)
        return None

    def fetch_metrics_from_endpoints(self, endpoints, max_workers=16):
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.fetch_metrics_from_endpoint, ep): ep
                for ep in endpoints
            }
            for future in as_completed(futures):
                ep = futures[future]
                results[ep] = future.result()
        return results

    def fetch_runtime_version_from_endpoint(
        self, endpoint: str, runtime: str
    ) -> Optional[str]:
        path = BackendVersionAPI.get(runtime)
        if path is None:
            return None

        url = f"http://{endpoint}/{path}"
        try:
            resp = requests.get(
                url,
                timeout=self.config.timeout,
                verify=not self.config.insecure_tls,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("version", None)
            else:
                logger.warning(
                    f"[{endpoint}] Bad status {resp.status_code} when fetching {runtime} version from {url}"
                )
        except Exception as e:
            logger.error(
                f"[{endpoint}] Error {e} when fetching {runtime} version from {url}"
            )
        return None
