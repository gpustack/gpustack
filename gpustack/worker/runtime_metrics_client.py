import logging
from typing import Optional
import requests

from gpustack import envs
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.openmetrics.parser import (
    text_string_to_metric_families as openmetrics_text_string_to_metric_families,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpustack.schemas.models import BackendEnum


logger = logging.getLogger(__name__)

OPENMETRICS_CONTENT_TYPE = "application/openmetrics-text"

BackendVersionAPI = {
    BackendEnum.VLLM.value: ["version"],
    BackendEnum.SGLANG.value: ["server_info", "get_server_info"],
    BackendEnum.ASCEND_MINDIE.value: ["info"],
}


def parse_metrics_text(text: str, content_type: Optional[str] = None):
    """
    Parse an exposition response, picking the parser that matches its content type.

    Runtimes serving OpenMetrics (such as vLLM with the Rust frontend) declare a
    counter by its base name and suffix the samples with _total. The Prometheus
    text parser reports those samples as separate untyped families instead, which
    leaves every counter family empty and breaks the unified metric mapping.
    """
    # Media types are case-insensitive, so a runtime answering with
    # "Application/OpenMetrics-Text" must not be sent down the text path.
    if content_type and OPENMETRICS_CONTENT_TYPE in content_type.lower():
        try:
            return list(openmetrics_text_string_to_metric_families(text))
        except Exception as e:
            logger.warning(
                f"Failed to parse OpenMetrics exposition, falling back to the "
                f"Prometheus text format. Counter families will come back empty "
                f"and their samples as untyped ones, so counter-derived metrics "
                f"are expected to be missing until this parses: {e}"
            )
    try:
        return list(text_string_to_metric_families(text))
    except Exception as e:
        # Materialized here on purpose: left lazy, a malformed payload would
        # raise while the caller iterates and be retried as if it were a
        # request failure, which no amount of retrying can fix.
        logger.warning(f"Failed to parse metrics exposition: {e}")
        return []


class Config:
    def __init__(
        self, timeout=3, max_retries=2, base_delay=1, max_delay=3, insecure_tls=True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.insecure_tls = insecure_tls
        self.scheme = envs.GPUSTACK_INSTANCE_SCHEME


class Client:
    def __init__(self, config=None):
        self.config = config or Config()

    def fetch_metrics_from_endpoint(self, endpoint):
        url = f"{self.scheme}://{endpoint}/metrics"

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
                    for family in parse_metrics_text(
                        resp.text, resp.headers.get("Content-Type")
                    ):
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
        """
        Try to fetch the runtime version from all possible API paths. Return on first success.
        Log last error or warning for troubleshooting.
        """

        paths = BackendVersionAPI.get(runtime)
        if paths is None:
            return None

        error_msg = ""
        warning_msg = ""
        for path in paths:
            url = f"{self.scheme}://{endpoint}/{path}"
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
                    warning_msg = f"[{endpoint}] Bad status {resp.status_code} when fetching {runtime} version from {url}"
            except Exception as e:
                error_msg = (
                    f"[{endpoint}] Error {e} when fetching {runtime} version from {url}"
                )

        if error_msg:
            logger.error(error_msg)
        elif warning_msg:
            logger.warning(warning_msg)
        return None
