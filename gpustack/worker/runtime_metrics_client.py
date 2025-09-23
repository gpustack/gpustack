import logging
import requests
from prometheus_client.parser import text_string_to_metric_families
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


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

    def fetch_metrics(self, endpoint):
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

    def fetch_from_endpoints(self, endpoints, max_workers=16):
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.fetch_metrics, ep): ep for ep in endpoints}
            for future in as_completed(futures):
                ep = futures[future]
                results[ep] = future.result()
        return results
