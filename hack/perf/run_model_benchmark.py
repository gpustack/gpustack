#!/usr/bin/env python3
"""
Automated GPUStack serving benchmark runner.

This script reads:
1. A model/run configuration YAML
2. A benchmark profile YAML

Then it drives the full benchmark lifecycle through GPUStack's HTTP API:
1. Create a model deployment for one run
2. Wait until the model instance is `running`
3. Optionally warm up the OpenAI-compatible endpoint
4. Create one or more benchmark jobs for the selected test cases
5. Watch benchmark state over SSE until completion
6. Save the final benchmark payload as JSON
7. Scale the model back to zero replicas before moving to the next run

Typical usage:

```bash
python3 hack/perf/run_model_benchmark.py \
  --config .cache/plan/benchmark/high-throughput/qwen_3.5_35b_fp8.yaml \
  --profile gpustack/assets/profiles_config/profiles_config.yaml \
  --gpustack-url https://YOUR_GPUSTACK \
  --gpustack-token $GPUSTACK_TOKEN \
  --cluster-id 1 \
  --output-dir benchmark_results
```

Run only a subset of runs:

```bash
python3 hack/perf/benchmark_serving.py \
  --config .../qwen_3.5_9b.yaml \
  --profile .../profiles_config.yaml \
  --gpustack-url https://YOUR_GPUSTACK \
  --gpustack-token $GPUSTACK_TOKEN \
  --cluster-id 1 \
  --run-names vllm-standard,sgl-throughput-bundle
```

Override test cases or request rates from the profile:

```bash
python3 hack/perf/benchmark_serving.py \
  --config .../qwen_3.5_122b_a10b_fp8.yaml \
  --profile .../profiles_config.yaml \
  --gpustack-url https://YOUR_GPUSTACK \
  --gpustack-token $GPUSTACK_TOKEN \
  --cluster-id 1 \
  --test-cases Throughput,Long\\ Context \
  --request-rates 1,4,8
```

Expected config YAML shape:

```yaml
model: "Qwen/Qwen3.5-35B-A3B-FP8"
source: "model_scope"  # or "huggingface"
health_check:
  init_delay: 60
  timeout: 1800
  interval: 5.0
warmup:
  num_requests: 10
test_cases:
  - name: Throughput
runs:
  - name: vllm-standard
    backend: vLLM
    backend_version: 0.17.1
    backend_parameters:
      - --reasoning-parser=qwen3
      - --max-model-len=32768
```

Expected profile YAML shape:

```yaml
profiles:
  - name: Throughput
    request_rate: 4
    total_requests: 100
    dataset_name: sharegpt
```
"""

import json
import logging
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm_benchmark")


class EngineType(Enum):
    """Supported inference engine types"""

    VLLM = "vLLM"
    SGLANG = "SGLang"
    TRTLLM = "TRT-LLM"


class Source(Enum):
    Huggingface = "huggingface"
    ModelScope = "model_scope"


@dataclass
class HealthCheck:
    init_delay: int = 60
    timeout: int = 30
    interval: float = 1.0


@dataclass
class Model:
    """Configuration for a model test run"""

    name: str
    test_cases: List[str]
    backend: EngineType
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = None
    envs: Optional[Dict[str, str]] = None
    args: Optional[List[str]] = None
    health_check: Optional[HealthCheck] = None
    warmup_num_requests: Optional[int] = None
    stop_model_after_run: bool = True

    instance_name: Optional[str] = None
    model_id: Optional[int] = None
    model_name: Optional[str] = None
    benchmark_id: Optional[int] = None
    benchmark_name: Optional[str] = None
    request_rates: Optional[List[int]] = None


class EngineManager:
    """
    Translate one benchmark run into concrete GPUStack API operations.

    Important behavior:
    - one `Model` dataclass instance corresponds to one deployment/run in the YAML
    - each run may execute multiple benchmark profiles (`test_cases`)
    - each benchmark result is written to one JSON file under `output_dir`
    """

    def __init__(
        self,
        model: str,
        source: str,
        gpustack_url: str,
        gpustack_token: str,
        cluster_id: int,
        output_dir: str = "benchmark_results",
    ):
        self.model_repo_id = model
        self.source = source
        self.gpustack_url = gpustack_url.rstrip("/")
        self.gpustack_token = gpustack_token
        self.cluster_id = int(cluster_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gpustack_token}",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = 60,
    ) -> Any:
        url = f"{self.gpustack_url}{path}"
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"
        logger.debug("HTTP %s %s", method, url)
        body = None
        headers = self._headers()
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            return urllib.request.urlopen(
                request,
                timeout=timeout,
                context=self.ssl_context,
            )
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{method} {url} failed: {exc.code} {details}") from exc

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "benchmark"

    def _repo_model_name(self) -> str:
        return self._slugify(self.model_repo_id.split("/")[-1])

    def _timestamp(self) -> str:
        return time.strftime("%Y%m%d%H%M%S")

    def _bounded_name(self, *parts: str, max_length: int = 63) -> str:
        """
        Build a readable, stable name that always fits Kubernetes-style label limits.
        Keeps a human-readable prefix and appends a short hash when truncation is needed.
        """
        base = "_".join(part for part in parts if part)
        if len(base) <= max_length:
            return base

        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        reserve = len(digest) + 1
        prefix = base[: max_length - reserve].rstrip("_.-")
        if not prefix:
            prefix = "benchmark"
        return f"{prefix}_{digest}"

    def _build_benchmark_name(
        self,
        model_name: str,
        profile_slug: str,
        request_rate: int,
        ts: str,
        max_length: int = 63,
    ) -> str:
        """
        Prefer a readable timestamped name when it fits.
        Fall back to the readable name without timestamp, then to a bounded hashed name.
        """
        readable_name = f"{model_name}_{profile_slug}_r{request_rate}"
        timestamped_name = f"{readable_name}_{ts}"
        if len(timestamped_name) <= max_length:
            return timestamped_name

        if len(readable_name) <= max_length:
            return readable_name

        return self._bounded_name(model_name, profile_slug, f"r{request_rate}", ts)

    def _iter_sse_payloads(self, response: Any) -> Iterator[Dict[str, Any]]:
        event_lines: List[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                if event_lines:
                    payload = "\n".join(event_lines)
                    event_lines = []
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug("Skip non-JSON SSE payload: %s", payload)
                continue

            if line.startswith("data:"):
                event_lines.append(line[5:].strip())
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skip non-JSON stream line: %s", line)

        if event_lines:
            payload = "\n".join(event_lines)
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                logger.debug("Skip trailing non-JSON SSE payload: %s", payload)

    def _get_model(self, model_id: int) -> Dict[str, Any]:
        with self._request("GET", f"/v2/models/{model_id}") as response:
            return json.loads(response.read().decode("utf-8"))

    def _list_models(
        self,
        *,
        search: Optional[str] = None,
        timeout: int = 60,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "perPage": 100,
            "page": 1,
            "cluster_id": self.cluster_id,
        }
        if search:
            params["search"] = search

        with self._request(
            "GET", "/v2/models", params=params, timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("items", [])

    def _list_model_instances(
        self, model_id: int, timeout: int = 60
    ) -> List[Dict[str, Any]]:
        with self._request(
            "GET",
            "/v2/model-instances",
            params={"model_id": model_id, "perPage": 100, "page": 1},
            timeout=timeout,
        ) as response:
            response = json.loads(response.read().decode("utf-8"))
        return response.get("items", [])

    def _result_path(self, benchmark_name: str) -> Path:
        filename = f"{self._slugify(benchmark_name)}.json"
        return self.output_dir / filename

    def _dump_result(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _is_retryable_request_error(self, exc: Exception) -> bool:
        return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))

    def _matches_model_payload(
        self,
        existing: Dict[str, Any],
        expected_name: str,
        payload: Dict[str, Any],
    ) -> bool:
        if existing.get("name") != expected_name:
            return False
        if existing.get("cluster_id") != self.cluster_id:
            return False
        if existing.get("backend") != payload.get("backend"):
            return False
        if existing.get("source") != payload.get("source"):
            return False
        if payload.get("source") == Source.Huggingface.value:
            return existing.get("huggingface_repo_id") == payload.get(
                "huggingface_repo_id"
            )
        if payload.get("source") == Source.ModelScope.value:
            return existing.get("model_scope_model_id") == payload.get(
                "model_scope_model_id"
            )
        return True

    def _find_existing_model(
        self,
        model_name: str,
        payload: Dict[str, Any],
        *,
        timeout: int = 30,
    ) -> Optional[Dict[str, Any]]:
        try:
            candidates = self._list_models(search=model_name, timeout=timeout)
        except Exception as exc:
            if self._is_retryable_request_error(exc):
                logger.warning("Failed to query existing model %s: %s", model_name, exc)
                return None
            raise

        for candidate in candidates:
            if self._matches_model_payload(candidate, model_name, payload):
                return candidate
        return None

    def _apply_model_identity(self, config: Model, payload: Dict[str, Any]) -> None:
        config.model_id = payload["id"]
        config.model_name = payload["name"]

    def _update_existing_model(self, config: Model, payload: Dict[str, Any]) -> None:
        if config.model_id is None:
            raise RuntimeError("Cannot update model without model id")

        with self._request(
            "PUT", f"/v2/models/{config.model_id}", json_body=payload, timeout=120
        ) as response:
            updated = json.loads(response.read().decode("utf-8"))
        self._apply_model_identity(config, updated)

    def _create_or_reuse_model(
        self,
        config: Model,
        payload: Dict[str, Any],
        *,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        model_name = payload["name"]
        existing = self._find_existing_model(model_name, payload)
        if existing:
            self._apply_model_identity(config, existing)
            logger.info(
                "Reusing existing model %s (id=%s) for run %s",
                config.model_name,
                config.model_id,
                config.name,
            )
            self._update_existing_model(config, payload)
            return

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                with self._request(
                    "POST", "/v2/models", json_body=payload, timeout=120
                ) as response:
                    created = json.loads(response.read().decode("utf-8"))
                self._apply_model_identity(config, created)
                logger.info(
                    "Created model %s (id=%s) for run %s",
                    config.model_name,
                    config.model_id,
                    config.name,
                )
                return
            except RuntimeError as exc:
                last_error = exc
                if "already exists" in str(exc).lower():
                    existing = self._find_existing_model(model_name, payload)
                    if existing:
                        self._apply_model_identity(config, existing)
                        logger.info(
                            "Detected existing model %s (id=%s) after create conflict",
                            config.model_name,
                            config.model_id,
                        )
                        self._update_existing_model(config, payload)
                        return
                raise
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_request_error(exc):
                    raise

                existing = self._find_existing_model(model_name, payload)
                if existing:
                    self._apply_model_identity(config, existing)
                    logger.info(
                        "Found existing model %s (id=%s) after create timeout/error",
                        config.model_name,
                        config.model_id,
                    )
                    self._update_existing_model(config, payload)
                    return

                if attempt == retries:
                    raise

                logger.warning(
                    "Create model request failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    model_name,
                    attempt,
                    retries,
                    exc,
                    retry_delay,
                )
                time.sleep(retry_delay)

        if last_error is not None:
            raise last_error

    def _benchmark_metrics_ready(self, payload: Dict[str, Any]) -> bool:
        return bool(
            payload.get("raw_metrics") is not None
            or payload.get("request_latency_mean") is not None
            or payload.get("tokens_per_second_mean") is not None
            or payload.get("requests_per_second_mean") is not None
        )

    def _wait_for_benchmark_result(
        self,
        benchmark_id: int,
        *,
        timeout: int = 180,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Wait for the benchmark detail endpoint to include synced metrics.

        The worker marks a benchmark as `completed` before it uploads parsed metrics
        back to `/v2/benchmarks/{id}/metrics`, so a detail fetch immediately after the
        completion event can still return empty metric fields.
        """
        deadline = time.time() + timeout
        last_payload: Dict[str, Any] = {}

        while time.time() < deadline:
            with self._request(
                "GET", f"/v2/benchmarks/{benchmark_id}", timeout=120
            ) as response:
                last_payload = json.loads(response.read().decode("utf-8"))

            if self._benchmark_metrics_ready(last_payload):
                return last_payload

            time.sleep(poll_interval)

        logger.warning(
            "Timed out waiting for benchmark %s metrics to sync; saving latest payload without raw metrics",
            benchmark_id,
        )
        return last_payload

    def setup_model(self, model: Model):
        """
        Create a GPUStack model deployment for one run definition.

        The YAML `backend_parameters`, `envs`, and backend/version fields are passed
        through almost directly to GPUStack's `/v2/models` API.
        """
        name = f"{self._repo_model_name()}-{model.name}"
        source = self.source
        payload = {
            "source": self.source,
            "huggingface_repo_id": self.model_repo_id,
            "huggingface_filename": None,
            "model_scope_model_id": None,
            "model_scope_file_path": None,
            "local_path": None,
            "description": None,
            "meta": {},
            "replicas": 1,
            "ready_replicas": 0,
            "categories": ["llm"],
            "placement_strategy": "spread",
            "cpu_offloading": None,
            "distributed_inference_across_workers": True,
            "worker_selector": {},
            "gpu_selector": None,
            "backend": model.backend.value,
            "backend_version": model.backend_version,
            "backend_parameters": model.backend_parameters or [],
            "image_name": None,
            "run_command": None,
            "env": model.envs or None,
            "restart_on_error": False,
            "distributable": False,
            "extended_kv_cache": {},
            "speculative_config": {},
            "generic_proxy": False,
            "cluster_id": self.cluster_id,
            "name": name,
            "enable_model_route": True,
        }

        if source == Source.Huggingface.value:
            payload["huggingface_repo_id"] = self.model_repo_id
        elif source == Source.ModelScope.value:
            payload["model_scope_model_id"] = self.model_repo_id

        self._create_or_reuse_model(model, payload)

    def monitor_model_startup(self, config: Model):
        """Poll the first model instance until it becomes `running` or fails."""
        health_check = config.health_check or HealthCheck()
        if health_check.init_delay > 0:
            logger.info(
                "Waiting %ss before polling model startup", health_check.init_delay
            )
            time.sleep(health_check.init_delay)

        deadline = time.time() + health_check.timeout
        last_state = None
        logged_waiting_instances = False
        poll_timeout = min(max(int(health_check.interval * 2), 10), 30)

        while time.time() < deadline:
            try:
                # Treat transient API/proxy timeouts as retryable during startup.
                instances = self._list_model_instances(
                    config.model_id, timeout=poll_timeout
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ) as exc:
                logger.warning(
                    "Polling model startup failed: %s; retrying in %ss",
                    exc,
                    health_check.interval,
                )
                time.sleep(health_check.interval)
                continue

            if not instances:
                if not logged_waiting_instances:
                    logger.info(
                        "Model %s (id=%s) has no instances yet; waiting...",
                        config.model_name,
                        config.model_id,
                    )
                    logged_waiting_instances = True
            else:
                logged_waiting_instances = False
                instance = instances[0]
                state = instance.get("state")
                if state != last_state:
                    logger.info(
                        "Model instance %s state: %s (%s)",
                        instance.get("name"),
                        state,
                        instance.get("state_message", ""),
                    )
                    last_state = state

                if state == "running":
                    config.instance_name = instance["name"]
                    return

                if state in {"error", "unreachable"}:
                    raise RuntimeError(
                        f"Model instance {instance.get('name')} failed: {instance.get('state_message')}"
                    )

            time.sleep(health_check.interval)

        raise TimeoutError(
            f"Timed out waiting for model {config.model_name} to become running"
        )

    def stop_model(self, config: Model):  # noqa: C901
        """Scale the model to zero replicas and wait until instances are gone."""
        if config.model_id is None:
            return

        payload: Optional[Dict[str, Any]] = None
        for attempt in range(1, 4):
            try:
                payload = self._get_model(config.model_id)
                break
            except Exception as exc:
                if not self._is_retryable_request_error(exc) or attempt == 3:
                    logger.error(
                        "Failed to fetch model %s (id=%s) before scale down: %s",
                        config.model_name,
                        config.model_id,
                        exc,
                    )
                    return
                logger.warning(
                    "Fetching model %s (id=%s) before scale down failed (attempt %s/3): %s; retrying",
                    config.model_name,
                    config.model_id,
                    attempt,
                    exc,
                )
                time.sleep(5)

        if payload is None:
            return

        payload["replicas"] = 0
        payload.pop("id", None)
        payload.pop("created_at", None)
        payload.pop("updated_at", None)
        payload.pop("ready_replicas", None)

        scale_down_sent = False
        for attempt in range(1, 4):
            try:
                with self._request(
                    "PUT",
                    f"/v2/models/{config.model_id}",
                    json_body=payload,
                    timeout=120,
                ):
                    pass
                scale_down_sent = True
                break
            except Exception as exc:
                if not self._is_retryable_request_error(exc) or attempt == 3:
                    logger.error(
                        "Failed to scale down model %s (id=%s): %s",
                        config.model_name,
                        config.model_id,
                        exc,
                    )
                    return
                logger.warning(
                    "Scale down request for model %s (id=%s) failed (attempt %s/3): %s; retrying",
                    config.model_name,
                    config.model_id,
                    attempt,
                    exc,
                )
                time.sleep(5)

        if not scale_down_sent:
            return

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                model = self._get_model(config.model_id)
                instances = self._list_model_instances(config.model_id, timeout=15)
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ) as exc:
                logger.warning("Polling model scale down failed: %s; retrying", exc)
                time.sleep(5)
                continue

            replicas = model.get("replicas")
            ready_replicas = model.get("ready_replicas")
            if replicas == 0 and not instances:
                logger.info(
                    "Stopped model %s (id=%s)", config.model_name, config.model_id
                )
                return

            logger.info(
                "Waiting for model %s to scale down: replicas=%s ready_replicas=%s instances=%s",
                config.model_name,
                replicas,
                ready_replicas,
                len(instances),
            )
            time.sleep(5)

        logger.warning(
            "Scale down request sent for model %s (id=%s), but instances still exist after timeout",
            config.model_name,
            config.model_id,
        )

    def warmup_service(self, config: Model):
        """
        Send a few small chat-completions requests before benchmarking.

        Warmup helps reduce noise from first-request effects such as lazy kernel
        initialization or cold tokenizer/model paths.
        """
        if not config.model_name or not config.warmup_num_requests:
            return

        payload = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 8,
            "stream": False,
        }
        warmup_errors = 0
        for _ in range(config.warmup_num_requests):
            try:
                with self._request(
                    "POST", "/v1/chat/completions", json_body=payload, timeout=120
                ):
                    pass
            except Exception as exc:
                warmup_errors += 1
                logger.warning("Warmup request failed: %s", exc)
                if warmup_errors >= 3:
                    raise
                time.sleep(2)
            else:
                time.sleep(0.2)

    def create_benchmark(
        self,
        config: Model,
        profile: Dict[str, Any],
        request_rate: Optional[int] = None,
    ):
        """
        Create one GPUStack benchmark job from one profile definition.

        `request_rate` can be overridden from the CLI. If not provided, the script
        uses the `request_rate` defined inside the selected profile YAML.
        """
        if not config.instance_name or config.model_id is None or not config.model_name:
            raise RuntimeError("Model is not ready for benchmark creation")

        effective_request_rate = (
            request_rate
            if request_rate is not None
            else profile.get("request_rate", 10)
        )
        profile_slug = self._slugify(profile["name"])
        ts = f"{self._timestamp()}"[-4:]
        benchmark_name = self._build_benchmark_name(
            config.model_name,
            profile_slug,
            effective_request_rate,
            ts,
        )
        payload = {
            "name": benchmark_name,
            "cluster_id": self.cluster_id,
            "model_name": config.model_name,
            "model_id": config.model_id,
            "model_instance_name": config.instance_name,
            "profile": profile["name"],
            "dataset_name": profile.get("dataset_name"),
            "dataset_input_tokens": profile.get("dataset_input_tokens"),
            "dataset_output_tokens": profile.get("dataset_output_tokens"),
            "dataset_seed": profile.get("dataset_seed"),
            "dataset_shared_prefix_tokens": profile.get("dataset_shared_prefix_tokens"),
            "request_rate": effective_request_rate,
            "total_requests": profile.get("total_requests"),
            "max_concurrency": profile.get("max_concurrency"),
        }
        with self._request(
            "POST", "/v2/benchmarks", json_body=payload, timeout=120
        ) as response:
            created = json.loads(response.read().decode("utf-8"))
        config.benchmark_id = created["id"]
        config.benchmark_name = created["name"]
        logger.info(
            "Created benchmark %s (id=%s) for test case %s",
            config.benchmark_name,
            config.benchmark_id,
            profile["name"],
        )

    def monitor_benchmark(
        self,
        config: Model,
        test_case: str,
        request_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Watch the benchmark SSE stream until completion and dump the final payload.

        The resulting JSON is the full `/v2/benchmarks/{id}` response, which makes
        it suitable for later offline analysis without re-querying GPUStack.
        """
        if config.benchmark_id is None:
            raise RuntimeError("Benchmark has not been created")
        if not config.benchmark_name:
            raise RuntimeError("Benchmark name is missing")

        result_path = self._result_path(config.benchmark_name)
        last_state = None
        watch_response = self._request(
            "GET",
            "/v2/benchmarks",
            params={"watch": "true"},
            timeout=3600,
        )

        with watch_response:
            for event in self._iter_sse_payloads(watch_response):
                payload = event.get("data", event)
                if not isinstance(payload, dict):
                    continue
                if payload.get("id") != config.benchmark_id:
                    continue

                state = payload.get("state")
                if state != last_state:
                    logger.info(
                        "Benchmark %s state: %s (%s)",
                        payload.get("name"),
                        state,
                        payload.get("state_message"),
                    )
                    last_state = state

                if state == "completed":
                    final_result = self._wait_for_benchmark_result(config.benchmark_id)
                    self._dump_result(result_path, final_result)
                    return final_result

                if state in {"error", "stopped", "unreachable"}:
                    raise RuntimeError(
                        f"Benchmark {payload.get('name')} failed: {payload.get('state_message')}"
                    )

        raise RuntimeError(
            f"Benchmark stream ended before completion: {config.benchmark_name}"
        )

    def parse_results(self, result_file: str) -> Dict[str, Any]:
        """Parse and extract metrics from benchmark result file"""
        result_path = Path(result_file)
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def run_engine_test(
        self,
        config: Model,
        profiles: Dict[str, Dict[str, Any]],
        output_dir: str,
    ):
        """
        Execute one run from the YAML end-to-end.

        One run may map to:
        - multiple test cases
        - multiple request rates per test case

        The execution order is:
        create model -> wait for ready -> warm up -> run benchmarks -> stop model
        """
        logger.info("Starting test for %s", config.name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.setup_model(config)
            self.monitor_model_startup(config)
            self.warmup_service(config)

            for test_case in config.test_cases:
                profile = profiles.get(test_case)
                if not profile:
                    logger.warning(
                        "Profile for test case '%s' not found, skipping", test_case
                    )
                    continue

                request_rates = config.request_rates or [None]
                for request_rate in request_rates:
                    self.create_benchmark(config, profile, request_rate)
                    self.monitor_benchmark(config, test_case, request_rate)
                    if request_rate is None:
                        logger.info("Completed test case: %s", test_case)
                    else:
                        logger.info(
                            "Completed test case: %s with request_rate=%s",
                            test_case,
                            request_rate,
                        )

        except Exception as e:
            logger.error("Error running test %s: %s", config.name, e)
            raise
        finally:
            if not config.stop_model_after_run:
                logger.info(
                    "Skipping stop model for %s (id=%s) because stop_model_after_run=false",
                    config.model_name,
                    config.model_id,
                )
            else:
                try:
                    self.stop_model(config)
                except Exception as exc:
                    logger.error(
                        "Unexpected error while stopping model %s (id=%s): %s",
                        config.model_name,
                        config.model_id,
                        exc,
                    )
                time.sleep(15)


def load_yaml(config_file: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profile(profile_file: str) -> Dict[str, Any]:
    """Load benchmark profile configuration"""
    data = load_yaml(profile_file)
    profile_dict = {}
    for profile in data.get("profiles", []):
        profile_dict[profile["name"]] = profile
    return profile_dict


def load_config(config_file: str) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file"""
    return load_yaml(config_file)


def create_engine_configs_from_config(
    config: Dict[str, Any],
    run_names: Optional[List[str]],
    test_cases: Optional[List[str]],
    request_rates: Optional[List[int]],
) -> List[Model]:
    """
    Materialize YAML `runs` into `Model` objects used by the executor.

    Precedence rules:
    1. `--run-names` filters which runs are created
    2. `--test-cases` overrides both run-level and top-level `test_cases`
    3. run-level `test_cases` override top-level `test_cases`
    4. `--request-rates` overrides the profile's `request_rate`
    """
    engine_configs = []

    health_check_config = config.get("health_check", {})
    default_health_check = HealthCheck(
        timeout=health_check_config.get("timeout", 30),
        interval=health_check_config.get("interval", 1.0),
        init_delay=health_check_config.get("init_delay", 60),
    )
    default_warmup_num_requests = config.get("warmup", {}).get("num_requests", 10)
    default_stop_model_after_run = config.get("stop_model_after_run", True)
    default_test_cases = [
        case["name"] if isinstance(case, dict) else str(case)
        for case in config.get("test_cases", [])
    ]

    run_name_filter = set(run_names or [])
    for run_config in config.get("runs", []):
        if run_name_filter and run_config["name"] not in run_name_filter:
            logger.info(
                "Skipping run %s as it's not in specified run names", run_config["name"]
            )
            continue

        if test_cases:
            selected_test_cases = test_cases
        elif "test_cases" in run_config:
            selected_test_cases = [
                case["name"] if isinstance(case, dict) else str(case)
                for case in run_config.get("test_cases", [])
            ]
        else:
            selected_test_cases = default_test_cases

        run_health_check_config = run_config.get("health_check", {})
        health_check = HealthCheck(
            timeout=run_health_check_config.get(
                "timeout", default_health_check.timeout
            ),
            interval=run_health_check_config.get(
                "interval", default_health_check.interval
            ),
            init_delay=run_health_check_config.get(
                "init_delay", default_health_check.init_delay
            ),
        )

        engine_config = Model(
            name=run_config["name"],
            test_cases=selected_test_cases,
            backend=EngineType(run_config["backend"]),
            backend_version=(
                str(run_config.get("backend_version"))
                if run_config.get("backend_version") is not None
                else None
            ),
            backend_parameters=run_config.get("backend_parameters", []),
            envs=run_config.get("envs", {}),
            args=run_config.get("args", []),
            health_check=health_check,
            warmup_num_requests=run_config.get(
                "warmup_num_requests", default_warmup_num_requests
            ),
            stop_model_after_run=run_config.get(
                "stop_model_after_run", default_stop_model_after_run
            ),
            request_rates=request_rates,
        )
        engine_configs.append(engine_config)

    return engine_configs


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Inference Engine Automated Performance Testing"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--profile", default="profile.yaml", help="Path to profile YAML file"
    )
    parser.add_argument("--model", help="Override the model repo id from config")
    parser.add_argument("--gpustack-url", required=True, help="GPUStack URL")
    parser.add_argument("--gpustack-token", required=True, help="GPUStack token")
    parser.add_argument(
        "--cluster-id",
        "--gpustack-cluster-id",
        dest="cluster_id",
        type=int,
        required=True,
        help="GPUStack cluster id",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--run-names",
        type=lambda s: [name.strip() for name in s.split(",") if name.strip()],
        default=[],
        help="Specific run names to execute, comma-separated",
    )
    parser.add_argument(
        "--test-cases",
        type=lambda s: [name.strip() for name in s.split(",") if name.strip()],
        default=[],
        help=(
            "Specific test case names to execute, comma-separated. "
            "Overrides run-level and config-level test_cases."
        ),
    )
    parser.add_argument(
        "--request-rates",
        type=lambda s: [int(rate.strip()) for rate in s.split(",") if rate.strip()],
        default=[],
        help="Override profile request_rate with one or more comma-separated values, e.g. 1,4,8,16",
    )
    parser.add_argument(
        "--stop-model-after-run",
        dest="stop_model_after_run",
        action="store_true",
        default=None,
        help="Stop the model after each run completes. Overrides config when provided.",
    )
    parser.add_argument(
        "--no-stop-model-after-run",
        dest="stop_model_after_run",
        action="store_false",
        help="Keep the model running after each run completes. Overrides config when provided.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    profile = load_profile(args.profile)
    model = args.model or config["model"]
    source = config["source"]
    output_dir = args.output_dir or config.get("output_dir", "benchmark_results")

    manager = EngineManager(
        model,
        source,
        args.gpustack_url,
        args.gpustack_token,
        args.cluster_id,
        output_dir,
    )

    engine_configs = create_engine_configs_from_config(
        config,
        args.run_names,
        args.test_cases or None,
        args.request_rates or None,
    )
    if not engine_configs:
        raise SystemExit("No matching runs found")

    if args.stop_model_after_run is not None:
        for engine_config in engine_configs:
            engine_config.stop_model_after_run = args.stop_model_after_run

    for engine_config in engine_configs:
        try:
            manager.run_engine_test(engine_config, profile, output_dir)
            logger.info("Successfully completed test: %s", engine_config.name)
        except Exception as e:
            logger.error("Failed to run test %s: %s", engine_config.name, e)

    logger.info("All tests completed")


class RedirectStdoutStderr:
    """Utility context manager for callers that want to redirect script output."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.target
        sys.stderr = self.target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


if __name__ == "__main__":
    main()
