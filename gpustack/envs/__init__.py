"""Configurable environment variables for GPUStack."""

import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))
# Backstop against leaked/long-held sessions accumulating as Postgres
# "idle in transaction" connections and exhausting the pool (#5678). Only
# fires while a transaction is open and idle -- an actively-running query,
# however long, is never affected. 0 disables it. Ignored for non-Postgres
# backends.
DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_SECONDS = int(
    os.getenv("GPUSTACK_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_SECONDS", 8 * 3600)
)

# Proxy configuration
PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))
PROXY_UPSTREAM_IDLE_TIMEOUT = int(
    os.getenv("GPUSTACK_PROXY_UPSTREAM_IDLE_TIMEOUT_SECONDS", 3)
)

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))

# JWT Expiration
JWT_TOKEN_EXPIRE_MINUTES = int(os.getenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", 120))

# Higress plugin configuration
HIGRESS_EXT_AUTH_TIMEOUT_MS = int(
    os.getenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", 30000)
)

# Server Cache
SERVER_CACHE_TTL_SECONDS = int(os.getenv("GPUSTACK_SERVER_CACHE_TTL_SECONDS", 600))
SERVER_CACHE_LOCKS_MAX_SIZE = int(
    os.getenv("GPUSTACK_SERVER_CACHE_LOCKS_MAX_SIZE", 10000)
)

# Server event bus queue capacity. Configurable via env so large clusters can tune the buffer.
EVENT_BUS_SUBSCRIBER_QUEUE_SIZE = int(
    os.getenv("GPUSTACK_EVENT_BUS_SUBSCRIBER_QUEUE_SIZE", 1024)
)

# Worker configuration
WORKER_HEARTBEAT_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_INTERVAL", 30)
)  # in seconds
WORKER_STATUS_SYNC_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_STATUS_SYNC_INTERVAL", 30)
)  # in seconds
WORKER_HEARTBEAT_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD", 150)
)  # 2.5 minutes in seconds
WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
# Worker unreachable check mode: auto, enabled, disabled
# - auto: automatically disable check when worker count > 50 (default)
# - enabled: always perform unreachable check
# - disabled: never perform unreachable check
WORKER_UNREACHABLE_CHECK_MODE = os.getenv(
    "GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE", "auto"
).lower()

# GPU instance configuration
# Interval at which the server re-confirms the worker-side status of Ready
# GPU instances. The reconciler is event-driven and stops touching a row once
# it is fully Ready, so without this periodic sweep a worker-side change after
# Ready would never be synced back. A value <= 0 disables the sweep.
GPU_INSTANCE_RECONFIRM_INTERVAL = int(
    os.getenv("GPUSTACK_GPU_INSTANCE_RECONFIRM_INTERVAL", 60)
)  # in seconds

# Model instance configuration
MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
MODEL_INSTANCE_HEALTH_CHECK_INTERVAL = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL", 3)
)
DISABLE_OS_FILELOCK = os.getenv("GPUSTACK_DISABLE_OS_FILELOCK", "false").lower() in [
    "true",
    "1",
]

# Opt out of automatically writing gpustack's configured port ranges to
# /proc/sys/net/ipv4/ip_local_reserved_ports. Use when the environment already
# manages the reservation, or when the configured ranges would starve the
# ephemeral pool after reservation.
SKIP_RESERVE_EPHEMERAL_PORTS = os.getenv(
    "GPUSTACK_SKIP_RESERVE_EPHEMERAL_PORTS", "false"
).lower() in ["true", "1"]
# Add debug logs for slow worker status collection, default to 3 minutes
WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS = float(
    os.getenv("GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS", 180)
)

# Model evaluation cache configuration
MODEL_EVALUATION_CACHE_MAX_SIZE = int(
    os.getenv("GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE", 1000)
)
MODEL_EVALUATION_CACHE_TTL = int(os.getenv("GPUSTACK_MODEL_EVALUATION_CACHE_TTL", 3600))

# Scheduler configuration (server-side)
SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE", 100)
)
SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE", 5)
)
# Scale-down scoring weights (relative, normalized in score chain)
SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE", 100)
)
SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE", 10)
)
SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE", 1)
)

MIGRATION_DATA_DIR = os.getenv("GPUSTACK_MIGRATION_DATA_DIR", None)

DATA_MIGRATION = os.getenv("GPUSTACK_DATA_MIGRATION", "false").lower() == "true"

GATEWAY_PORT_CHECK_INTERVAL = int(
    os.getenv("GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL", 2)
)  # in seconds

GATEWAY_PORT_CHECK_RETRY_COUNT = int(
    os.getenv("GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT", 300)
)  # number of retries

GATEWAY_MIRROR_INGRESS_NAME = os.getenv(
    "GPUSTACK_GATEWAY_MIRROR_INGRESS_NAME", "gpustack"
)

GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES = [
    ct.strip()
    for ct in os.getenv(
        "GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES",
        "application/json,text/event-stream",
    ).split(",")
    if ct.strip()
]

# Heuristics for partial-stream usage estimation.
# Used by metrics_collector when a gateway report arrives with completed=false
# (client disconnect, upstream cancel) and token fields are blank or partial.
# Defaults target English-leaning GPT-style tokenizers; tune for CJK or other
# tokenizer families as needed.
# Clamped to >= 1 so an operator typo (e.g. ``=0``) can't make
# ``_estimate_partial_usage`` divide by zero on every incomplete report.
USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN = max(
    1, int(os.getenv("GPUSTACK_USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN", 4))
)
USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK = max(
    1, int(os.getenv("GPUSTACK_USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK", 1))
)

# Timezone used to bucket the ``model_usages.date`` daily rollup (and the
# matching ``model_usage_details.date`` audit column). Empty (default) ⇒ use
# the operating system's local timezone (resolved from ``TZ`` env var /
# ``/etc/localtime``). Set to an IANA name (``Asia/Shanghai``, ``UTC``, ...)
# to override — useful when the server container runs in UTC but operators
# expect rollups to follow a different region's calendar day.
USAGE_ROLLUP_TIMEZONE = os.getenv("GPUSTACK_USAGE_ROLLUP_TIMEZONE", "")

# Usage details archival.
# Rows in ``model_usage_details`` older than the retention threshold (anchored
# on COALESCE(completed_at, created_at)) are moved to
# ``model_usage_details_archive`` by a leader-only background controller.
# The controller runs once on server startup and then on the configured cron
# schedule (UTC). Default ``0 3 * * *`` = daily at 03:00 UTC — picked to land
# in a typical off-peak window for most regions.
USAGE_DETAILS_RETENTION_MONTHS = int(
    os.getenv("GPUSTACK_USAGE_DETAILS_RETENTION_MONTHS", 13)
)
USAGE_DETAILS_ARCHIVE_CRON = os.getenv(
    "GPUSTACK_USAGE_DETAILS_ARCHIVE_CRON", "0 3 * * *"
)
# Per-batch row count for archival moves. Smaller batches keep transactions
# short on environments with replication lag concerns; larger batches reduce
# round-trip overhead.
USAGE_DETAILS_ARCHIVE_BATCH_SIZE = int(
    os.getenv("GPUSTACK_USAGE_DETAILS_ARCHIVE_BATCH_SIZE", 1000)
)

# Hard cap on the in-memory ``gateway_details_buffer`` (per-request audit
# rows held between flushes). Bounds memory growth when flushes fail
# persistently (DB down, schema drift) and the failure-path re-buffer keeps
# piling up alongside new ingest. Oldest entries are dropped on overflow
# with a warning log; the rollup buffer is naturally bounded by key
# cardinality so it does not need a separate cap.
USAGE_DETAILS_BUFFER_MAX_SIZE = int(
    os.getenv("GPUSTACK_USAGE_DETAILS_BUFFER_MAX_SIZE", 100000)
)

# ``resource_events`` hot/cold archival — same shape as the model_usage_details
# pair above. The events table grows much slower (lifecycle events, not per
# request), so the defaults are conservative.
USAGE_EVENTS_RETENTION_MONTHS = int(
    os.getenv("GPUSTACK_USAGE_EVENTS_RETENTION_MONTHS", 13)
)
USAGE_EVENTS_ARCHIVE_CRON = os.getenv(
    "GPUSTACK_USAGE_EVENTS_ARCHIVE_CRON", "30 3 * * *"
)
USAGE_EVENTS_ARCHIVE_BATCH_SIZE = int(
    os.getenv("GPUSTACK_USAGE_EVENTS_ARCHIVE_BATCH_SIZE", 5000)
)

# ``metered_usage`` hot/cold archival — hourly rollup rows older than the
# retention window move to ``metered_usage_archive``. Retention must stay far
# larger than the collector's settlement horizon (hours) so a still-being-
# written bucket is never archived; 13 months is safe by orders of magnitude.
METERED_USAGE_RETENTION_MONTHS = int(
    os.getenv("GPUSTACK_METERED_USAGE_RETENTION_MONTHS", 13)
)
METERED_USAGE_ARCHIVE_CRON = os.getenv(
    "GPUSTACK_METERED_USAGE_ARCHIVE_CRON", "0 4 * * *"
)
METERED_USAGE_ARCHIVE_BATCH_SIZE = int(
    os.getenv("GPUSTACK_METERED_USAGE_ARCHIVE_BATCH_SIZE", 5000)
)

# metered_usage collector tick — periodic safety net that flushes accumulated
# seconds for long-running metered resources that haven't had a lifecycle
# event since the last tick.
RESOURCE_USAGE_TICK_SECONDS = int(
    os.getenv("GPUSTACK_RESOURCE_USAGE_TICK_SECONDS", 300)
)
STORAGE_USAGE_TICK_SECONDS = int(os.getenv("GPUSTACK_STORAGE_USAGE_TICK_SECONDS", 300))

# Grace window before an elapsed hour-bucket is sealed (finalized for billing).
# A bucket is sealed once now >= bucket_end + grace, so this absorbs late events
# / clock skew; keep it comfortably larger than the tick interval. After sealing
# a bucket is immutable — late segments for it are dropped (and logged).
METERED_USAGE_SEAL_GRACE_SECONDS = int(
    os.getenv("GPUSTACK_METERED_USAGE_SEAL_GRACE_SECONDS", 900)
)

DEFAULT_CLUSTER_KUBERNETES = (
    os.getenv("GPUSTACK_DEFAULT_CLUSTER_KUBERNETES", "false").lower() == "true"
)

# Benchmark configuration
BENCHMARK_DATASET_SHAREGPT_PATH = os.getenv(
    "GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH",
    "/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json",
)
BENCHMARK_REQUEST_TIMEOUT = int(
    os.getenv("GPUSTACK_BENCHMARK_REQUEST_TIMEOUT", 3600)  # 1 hour
)  # in seconds

# Usage breakdown configuration
# Upper bound on the number of buckets a single no-pagination (page=-1)
# breakdown request may return — the trend charts and exports fetch the whole
# series unpaginated. A request whose grouping × date range would exceed this
# is rejected (HTTP 400) rather than silently truncated, so the caller narrows
# the range or adds filters. Tune up for very wide dashboards, or down to cap
# memory/payload more aggressively.
USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS = int(
    os.getenv("GPUSTACK_USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS", 50000)
)
