"""Configurable environment variables for GPUStack."""

import logging
import os
from typing import List, Set

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
# Diagnostic: when non-empty, every executed SQL whose text contains this
# substring gets its Python call stack logged once per distinct call site
# (deduplicated), so a high-frequency query can be attributed to its caller
# without drowning the log. Empty (default) disables the check entirely.
DB_TRACE_SQL_SUBSTR = os.getenv("GPUSTACK_DB_TRACE_SQL_SUBSTR", "")
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))
# Bound how long a pooled connection may be reused so a node that a database
# failover demoted cannot be talked to forever. SQLAlchemy's default is -1,
# meaning a connection is never recycled; 1800 matches the prevailing default
# for pooled connection lifetime (HikariCP's maxLifetime). 0 disables it.
DB_POOL_RECYCLE = int(os.getenv("GPUSTACK_DB_POOL_RECYCLE", 1800))
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

# Anything that ends up *inside* a WasmPlugin CR is configured under
# ``gateway_plugin`` in config.yaml instead of here, so the two mechanisms keep
# distinct effective-time semantics: a value there takes effect on the next
# reconcile, a value here on restart. What remains below governs how the server
# maintains those CRs, and never appears in one.
#
# The two exceptions are the variables that predate that split. They are honored
# as the *default* the config file may override, because dropping them outright
# would fail silently: a deployment that had raised the ext-auth timeout would
# quietly get the stock one, and a deployment that had added a content type
# would quietly stop metering it -- an accounting error nobody notices until a
# bill is wrong.


_warned_deprecated_envs: Set[str] = set()


def _deprecated_gateway_env(name: str, replacement: str, default, parse):
    """Read a variable that has moved into ``gateway_plugin``.

    Warned about once per process rather than per read: this is called from a
    pydantic ``default_factory``, so it runs every time the plugin's config is
    rendered.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    if name not in _warned_deprecated_envs:
        _warned_deprecated_envs.add(name)
        logging.getLogger(__name__).warning(
            f"{name} is deprecated and will be removed. Set {replacement} in "
            "the config file instead; the config file wins where both are set."
        )
    try:
        return parse(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            f"Ignoring {name}: {raw!r} is not a valid value."
        )
        return default


def deprecated_ext_auth_timeout_ms() -> int:
    """Default for ``gateway_plugin.gpustack-ext-auth.config.authz.timeout``."""
    return _deprecated_gateway_env(
        "GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS",
        "gateway_plugin.gpustack-ext-auth.config.authz.timeout",
        30000,
        int,
    )


def deprecated_ai_statistics_content_types() -> List[str]:
    """Default for ``gateway_plugin.ai-statistics.config.enable_content_types``."""
    return _deprecated_gateway_env(
        "GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES",
        "gateway_plugin.ai-statistics.config.enable_content_types",
        ["application/json", "text/event-stream"],
        lambda raw: [part.strip() for part in raw.split(",") if part.strip()],
    )


# How often the gateway auth reconciler recomputes the key tables from the
# database in full. This is a security parameter, not a tuning knob: deletions
# that bypass the ORM (a principal cascade, direct SQL) produce no event, and
# on a PUBLIC route -- where the plugin no longer calls the server per request
# -- this interval is the worst-case time such a key keeps being accepted.
# Cheap to keep short: the reconciler diffs before writing, so an unchanged
# recomputation costs one read and no CR write.
GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS = int(
    os.getenv("GPUSTACK_GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS", 30)
)

# Whether a custom API key -- one whose secret the user supplied rather than
# this server generating it -- may be authenticated at the gateway.
#
# Off, a custom key keeps working exactly as before: the gateway cannot verify
# it, so every request carrying one asks the server. On, it is given the same
# fast digest a generated key gets, which is what lets the gateway answer
# locally and, on a public route, without the server at all.
#
# The cost is specific and worth stating. ``ApiKeyCreate.custom`` has no entropy
# requirement of any kind, so ``custom: "123456"`` is a key that can exist. What
# is published for it travels in a WasmPlugin CR -- reachable through
# ``istioctl proxy-config``, ``/config_dump`` and support bundles, a wider
# audience than the database the argon2 hash never leaves. Against a weak secret
# an offline search is then trivial, and no part of the system knows which
# custom keys are weak.
#
# Two hashes are published, and the weaker one is not the digest. The digest is
# salted, so it has to be attacked one key at a time. But a custom key has no
# access key inside it, so ``get_key_pair`` derives one by hashing the whole
# credential -- and that value is what the ``keys`` table is *indexed by*. It is
# an unsalted blake2b of the secret itself, identical for the same secret in
# every deployment, which is what makes precomputation pay: one table over a
# password list, tried against every custom key ever published anywhere. The
# salt on the digest buys nothing while the index next to it is salt-free.
#
# Deployments that mint their custom keys from a random source are unaffected;
# deployments that let people choose them should turn this off.
GATEWAY_AUTH_ALLOW_CUSTOM_KEYS = (
    os.getenv("GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", "true").lower() != "false"
)

# Byte budget for the parts of the ext-auth CR the reconciler owns: the key
# tables and one match rule per PUBLIC route. Both are sized from it, so
# whichever grows leaves less room for the other instead of the two overrunning
# a shared limit independently.
#
# The ceiling is etcd's 1.5 MiB (1536 KiB) object limit, and exceeding it is not
# a partial failure: the write is refused outright, the tables freeze at their
# last good state, and revocations stop propagating -- on a PUBLIC route that is
# the only path they have. So this leaves ~28% for the rest of the object
# (metadata, managedFields, the static config block) and for error in the
# per-entry estimates the reconciler sizes with.
#
# Overflow itself is benign on both sides: a key past the budget authenticates
# at the server on every request, and a public route past it authorizes there
# per request. Both are the behaviour that predates this mechanism.
GATEWAY_AUTH_MAX_CR_BYTES = int(
    os.getenv("GPUSTACK_GATEWAY_AUTH_MAX_CR_BYTES", 1_100_000)
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

# Opt-in (default off): drop a runner image's bundled cuda-compat and use the host
# driver so consumer GPUs can run images built for a newer CUDA minor (same major).
# Overridable per-model via the same env name in the model's env.
ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY_ENV = (
    "GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY"
)
ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY = os.getenv(
    ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY_ENV, "false"
).lower() in ["true", "1"]

# Host IPC namespace for inference containers. Unset (the default) derives
# it from the instance: only a deployment attached to a shared cache
# service runs host-IPC, which its CUDA-IPC zero-copy transfer path
# requires. Setting the env (globally here, or per-model via the same name
# in the model's env) overrides the derivation either way — e.g. "false"
# on PodSecurity-enforcing Kubernetes clusters, where hostIPC pods are
# rejected. Docker ignores shm_size under host IPC.
HOST_IPC_ENV = "GPUSTACK_HOST_IPC"
HOST_IPC = os.getenv(HOST_IPC_ENV)

# GPU instance configuration
# Interval at which the controller re-observes a still-transitioning (non-
# settled) GPU instance via an in-memory requeue, instead of writing its own
# status back to the DB to self-trigger the next poll. Ready-row drift is
# picked up by the downstream watch, so only transitioning rows re-observe on
# this cadence. The PV / PVT finalize controllers reuse this cadence to re-probe
# a still-finalizing row, which is just another transitioning row. Clamped to
# >= 1s at use to avoid a busy loop.
GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL = int(
    os.getenv("GPUSTACK_GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL", 15)
)  # in seconds
# Optional low-frequency fallback sweep: with the Ready-row reconfirm chain
# retired, a settled Ready row's worker-side drift flows back only via the
# downstream watch. If the watch misses an event across a reconnect gap, this
# opt-in sweep periodically re-observes Ready rows so the drift is eventually
# reconciled. 0 (default) disables it; set a low frequency (seconds) only if a
# watch-gap coverage hole is observed.
GPU_INSTANCE_READY_SWEEP_INTERVAL = int(
    os.getenv("GPUSTACK_GPU_INSTANCE_READY_SWEEP_INTERVAL", 0)
)  # in seconds
# How long a not-yet-Ready GPU instance may stay Unknown because its worker-side
# CR is unreadable, before the controller settles it to Stopped.
#
# An unreadable CR is normally eventual consistency (it is about to appear), so
# the row is held at Unknown rather than prematurely stopped and stranded. But
# Unknown is a METERED phase, so an unbounded hold means an instance that no
# longer exists anywhere keeps accruing uptime with no upper limit — which is
# exactly what a cluster teardown / uninstall-reinstall upgrade produces in bulk.
# Bounding the hold keeps the eventual-consistency tolerance while capping the
# billing exposure. Generous by default (30 min) because the cost of settling too
# early is a spuriously Stopped instance the user must restart. 0 disables the
# bound (restores the previous unbounded behaviour).
GPU_INSTANCE_UNREADABLE_CR_TOLERANCE = int(
    os.getenv("GPUSTACK_GPU_INSTANCE_UNREADABLE_CR_TOLERANCE", 1800)
)  # in seconds

# Model instance configuration
MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
MODEL_INSTANCE_HEALTH_CHECK_INTERVAL = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL", 3)
)
# Period, in seconds, for forcing an authoritative (uncached) DB reconciliation
# of locally-tracked model instances in the worker state sync. 0 disables it,
# leaving the sync purely cache-backed. It exists only as a backstop for a
# watch cache that silently diverged from DB truth without a reconnect (e.g. a
# coordinator dropping a DELETED on a live stream); enabling it reintroduces one
# uncached full SELECT per worker per period, so keep it well above the health
# check interval when set.
MODEL_INSTANCE_STATE_RECONCILE_INTERVAL = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_STATE_RECONCILE_INTERVAL", 0)
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

# Platform-wide timezone for calendar-based rollups and time-of-day display.
# It buckets the ``model_usages.date`` daily rollup (and the matching
# ``model_usage_details.date`` audit column), the ``metered_usage`` GPU/storage
# time buckets, and renders Last Active / resource-event times — and is the
# canonical calendar for any other feature that needs an operator-chosen
# timezone. Empty (default) ⇒ use the operating system's local timezone
# (resolved from ``TZ`` env var / ``/etc/localtime``). Set to an IANA name
# (``Asia/Shanghai``, ``UTC``, ...) to override — useful when the server
# container runs in UTC but operators expect a different region's calendar.
#
# ``GPUSTACK_USAGE_ROLLUP_TIMEZONE`` is the pre-rename name, kept as a
# deprecated alias: ``GPUSTACK_TIMEZONE`` wins when both are set; otherwise the
# legacy value is honored. ``USING_DEPRECATED_TIMEZONE`` records that the value
# came from the legacy var so ``resolve_rollup_tz`` can emit a one-time
# deprecation warning — deferred out of import time, since this module loads
# before logging is configured (logging here is a Python anti-pattern).
_timezone = os.getenv("GPUSTACK_TIMEZONE", "")
_legacy_rollup_timezone = os.getenv("GPUSTACK_USAGE_ROLLUP_TIMEZONE", "")
USING_DEPRECATED_TIMEZONE = bool(_legacy_rollup_timezone and not _timezone)
TIMEZONE = _timezone or _legacy_rollup_timezone

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

# Usage export configuration
#
# Separate from the breakdown ceiling above because the two fail differently:
# ``/breakdown`` materializes every bucket into one JSON body, while
# ``/breakdown/export`` streams off a server-side cursor, so this process's
# memory is flat. That is not a free lunch — the database still computes and
# sorts the whole GROUP BY before the first row, and the export holds a
# connection throughout — so this ceiling is set by query cost, not by ours.
# 100k stays inside what is survivable without the safeguards a much larger
# one would need (statement timeout, per-user export concurrency, a read
# replica). Raise it once those exist.
USAGE_EXPORT_MAX_ROWS = int(os.getenv("GPUSTACK_USAGE_EXPORT_MAX_ROWS", 100000))
# Above this the UI warns that the export will take a while (it still runs).
# Tunable alongside the hard limit, so a deployment that raises one doesn't
# keep warning at 10k about exports it happily does at 200k.
USAGE_EXPORT_SOFT_ROWS = int(os.getenv("GPUSTACK_USAGE_EXPORT_SOFT_ROWS", 10000))

# The rest are internal guards, deliberately NOT environment-tunable. They
# bound what a hand-crafted request can make the server do; no deployment has
# a reason to move them, and exposing them as GPUSTACK_* would advertise knobs
# that only invite mis-tuning.

# Max logical tables one export request may ask for. Each is an independent
# aggregate query, so this gates fan-out. The UI can only ever ask for the
# tabs it renders — at most 4 today (3 built-in + the enterprise Organization
# breakdown) — so this is headroom, not a limit anyone should reach.
USAGE_EXPORT_MAX_SHEETS = 10
# Rows pulled off the server-side cursor per batch. Bounds peak memory and
# nothing else — enrichment is memoized per entity for the whole export, so it
# does not scale with this. Raising it buys very little: a 105k-row export
# spends about 2s in total waiting on its 106 fetches, so eliminating them all
# would not be felt.
USAGE_EXPORT_STREAM_CHUNK_ROWS = 1000
# Most files one split export may write, summed over its sheets — the real
# ceiling on the whole feature at 10 x USAGE_EXPORT_MAX_ROWS, ~1M rows in one
# download. Sized by what a browser download can finish rather than by what the
# server can produce: ~83MB of CSV, a minute or two on the wire, and no resume
# — a proxy idle timeout or a closed laptop costs the whole thing. Past this
# the estimate offers only "shorten the range", because anyone who needs
# millions of rows wants the API in a loop over date ranges.
USAGE_EXPORT_MAX_SPLIT_MEMBERS = 10
# xlsx cannot hold more than this per worksheet — a format limit, not ours.
XLSX_MAX_ROWS_PER_SHEET = 1048576

# Scheduled scaling (tidal) reconcile cadence. The loop is level-triggered — it
# recomputes the count each pass from (now, windows, baseline) — so this bounds
# only how long a window boundary can go unnoticed, never correctness. Cron
# resolution is a minute and the downstream scale-up (instance create, schedule,
# weight load) takes minutes anyway, so polling faster mostly re-reads unchanged
# rows. Clamped to >= 1s to avoid a busy loop.
SCALING_SCHEDULER_INTERVAL = max(
    1, int(os.getenv("GPUSTACK_SCALING_SCHEDULER_INTERVAL", 30))
)  # in seconds
