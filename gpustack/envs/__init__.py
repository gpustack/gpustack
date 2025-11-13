"""Configurable environment variables for GPUStack."""

import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 5))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 10))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))

# Proxy configuration
PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))

# JWT Expiration
JWT_TOKEN_EXPIRE_MINUTES = int(os.getenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", 120))

# Higress plugin configuration
HIGRESS_EXT_AUTH_TIMEOUT_MS = int(
    os.getenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", 3000)
)

# Worker configuration
WORKER_HEARTBEAT_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD", 150)
)  # 2.5 minutes in seconds
WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD", 300)
)  # 5 minutes in seconds

# Model instance configuration
MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
