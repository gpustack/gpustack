def metric_name(name: str) -> str:
    METRIC_PREFIX = "gpustack:"
    return f"{METRIC_PREFIX}{name}"
