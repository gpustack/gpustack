"""Backend-aware accounting for offloaded PLE and Engram tables."""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from packaging.version import InvalidVersion, Version

from gpustack.schemas.models import BackendEnum, Model, get_backend
from gpustack.policies.utils import (
    get_ram_claim_from_model_env,
    get_vram_claim_from_model_env,
)
from gpustack.utils.command import (
    find_last_parameter,
    flatten_to_argv,
)

logger = logging.getLogger(__name__)
LookupTableKind = Literal["ple", "engram"]

_LOOKUP_ARCHITECTURES = {
    "DeepseekV41ForCausalLM": "engram",
    "Qwen4ExpForCausalLM": "ple",
    "Qwen4ExpForConditionalGeneration": "ple",
}


# Offload defaults are verified against these upstream release families.
_VERIFIED_VLLM_LOOKUP_VERSIONS = {(0, 30), (0, 31), (0, 32)}


def _backend_version(model: Model) -> Optional[Version]:
    """Resolve the version whose lookup defaults are used for estimation.

    Auto without a custom image uses the newest verified release family. This
    is an estimation assumption, not a constraint on the worker's runner choice.
    Explicit invalid versions and custom images without a version return None.
    """
    if not model.backend_version and not model.image_name:
        return Version(".".join(map(str, max(_VERIFIED_VLLM_LOOKUP_VERSIONS))))
    try:
        return Version(model.backend_version) if model.backend_version else None
    except InvalidVersion:
        return None


def _verified_lookup_version(model: Model) -> bool:
    """Check whether verified offload defaults apply.

    Custom images, prereleases, and unlisted release families are excluded
    because their lookup table behavior may differ from the verified releases.
    """
    version = _backend_version(model)
    return bool(
        not model.image_name
        and version
        and not version.is_prerelease
        and not version.is_devrelease
        and version.release[:2] in _VERIFIED_VLLM_LOOKUP_VERSIONS
    )


def get_offloaded_lookup_table_kind(
    model: Model, architectures: Optional[List[str]], config: Any = None
) -> Optional[LookupTableKind]:
    """Resolve supported backend offload settings without guessing custom defaults."""
    backend = get_backend(model)
    if backend not in (BackendEnum.VLLM, BackendEnum.SGLANG):
        return None
    kind = next(
        (
            _LOOKUP_ARCHITECTURES[a]
            for a in architectures or []
            if a in _LOOKUP_ARCHITECTURES
        ),
        None,
    )
    if kind is None:
        return None
    try:
        enabled = (
            _sglang_cpu_offload_enabled(model, kind, config)
            if backend == BackendEnum.SGLANG
            else _vllm_cpu_offload_enabled(model)
        )
    except (ValueError, TypeError):
        logger.warning(
            "Cannot interpret lookup table offload settings for %s", model.name
        )
        return None
    return kind if enabled else None


def _sglang_cpu_offload_enabled(
    model: Model, kind: LookupTableKind, config: Any
) -> bool:
    """Resolve SGLang offload settings for resource estimation.

    PLE follows the CUDA default of offloading when the runtime dtype is BF16,
    with explicit flags taking precedence. Platform-specific defaults are not
    modeled; deployments with different behavior can override resource claims.
    Engram requires explicit CPU placement.
    """
    parameters = model.backend_parameters
    if kind == "ple":
        storage = find_last_parameter(parameters, ["ple-offload-backend"])
        if storage not in (None, "pinned", "file"):
            raise ValueError("Unsupported PLE storage")
        enabled = None
        # BooleanOptionalAction uses the last positive or negative flag.
        for arg in flatten_to_argv(parameters):
            if arg == "--ple-offload-embedding":
                enabled = True
            elif arg == "--no-ple-offload-embedding":
                enabled = False
        if enabled is not None:
            return enabled
        return _sglang_ple_dtype(model, config) == "bfloat16"
    value = find_last_parameter(parameters, ["engram-config"])
    config = json.loads(value) if value is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Invalid Engram configuration")
    return config.get("placement") == "cpu" and config.get("lookup") != "peer_read"


def _sglang_ple_dtype(model: Model, config: Any) -> str:
    """Resolve the runtime dtype used by SGLang's PLE offload default.

    Explicit dtype overrides model metadata. Auto preserves BF16/FP16 and
    downcasts FP32 (also the missing-metadata default) to FP16 for Qwen.
    Weight quantization and the PLE embedding's storage dtype are independent
    of this runtime dtype.
    """
    dtype = find_last_parameter(model.backend_parameters, ["dtype"])
    auto = dtype is None or dtype.lower() == "auto"
    if auto:
        text_config = _get(config, "text_config") or config
        dtype = (
            _get(text_config, "dtype")
            or _get(text_config, "torch_dtype")
            or _get(config, "dtype")
            or _get(config, "torch_dtype")
            or "float32"
        )
    dtype = str(dtype).removeprefix("torch.").lower()
    dtype = {"half": "float16", "float": "float32"}.get(dtype, dtype)
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported SGLang runtime dtype")
    return "float16" if auto and dtype == "float32" else dtype


def _engram_config(model: Model) -> Dict[str, Any]:
    """Parse Engram settings used by the lookup table estimator.

    Hyphenated and underscored aliases share last-value precedence. Any dotted
    Engram arguments form a separate object that replaces the entire JSON
    object, regardless of argument order. Only cpu_offload is needed here and
    must be a boolean. Malformed configuration raises ValueError or TypeError.
    """
    parameters = model.backend_parameters
    # vLLM accepts both spellings; resolve aliases together to preserve ordering.
    value = find_last_parameter(parameters, ["engram-config", "engram_config"])
    config = json.loads(value) if value is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Invalid Engram configuration")
    # vLLM appends the object built from all dotted fields after regular JSON
    # arguments. Unrelated dotted fields therefore also discard cpu_offload.
    # https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/argparse_utils.py
    if any(
        arg.split("=", 1)[0].replace("_", "-").startswith("--engram-config.")
        for arg in flatten_to_argv(parameters)
    ):
        config = {}
    dotted = find_last_parameter(
        parameters, ["engram-config.cpu_offload", "engram_config.cpu_offload"]
    )
    if dotted is not None:
        config["cpu_offload"] = json.loads(dotted.lower())
    if "cpu_offload" in config and type(config["cpu_offload"]) is not bool:
        raise ValueError("Invalid cpu_offload setting")
    return config


def _vllm_cpu_offload_enabled(model: Model) -> bool:
    """Resolve whether the estimator should account for CPU-offloaded tables.

    Explicit configuration takes precedence over applicable legacy environment
    settings, then verified version defaults. Unverified defaults return False
    so the estimator does not deduct weights based on assumed backend behavior.
    """
    config = _engram_config(model)
    if "cpu_offload" in config:
        return config["cpu_offload"]
    version = _backend_version(model)
    env = model.env or {}
    # v0.30 reads this environment variable; v0.31 and v0.32 default to True
    # directly in EngramConfig. Explicit configuration takes precedence in both.
    # https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/engram.py
    if "VLLM_PLE_CPU_OFFLOAD" in env and (
        version is None or version.release[:2] < (0, 31)
    ):
        return bool(int(env["VLLM_PLE_CPU_OFFLOAD"]))
    if _verified_lookup_version(model):
        return True
    logger.debug(
        "Lookup offload defaults are unverified for %s (version=%s, image=%s); "
        "use explicit Engram configuration or resource claims",
        model.name,
        model.backend_version,
        model.image_name,
    )
    return False


def estimate_offloaded_lookup_bytes(model: Model, config: Any) -> int:
    """Estimate table bytes from already-loaded architecture and precision metadata.

    Unsupported formats or incomplete dimensions retain the full-weight VRAM
    estimate. No weight files or additional model metadata are requested.

    For vLLM, the estimate assumes the common CUDA/ROCm behavior in v0.32.0;
    it does not model older platform-specific storage paths. For example,
    v0.30 ROCm can keep PLE tables on the GPU, so this deduction may understate
    VRAM needs and reserve table RAM unnecessarily. For such deployments, set
    both GPUSTACK_MODEL_VRAM_CLAIM and GPUSTACK_MODEL_RAM_CLAIM. The RAM override
    is the total per-worker requirement, including any local KV cache; use zero
    only when no host RAM reservation is needed.

    Args:
        model: Deployment settings, including offload and resource overrides.
        config: Loaded model configuration as a dictionary or attribute object.

    Returns:
        Bytes for one full, unreplicated set of lookup tables.
        Zero when offload cannot be established, metadata is unsupported, or
        explicit RAM and VRAM claims make estimation unnecessary.
    """
    if (
        get_vram_claim_from_model_env(model) is not None
        and get_ram_claim_from_model_env(model) is not None
    ):
        return 0
    kind = get_offloaded_lookup_table_kind(model, _get(config, "architectures"), config)
    if kind is None:
        return 0
    try:
        text_config = _get(config, "text_config") or config
        quantization = _get(text_config, "quantization_config")
        if quantization is None:
            quantization = _get(config, "quantization_config") or {}
        if not isinstance(quantization, dict):
            raise ValueError("Unsupported quantization metadata")
        _validate_overrides(model, quantization)
        _validate_custom_table_format(model, config, text_config, quantization)
        if kind == "engram":
            if (
                get_backend(model) == BackendEnum.SGLANG
                and quantization.get("engram_embed_quant", "fp8") != "fp8"
            ):
                raise ValueError("Unsupported SGLang Engram precision")
            return _estimate_engram(text_config, quantization)
        return _estimate_ple(model, config, text_config, quantization)
    except (ValueError, TypeError, KeyError):
        logger.warning(
            "Cannot estimate offloaded lookup tables from the configuration for %s; "
            "using the full weight size for VRAM; table RAM is not estimated. "
            "Set GPUSTACK_MODEL_VRAM_CLAIM "
            "and GPUSTACK_MODEL_RAM_CLAIM to override the estimate.",
            model.name,
        )
        return 0


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Missing or invalid lookup table dimension")
    return value


def _layer_ids(config: Any, key: str) -> List[int]:
    ids = _get(config, key)
    if not isinstance(ids, (list, tuple)) or any(
        type(layer) is not int or layer < 0 for layer in ids
    ):
        raise ValueError("Missing or invalid lookup table layers")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate lookup table layers")
    return list(ids)


def _validate_overrides(model: Model, quantization: Dict) -> None:
    quant = find_last_parameter(model.backend_parameters, ["quantization", "q"])
    if quant is not None and quant != quantization.get("quant_method"):
        raise ValueError("Runtime quantization differs from checkpoint metadata")
    # Normalize option names only; underscores inside JSON values are metadata.
    names = (
        arg.split("=", 1)[0].replace("_", "-")
        for arg in flatten_to_argv(model.backend_parameters)
        if arg.startswith("--")
    )
    if any(
        name in ("--hf-overrides", "--json-model-override-args")
        or name.startswith("--hf-overrides.")
        for name in names
    ):
        raise ValueError("Architecture overrides require an explicit resource claim")


def _estimate_engram(config: Any, quantization: Dict) -> int:
    layers = _layer_ids(config, "engram_layer_ids")
    if not layers:
        return 0
    rows = _get(config, "engram_num_embeddings")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(layers):
        raise ValueError("Engram table sizes must match the configured layers")
    dim = _positive_int(_get(config, "engram_head_dim"))
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("scale_fmt") != "ue8m0"
        or quantization.get("weight_block_size") != [32, 32]
        or dim % 32
        or quantization.get("modules_to_not_convert")
        or quantization.get("ignored_layers")
    ):
        raise ValueError("Unsupported Engram table precision")
    row_bytes = _engram_row_bytes(quantization, layers, dim)
    return sum(_positive_int(row) for row in rows) * row_bytes


def _validate_custom_table_format(
    model: Model, root: Any, config: Any, quantization: Dict
) -> None:
    """Reject known custom table layouts that need explicit resource claims.

    Community NVFP4/INT4 tables and external PLE tables can retain the backbone's
    FP8 metadata. Their custom loaders do not follow the native table layout.
    """
    fields = ("engram_quant", "engram_quant_block_size", "engram_quant_global_scale")
    if any(
        _get(source, field) is not None
        for source in (root, config, quantization)
        for field in fields
    ):
        raise ValueError("Unsupported custom Engram table format")
    if any(
        name == "DSV41_ENGRAM_DTYPE" or name.startswith("VLLM_PLE_QUANT_")
        for name in model.env or {}
    ):
        raise ValueError("Custom table loading requires an explicit resource claim")


def _engram_row_bytes(quantization: Dict, layers: List[int], dim: int) -> int:
    """Count packed table values and UE8M0 scales for supported Engram formats.

    This describes storage in a compatible engine, not engine availability.
    Partial layer conversions and alternative scale layouts are unsupported.
    """
    table_format = quantization.get("engram_embed_quant", "fp8")
    if table_format not in ("fp8", "mxfp4"):
        raise ValueError("Unsupported Engram embedding quantization")
    if (
        quantization.get("engram_embed_block_size", 32) != 32
        or quantization.get("engram_embed_scale_dtype", "ue8m0") != "ue8m0"
    ):
        raise ValueError("Unsupported Engram embedding scales")
    if "engram_embed_layers" in quantization and set(
        _layer_ids(quantization, "engram_embed_layers")
    ) != set(layers):
        raise ValueError("Unsupported partial Engram table quantization")
    # MXFP4 packs two values per byte; both formats retain one scale byte per 32
    # values. Expert quantization does not determine the embedding table format.
    # https://github.com/vllm-project/vllm/pull/61530
    return (dim // 2 if table_format == "mxfp4" else dim) + dim // 32


def _estimate_ple(model: Model, root: Any, config: Any, quantization: Dict) -> int:
    layers = _layer_ids(config, "ple_layer_ids")
    if not layers:
        return 0
    vocab = _positive_int(_get(config, "ngram_vocab_size_base"))
    dim = _positive_int(_get(config, "ple_embed_dim"))
    ngram = _positive_int(_get(config, "ngram_size"))
    heads = _positive_int(_get(config, "heads_per_ngram")) * (ngram - 1)
    if heads <= 0 or dim % heads or min(layers) < 1:
        raise ValueError("Invalid PLE table dimensions")
    # Each head has roughly vocab rows and dim/heads columns. The base vocab
    # excludes prime rounding and padding from the table size, leaving those
    # bytes in the GPU estimate.
    # https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
    return (
        vocab
        * dim
        * sum(
            _ple_bytes_per_parameter(model, root, config, quantization, layer)
            for layer in layers
        )
    )


def _dtype_bytes(dtype: Any) -> int:
    sizes = {"float16": 2, "half": 2, "bfloat16": 2, "float32": 4, "float": 4}
    name = str(dtype).removeprefix("torch.")
    if name not in sizes:
        raise ValueError("Unknown table precision")
    return sizes[name]


def _ple_bytes_per_parameter(
    model: Model,
    root: Any,
    config: Any,
    quantization: Dict[str, Any],
    layer: int,
) -> int:
    embedding_dtype = _get(config, "ple_embedding_dtype")
    if embedding_dtype == "float8_e4m3fn":
        return 1
    if embedding_dtype is not None:
        raise ValueError("Unsupported PLE embedding dtype")
    method = quantization.get("quant_method")
    if (quantization and method is None) or method not in (None, "fp8"):
        raise ValueError("Unsupported PLE quantization")
    if method == "fp8" and (
        get_backend(model) == BackendEnum.SGLANG
        or not _ple_excluded(quantization, layer)
    ):
        # The global FP8 scale remains on GPU and is not deducted here.
        return 1
    dtype = (
        _get(config, "dtype")
        or _get(config, "torch_dtype")
        or _get(root, "dtype")
        or _get(root, "torch_dtype")
    )
    size = _dtype_bytes(dtype)
    runtime_dtype = find_last_parameter(model.backend_parameters, ["dtype"])
    if runtime_dtype not in (None, "auto") and _dtype_bytes(runtime_dtype) != size:
        raise ValueError("Runtime table precision differs from the checkpoint")
    return size


def _ple_excluded(quantization: Dict, layer: int) -> bool:
    # vLLM FP8 uses the first nonempty exclusion list and exact module matching.
    excluded = (
        quantization.get("ignored_layers")
        or quantization.get("modules_to_not_convert")
        or []
    )
    if not isinstance(excluded, (list, tuple)) or any(
        not isinstance(entry, str) for entry in excluded
    ):
        raise ValueError("Invalid PLE quantization exclusions")
    suffix = f"layers.{layer - 1}.ple.ple_embedding.ngram_embedding"
    prefixes = [f"model.{suffix}", f"model.language_model.{suffix}"]
    # PLE checkpoint shards form one runtime embedding parameter in vLLM.
    return any(
        entry == prefix or entry.startswith(prefix + ".shard_")
        for entry in excluded
        for prefix in prefixes
    )
