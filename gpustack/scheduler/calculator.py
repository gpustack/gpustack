import argparse
import asyncio
import dataclasses
import json
from enum import Enum
import logging
import os
import subprocess
from dataclasses import dataclass
import time
from typing import List, Optional, Dict, Tuple, Any
from dataclasses_json import dataclass_json
from transformers import PretrainedConfig

from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.config.config import get_global_config
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.local_path_filter import LocalPathFilter
from gpustack.schemas.models import (
    Model,
    SourceEnum,
    get_mmproj_filename,
    CategoryEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.convert import parse_duration, safe_int
from gpustack.utils.hub import (
    filter_filename,
    list_repo,
    match_hugging_face_files,
    match_model_scope_file_paths,
    get_pretrained_config,
    read_repo_file_content,
)
from gpustack.utils import platform

logger = logging.getLogger(__name__)
fetch_file_timeout_in_seconds = 15


class GPUOffloadEnum(str, Enum):
    Full = "full"
    Partial = "partial"
    Disable = "disable"


@dataclass_json
@dataclass
class LayerMemoryEstimate:
    uma: int
    nonuma: int
    handleLayers: Optional[int] = None


@dataclass_json
@dataclass
class MemoryEstimate:
    fullOffloaded: bool
    ram: LayerMemoryEstimate
    vrams: List[LayerMemoryEstimate]
    offloadLayers: Optional[int] = None  # Not available for diffusion models

    def to_log_string(self) -> str:
        vram_strings = ', '.join(
            [
                f"(uma:{vram.uma}, non-uma:{vram.nonuma}, layers:{vram.handleLayers})"
                for vram in self.vrams
            ]
        )
        return (
            f"layers: {self.offloadLayers}, "
            f"{'full offloaded, ' if self.fullOffloaded else ''}"
            f"ram: (uma:{self.ram.uma}, non-uma:{self.ram.nonuma}, layers:{self.ram.handleLayers}), "
            f"vrams: [{vram_strings}]"
        )


@dataclass_json
@dataclass
class Estimate:
    items: List[MemoryEstimate]
    architecture: str
    embeddingOnly: bool = False
    imageOnly: bool = False
    distributable: bool = False
    reranking: bool = False
    contextSize: Optional[int] = None


@dataclass_json
@dataclass
class Architecture:
    # Describe the model architecture,
    # value from "model", "projector", "adapter" and so on.
    type: Optional[str] = "model"
    # Describe the model architecture name.
    architecture: Optional[str] = None
    # Describe the clip's projector type,
    # only used when type is "projector".
    clipProjectorType: Optional[str] = None
    # Describe the adapter type,
    # only used when type is "adapter".
    adapterType: Optional[str] = None
    # Describe the diffusion model architecture,
    # only used when type is "diffusion".
    diffusionArchitecture: Optional[str] = None
    # Describe the conditioners of the diffusion model,
    # only used when type is "diffusion".
    diffusionConditioners: Optional[List[Dict]] = None
    # Describe the autoencoder of the diffusion model,
    # only used when type is "diffusion".
    diffusionAutoencoder: Optional[Dict] = None

    def is_deployable(self) -> bool:
        """
        Check if the model is deployable.
        Returns:
            bool: True if the model is deployable, False otherwise.
        """

        if self.type in ["projector", "adapter"] and not self.architecture:
            return False

        if self.architecture == "diffusion":
            return bool(self.diffusionConditioners and self.diffusionAutoencoder)

        return True

    def __str__(self) -> str:
        """
        Get a string representation of the architecture.
        """

        if self.type == "projector":
            return f"projector({self.clipProjectorType})"
        elif self.type == "adapter":
            return f"adapter({self.adapterType})"
        else:
            if self.architecture == "diffusion":
                return f"diffusion model({self.diffusionArchitecture})"
            else:
                return f"model({self.architecture})"


@dataclass_json
@dataclass
class GGUFParserOutput:
    estimate: Estimate
    architecture: Optional[Architecture] = None


@dataclass
class ModelResourceClaim:
    model: Model
    resource_claim_estimate: Estimate
    resource_architecture: Optional[Architecture] = None

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        if self.model.id and self.model.updated_at:
            return hash((self.model.id, self.model.updated_at))
        return hash(self.model.model_source_index)

    def __eq__(self, other):
        if isinstance(other, ModelResourceClaim):
            return self.__hash__() == other.model.__hash__()
        return False


def _get_empty_estimate(n_gpu: int = 1) -> Tuple[Estimate, Architecture]:
    empty_layer_memory_estimate = LayerMemoryEstimate(
        uma=0, nonuma=0, handleLayers=None
    )
    memory_estimate = MemoryEstimate(
        offloadLayers=999,
        fullOffloaded=True,
        ram=empty_layer_memory_estimate,
        vrams=[empty_layer_memory_estimate for _ in range(n_gpu)],
    )
    e = Estimate(
        items=[memory_estimate],
        contextSize=0,
        architecture="",
        embeddingOnly=False,
        imageOnly=False,
        distributable=False,
        reranking=False,
    )
    a = Architecture()
    return e, a


def _gguf_parser_env(model: Model) -> dict:
    env = os.environ.copy()
    if model.source == SourceEnum.HUGGING_FACE:
        global_config = get_global_config()
        if global_config.huggingface_token:
            env["HF_TOKEN"] = str(global_config.huggingface_token)
    return env


class NoExitArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise argparse.ArgumentError(None, message)


@dataclass
class GGUFParserCommandMutableParameters:
    # NB(thxCode): Partial options are not applied to backend, but to the parser.
    # We can receive these options from the backend advanced config.

    backend_version: Optional[str] = None

    # Estimate
    flash_attention: Optional[bool] = None
    main_gpu: Optional[int] = None
    parallel_size: int = 4
    platform_footprint: str = "150,500"
    # Estimate/LLaMACpp
    batch_size: Optional[int] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    ctx_size: int = 8192
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    rope_scale: Optional[float] = None
    rope_scaling: Optional[str] = None
    yarn_orig_ctx: Optional[int] = None
    override_tensor: Optional[List[str]] = None
    gpu_layers_draft: Optional[int] = None
    mmap: bool = False
    no_kv_offload: Optional[bool] = None
    split_mode: Optional[str] = None
    ubatch_size: Optional[int] = None
    visual_max_image_size: Optional[int] = None
    max_projected_cache: int = 10
    swa_full: bool = False
    # Estimate/StableDiffusionCpp
    image_autoencoder_tiling: bool = True
    image_batch_count: Optional[int] = None
    image_free_compute_memory_immediately: Optional[bool] = None
    image_height: Optional[int] = None
    image_no_autoencoder_offload: Optional[bool] = None
    image_no_conditioner_offload: Optional[bool] = None
    image_no_control_net_offload: Optional[bool] = None
    image_width: Optional[int] = None
    # Load
    cache_expiration: str = "0"
    skip_cache: Optional[bool] = None
    skip_dns_cache: Optional[bool] = None
    skip_proxy: Optional[bool] = None
    skip_range_download_detect: Optional[bool] = None
    skip_tls_verify: Optional[bool] = None
    header: Optional[List[str]] = None

    def from_args(self, args: List[str]):
        parser = NoExitArgumentParser(exit_on_error=False, allow_abbrev=False)

        # Default any True arguments here,
        # so that they can be set to False later.
        parser.set_defaults(image_autoencoder_tiling=True)

        # Estimate
        parser.add_argument(
            "--flash-attention",
            "--flash-attn",
            "--diffusion-fa",
            "-fa",
            type=bool,
            action=argparse.BooleanOptionalAction,  # generated "--no-flash-attention", "--no-flash-attn", "--no-diffusion-fa"
            required=False,
        )
        parser.add_argument(
            "--main-gpu",
            "-mg",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--parallel-size",
            "--parallel",
            "-np",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--platform-footprint",
            type=str,
            required=False,
        )
        # Estimate/LLaMACpp
        parser.add_argument(
            "--batch-size",
            "-b",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--cache-type-k",
            "-ctk",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--cache-type-v",
            "-ctv",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--ctx-size",
            "-c",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--rope-freq-base",
            type=float,
            required=False,
        )
        parser.add_argument(
            "--rope-freq-scale",
            type=float,
            required=False,
        )
        parser.add_argument(
            "--rope-scale",
            type=float,
            required=False,
        )
        parser.add_argument(
            "--rope-scaling",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--yarn-orig-ctx",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--override-tensor",
            "-ot",
            action='append',
            required=False,
        )
        parser.add_argument(
            "--gpu-layers-draft",
            "--n-gpu-layers-draft",
            "-ngld",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--mmap",
            type=bool,
            action=argparse.BooleanOptionalAction,  # generated "--no-mmap"
            required=False,
        )
        parser.add_argument(
            "--no-kv-offload",
            "-nkvo",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--split-mode",
            "-sm",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--ubatch-size",
            "-ub",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--visual-max-image-size",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--max-projected-cache",
            "--visual-max-image-cache",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--swa-full",
            action='store_true',
            required=False,
        )
        # Estimate/StableDiffusionCpp
        parser.add_argument(
            "--image-autoencoder-tiling",
            "--image-vae-tiling",
            "--vae-tiling",
            action='store_true',
            dest="image_autoencoder_tiling",
            required=False,
        )
        parser.add_argument(
            "--image-no-autoencoder-tiling",
            "--image-no-vae-tiling",
            action='store_false',
            dest="image_autoencoder_tiling",
            required=False,
        )
        parser.add_argument(
            "--image-batch-count",
            "--batch-count",
            "--image-max-batch",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--image-free-compute-memory-immediately",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--image-height",
            "--height",
            "--image-max-height",
            type=int,
            required=False,
        )
        parser.add_argument(
            "--image-no-autoencoder-offload",
            "--vae-on-cpu",
            "--image-no-vae-model-offload",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--image-no-conditioner-offload",
            "--clip-on-cpu",
            "--image-no-text-encoder-model-offload",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--image-no-control-net-offload",
            "--control-net-cpu",
            "--image-no-control-net-model-offload",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--image-width",
            "--width",
            "--image-max-width",
            type=int,
            required=False,
        )
        # Load
        parser.add_argument(
            "--cache-expiration",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--skip-cache",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--skip-dns-cache",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--skip-proxy",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--skip-range-download-detect",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--skip-tls-verify",
            action='store_true',
            required=False,
        )
        parser.add_argument(
            "--header",
            action='append',
            required=False,
        )

        slogger = logger.getChild("gguf_parser_command")

        try:
            args_parsed = parser.parse_known_args(args=args)
            for attr_name in [attr.name for attr in dataclasses.fields(self.__class__)]:
                try:
                    attr_value = getattr(args_parsed[0], attr_name, None)
                    if attr_value is not None:
                        try:
                            setattr(self, attr_name, attr_value)
                        except ValueError as e:
                            slogger.warning(
                                f"Failed to receive mutable parameter {attr_name}: {e}"
                            )
                except AttributeError:
                    # If reach here, that means the field is an internal property,
                    # which would not register in the argument parser.
                    pass
        except (argparse.ArgumentError, argparse.ArgumentTypeError) as e:
            slogger.warning(f"Failed to parse mutable parameters: {e}")

    def extend_command(self, command: List[str]):
        internal_properties = [
            "backend_version",
        ]

        for attr_name in [attr.name for attr in dataclasses.fields(self.__class__)]:
            if attr_name in internal_properties:
                # Skip internal properties.
                continue

            attr_value = getattr(self, attr_name, None)
            if attr_value is not None:
                if isinstance(attr_value, bool):
                    command.append(
                        f"--{attr_name.replace('_', '-')}={'true' if attr_value else 'false'}"
                    )
                elif isinstance(attr_value, int):
                    command.append(f"--{attr_name.replace('_', '-')}={str(attr_value)}")
                elif isinstance(attr_value, list):
                    for sv in attr_value:
                        command.append(f"--{attr_name.replace('_', '-')}={str(sv)}")
                else:
                    command.append(f"--{attr_name.replace('_', '-')}={str(attr_value)}")


async def _gguf_parser_command(
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    bin_path = pkg_resources.files("gpustack.third_party.bin.gguf-parser").joinpath(
        "gguf-parser" + (".exe" if platform.system() == "windows" else "")
    )

    # Preset the command with immutable arguments.
    command = [
        bin_path,
        "--skip-tokenizer",
        "--skip-metadata",
        "--json",
    ]

    # Extend the command with mutable arguments.
    params = GGUFParserCommandMutableParameters(backend_version=model.backend_version)
    params.from_args(model.backend_parameters)
    params.extend_command(command)
    # Extend the command with controlled arguments.
    cache_dir = kwargs.get("cache_dir")
    if cache_dir:
        command.extend(["--cache-path", cache_dir])

    if offload == GPUOffloadEnum.Full:
        command.extend(["--gpu-layers", "-1"])
    elif offload == GPUOffloadEnum.Partial:
        command.extend(["--gpu-layers-step", "1"])
    elif offload == GPUOffloadEnum.Disable:
        command.extend(["--gpu-layers", "0"])

    tensor_split = kwargs.get("tensor_split")
    if tensor_split:
        if all(i < 1024 * 1024 for i in tensor_split):
            # user provided
            tensor_split_str = ",".join([str(i) for i in tensor_split])
        else:
            # computed by the system, convert to MiB to prevent overflow
            tensor_split_str = ",".join(
                [str(int(i / (1024 * 1024))) for i in tensor_split]
            )
        command.extend(["--tensor-split", tensor_split_str])

    rpc = kwargs.get("rpc")
    if rpc:
        rpc_str = ",".join([v for v in rpc])
        command.extend(["--rpc", rpc_str])
    source_args = await _gguf_parser_command_args_from_source(
        model, cache_expiration=params.cache_expiration, **kwargs
    )
    command.extend(source_args)

    return command


async def _try_parse_on_workers(
    model: Model,
    workers: List[Worker],
    offload: GPUOffloadEnum,
    **kwargs,
) -> Optional[ModelResourceClaim]:
    """
    Try to parse GGUF on specified workers concurrently.
    """
    if not workers:
        return None

    # Prepare parameters once
    offload_str = offload.value  # "full", "partial", "disable"

    # Prepare override parameters (only pass necessary ones)
    parse_kwargs = {}
    for key in ("tensor_split", "rpc"):
        if key in kwargs:
            parse_kwargs[key] = kwargs[key]

    async def try_parse_on_worker(worker: Worker) -> Optional[ModelResourceClaim]:
        """Try to parse GGUF on a single worker."""
        try:
            async with WorkerFilesystemClient() as fs_client:
                output_dict = await fs_client.parse_gguf(
                    worker,
                    model,
                    offload=offload_str,
                    **parse_kwargs,
                )
                claim = GGUFParserOutput.from_dict(output_dict)
                if offload == GPUOffloadEnum.Disable:
                    clear_vram_claim(claim)

                logger.info(
                    f"Successfully parsed GGUF on worker {worker.name} "
                    f"for model {model.name}"
                )
                return ModelResourceClaim(
                    model=model,
                    resource_claim_estimate=claim.estimate,
                    resource_architecture=claim.architecture,
                )
        except Exception as e:
            logger.debug(f"Failed to parse GGUF on worker {worker.name}: {e}")
            return None

    # Concurrently try all workers and return the first successful result
    tasks = [try_parse_on_worker(worker) for worker in workers]

    # Use as_completed to get results as they finish
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        if result:
            return result

    return None


async def get_pretrained_config_with_workers(
    model: Model,
    workers: Optional[List[Worker]] = None,
    trust_remote_code: bool = False,
) -> Optional[Any]:
    """
    Unified async entry point for getting pretrained config.

    Handles all model sources with appropriate fallback strategies:
    - For LOCAL_PATH model which is not available locally, get from workers
    - For others, AutoConfig and fallback to config.json

    Args:
        model: Model to get config for
        workers: Available workers (for LOCAL_PATH)
        trust_remote_code: Whether to trust remote code

    Returns:
        PretrainedConfig object or None

    Raises:
        ValueError: If config is required but cannot be loaded
    """
    pretrained_config = None
    timeout_in_seconds = 15

    try:
        if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(
            model.local_path
        ):
            pretrained_config = await get_pretrained_config_from_workers(
                model,
                workers,
            )
        else:
            pretrained_config = await asyncio.wait_for(
                asyncio.to_thread(
                    get_pretrained_config, model, trust_remote_code=trust_remote_code
                ),
                timeout=timeout_in_seconds,
            )
    except Exception as e:
        if should_fallback_load_config_json(e, model):
            config_dict = await asyncio.wait_for(
                asyncio.to_thread(
                    read_repo_file_content,
                    model,
                    "config.json",
                    token=get_global_config().huggingface_token,
                ),
                timeout=timeout_in_seconds,
            )
            if config_dict:
                return PretrainedConfig.from_dict(config_dict)

            # If config_dict is None for LOCAL_PATH, provide a clearer error message
            if model.source == SourceEnum.LOCAL_PATH:
                raise ValueError(
                    f"Model path '{model.local_path}' does not exist or config.json is not found. "
                    f"Please ensure the model files are available at the specified path."
                )

        if model.env and model.env.get("GPUSTACK_SKIP_MODEL_EVALUATION"):
            return pretrained_config

        # Error handling for different model categories
        if any(
            cat in model.categories
            for cat in [CategoryEnum.IMAGE, CategoryEnum.UNKNOWN]
        ):
            return pretrained_config

        if pretrained_config is None and (
            CategoryEnum.LLM in model.categories or isinstance(e, ValueError)
        ):
            raise e

    return pretrained_config


def should_fallback_load_config_json(e: Exception, model: Model) -> bool:
    """
    Determine whether to fallback to loading config.json based on the exception and model.

    Args:
        e: The exception encountered during loading
        model: The model being processed
    Returns:
        bool: True if should fallback to loading config.json, False otherwise
    """

    # For LOCAL_PATH models, the path must be a valid directory
    if model.source == SourceEnum.LOCAL_PATH and not (
        model.local_path and os.path.isdir(model.local_path)
    ):
        return False

    # Fallback for backend version specified or import errors
    return model.backend_version is not None or isinstance(e, ImportError)


async def check_diffusers_model_index_from_workers(
    model: Model,
    workers: List[Worker],
) -> bool:
    """
    Check if a LOCAL_PATH model is a diffusers model by querying workers.

    This function is specifically for LOCAL_PATH models that are not available
    locally on the server. It uses the optimized worker query strategy.

    Args:
        model: Model with source LOCAL_PATH
        workers: List of workers to query

    Returns:
        True if model_index.json contains _diffusers_version, False otherwise
    """
    if not workers:
        return False

    # Read model_index.json from workers
    data = await read_local_path_file_from_workers(model, "model_index.json", workers)

    if data is None:
        return False

    # Check for _diffusers_version key
    if isinstance(data, dict) and "_diffusers_version" in data:
        return True
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "_diffusers_version" in item:
                return True

    return False


async def read_local_path_file_from_workers(  # noqa: C901
    model: Model,
    file_path: str,
    workers: List[Worker],
) -> Optional[Dict[str, Any]]:
    """
    Read a file from LOCAL_PATH model by querying workers.

    Steps:
    1. Apply filters (GPU selector, label selector) to reduce broadcast scope
    2. Broadcast to filtered workers

    Args:
        model: Model with source LOCAL_PATH
        file_path: Relative path to the file (e.g., "config.json", "model_index.json")
        workers: List of workers to query

    Returns:
        File content as dict if successful, None otherwise
    """
    if not workers:
        return None

    # Build full file path
    fp = os.path.join(model.local_path, file_path)

    async def try_read_from_worker(worker: Worker) -> Optional[Dict[str, Any]]:
        """Try to read file from a single worker."""
        try:
            async with WorkerFilesystemClient() as filesystem_client:
                logger.info(f"Trying to read {file_path} from worker {worker.id}")
                content = await filesystem_client.read_model_config(worker, fp)
                if content:
                    logger.info(
                        f"Successfully read {file_path} from worker {worker.id}"
                    )
                    return content
                return None
        except Exception as e:
            logger.debug(f"Failed to read {file_path} from worker {worker.id}: {e}")
            return None

    # Step 1: Apply filters to reduce broadcast scope
    filtered_workers = workers
    messages = []

    # Apply GPUMatchingFilter
    if model.gpu_selector:
        gpu_filter = GPUMatchingFilter(model)
        filtered_workers, gpu_messages = await gpu_filter.filter(filtered_workers)
        messages.extend(gpu_messages)

    # Apply LabelMatchingFilter
    if model.worker_selector:
        label_filter = LabelMatchingFilter(model)
        filtered_workers, label_messages = await label_filter.filter(filtered_workers)
        messages.extend(label_messages)

    # Apply LocalPathFilter for LOCAL_PATH models
    local_path_filter = LocalPathFilter(model)
    filtered_workers, local_path_messages = await local_path_filter.filter(
        filtered_workers
    )
    messages.extend(local_path_messages)

    if messages:
        for msg in messages:
            logger.info(f"Worker filtering for {file_path} read: {msg}")

    # Step 2: Broadcast to filtered workers
    if not filtered_workers:
        logger.warning(f"No workers available after filtering for {file_path}")
        return None

    logger.info(
        f"Broadcasting {file_path} read request to {len(filtered_workers)} filtered workers "
        f"(reduced from {len(workers)} total workers)"
    )
    tasks = [try_read_from_worker(worker) for worker in filtered_workers]
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        if result:
            return result

    return None


async def get_pretrained_config_from_workers(
    model: Model,
    workers: List[Worker],
) -> Optional[Any]:
    """
    Get pretrained config from remote workers for LOCAL_PATH models.

    Args:
        model: Model with source LOCAL_PATH
        workers: List of workers to query

    Returns:
        PretrainedConfig object if successful, None otherwise
    """
    if not workers:
        return None

    config_dict = await read_local_path_file_from_workers(model, "config.json", workers)

    if not config_dict:
        return None

    from transformers import PretrainedConfig

    return PretrainedConfig.from_dict(config_dict)


async def _calculate_from_workers(  # noqa: C901
    model: Model,
    workers: List[Worker],
    offload: GPUOffloadEnum,
    **kwargs,
) -> Optional[ModelResourceClaim]:
    """
    Calculate model resource claim by running gguf-parser on a worker.

    Args:
        model: Model to calculate the resource claim for.
        workers: List of available workers.
        offload: GPU offload strategy.
        kwargs: Additional arguments to pass to the GGUF parser.

    Returns:
        ModelResourceClaim if successful, None otherwise.
    """
    # Step 1: Apply worker filters before broadcasting
    filtered_workers = workers
    messages = []

    # Apply GPUMatchingFilter
    if model.gpu_selector:
        from gpustack.policies.worker_filters.gpu_matching_filter import (
            GPUMatchingFilter,
        )

        gpu_filter = GPUMatchingFilter(model)
        filtered_workers, gpu_messages = await gpu_filter.filter(filtered_workers)
        messages.extend(gpu_messages)

    # Apply LabelMatchingFilter
    if model.worker_selector:
        from gpustack.policies.worker_filters.label_matching_filter import (
            LabelMatchingFilter,
        )

        label_filter = LabelMatchingFilter(model)
        filtered_workers, label_messages = await label_filter.filter(filtered_workers)
        messages.extend(label_messages)

    # Apply LocalPathFilter for LOCAL_PATH models
    local_path_filter = LocalPathFilter(model)
    filtered_workers, local_path_messages = await local_path_filter.filter(
        filtered_workers
    )
    messages.extend(local_path_messages)

    if messages:
        for msg in messages:
            logger.info(f"Worker filtering for GGUF parsing: {msg}")

    # Step 2: Broadcasting to filtered workers
    if not filtered_workers:
        logger.warning("No workers available after filtering for GGUF parsing")
        return None

    logger.info(
        f"Broadcasting GGUF parse request to {len(filtered_workers)} filtered workers "
        f"(reduced from {len(workers)} total workers) for model {model.name}"
    )
    return await _try_parse_on_workers(model, filtered_workers, offload, **kwargs)


async def calculate_gguf_model_resource_claim(
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    workers: Optional[List[Worker]] = None,
    **kwargs,
) -> ModelResourceClaim:
    """
    Calculate the resource claim of the model.
    Args:
        model: Model to calculate the resource claim for.
        offload: GPU offload strategy.
        workers: Optional list of available workers for remote parsing.
        kwargs: Additional arguments to pass to the GGUF parser.
    """

    if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(model.local_path):
        # Try to calculate on worker if workers are provided
        if workers:
            try:
                result = await _calculate_from_workers(
                    model, workers, offload, **kwargs
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(
                    f"Failed to calculate on worker: {e}, falling back to empty estimate"
                )

        # Skip the calculation if the model is not available, policies like spread strategy still apply.
        # TODO Support user provided resource claim for better scheduling.
        e, a = _get_empty_estimate()
        tensor_split = kwargs.get("tensor_split")
        if tensor_split:
            e, a = _get_empty_estimate(n_gpu=len(tensor_split))
        return ModelResourceClaim(
            model=model,
            resource_claim_estimate=e,
            resource_architecture=a,
        )

    command = await _gguf_parser_command(model, offload, **kwargs)
    env = _gguf_parser_env(model)
    try:
        start_time = time.time()
        logger.trace(
            f"Running parser for model {model.name} with command: {' '.join(map(str, command))}"
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )

        cmd_output = stdout.decode()
        claim: GGUFParserOutput = GGUFParserOutput.from_json(cmd_output)
        latency = time.time() - start_time

        if offload == GPUOffloadEnum.Full:
            logger.trace(
                f"Finished running parser for full offload model instance {model.name}, latency: {latency:.2f}, "
                f"{claim.estimate.items[0].to_log_string()}"
            )
        elif offload == GPUOffloadEnum.Partial:
            logger.trace(
                f"Finished running parser for partial offloading model instance {model.name}, latency: {latency:.2f}, at least: "
                f"{claim.estimate.items[1].to_log_string() if len(claim.estimate.items) > 1 else claim.estimate.items[0].to_log_string()}"
            )
        elif offload == GPUOffloadEnum.Disable:
            logger.trace(
                f"Finished running parser for disabled offloading model instance {model.name}, latency: {latency:.2f}, "
                f"{claim.estimate.items[0].to_log_string()}"
            )
            clear_vram_claim(claim)

        return ModelResourceClaim(
            model=model,
            resource_claim_estimate=claim.estimate,
            resource_architecture=claim.architecture,
        )
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Failed to execute {command}, error: {e}, "
            + f"stderr: {e.stderr.decode()}, "
            + f"stdout: {e.stdout.decode()}"
        )
        raise Exception(e.stderr.decode() if stderr else e.stdout.decode()) from e
    except Exception as e:
        raise Exception(
            f"Failed to parse the output of {command}, error: {e}",
        )


def clear_vram_claim(claim: GGUFParserOutput):
    for item in claim.estimate.items:
        # gguf-parser provides vram claim when offloadLayers is 0 due to current llama.cpp behavior, but llama-box won't allocate such vram.
        if item.offloadLayers == 0:
            item.vrams = [
                LayerMemoryEstimate(uma=0, nonuma=0, handleLayers=0) for _ in item.vrams
            ]


async def _gguf_parser_command_args_from_source(model: Model, **kwargs) -> List[str]:
    """
    Get the model url based on the model source.
    Args:
        model: Model to get the url for.
    """

    if model.source not in [
        SourceEnum.HUGGING_FACE,
        SourceEnum.MODEL_SCOPE,
        SourceEnum.LOCAL_PATH,
    ]:
        raise ValueError(f"Unsupported source: {model.source}")

    try:
        if model.source in [SourceEnum.HUGGING_FACE, SourceEnum.MODEL_SCOPE]:
            cache_expiration = kwargs.get("cache_expiration")
            if cache_expiration and cache_expiration != "0":
                cache_expiration = parse_duration(cache_expiration)
            cache_expiration = safe_int(cache_expiration)
            if model.source == SourceEnum.HUGGING_FACE:
                repo_arg, file_arg, mmproj_arg = [
                    "--hf-repo",
                    "--hf-file",
                    "--hf-mmproj-file",
                ]
                repo_id = model.huggingface_repo_id
                file_name = model.huggingface_filename
            else:
                repo_arg, file_arg, mmproj_arg = [
                    "--ms-repo",
                    "--ms-file",
                    "--ms-mmproj-file",
                ]
                repo_id = model.model_scope_model_id
                file_name = model.model_scope_file_path

            args = [repo_arg, repo_id]

            global_config = get_global_config()
            repo_file_infos = await asyncio.wait_for(
                asyncio.to_thread(
                    list_repo,
                    repo_id,
                    model.source,
                    global_config.huggingface_token,
                    cache_expiration,
                ),
                timeout=fetch_file_timeout_in_seconds,
            )
            repo_files = [file.get("name", "") for file in repo_file_infos]
            model_filename = filter_filename(file_name, repo_files)
            if len(model_filename) == 0:
                raise ValueError(f"File {model_filename} not found in {repo_id}")

            args.extend([file_arg, model_filename[0]])

            mmproj_filename = get_mmproj_filename(model)
            mmproj_filename = filter_filename(mmproj_filename, repo_files)
            if mmproj_filename:
                args.extend([mmproj_arg, mmproj_filename[0]])

            return args
        elif model.source == SourceEnum.LOCAL_PATH:
            return ["--path", model.local_path]
    except asyncio.TimeoutError:
        raise Exception(
            f"Timeout when getting the file for model {model.name or model.readable_source}"
        )
    except Exception as e:
        raise Exception(
            f"Failed to get the file for model {model.name or model.readable_source}, error: {e}"
        )


def read_model_index_json(path: str) -> dict:
    """
    Read and parse model_index.json from local directory.

    Args:
        path: Directory path containing model_index.json

    Returns:
        Parsed JSON data from model_index.json, or None if not found

    Raises:
        json.JSONDecodeError: If model_index.json is invalid
        PermissionError: If permission denied
        OSError: For other I/O errors
    """
    model_index_path = os.path.join(path, "model_index.json")

    if not os.path.exists(model_index_path):
        return None

    try:
        with open(model_index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except PermissionError:
        logger.error(f"Permission denied reading model_index.json: {model_index_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse model_index.json: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to read model_index.json: {e}")
        raise


def calculate_llm_model_weight_size(path: str) -> int:
    """
    Calculate total size of LLM model weights in root directory.

    Args:
        path: Directory path to scan

    Returns:
        Total size in bytes of weight files

    Raises:
        FileNotFoundError: If path doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If permission denied
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    total_size = 0

    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(weight_file_extensions):
                    total_size += entry.stat().st_size
    except PermissionError:
        logger.error(f"Permission denied when accessing '{path}'.")
        raise

    return total_size


def calculate_diffusion_model_weight_size(path: str) -> int:
    """
    Calculate total size of diffusion model weights.

    Logic:
    1. Read model_index.json to get pipeline components
    2. Scan subdirectories defined in pipeline
    3. Sum up weight files (.safetensors, .bin, .pt, .pth)

    Args:
        path: Directory path containing model_index.json

    Returns:
        Total size in bytes of weight files

    Raises:
        FileNotFoundError: If model_index.json not found or path doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If permission denied
        json.JSONDecodeError: If model_index.json is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")

    # Read pipeline definition
    pipeline_data = read_model_index_json(path)

    if pipeline_data is None:
        raise FileNotFoundError(f"model_index.json not found in {path}")

    if not isinstance(pipeline_data, dict):
        raise TypeError(f"model_index.json in {path} is not a valid JSON object.")

    # Remove metadata keys (starting with _)
    component_dirs = {key for key in pipeline_data.keys() if not key.startswith('_')}

    total_size = 0

    # Scan each component directory
    for component_dir in component_dirs:
        component_path = os.path.join(path, component_dir)

        if not os.path.isdir(component_path):
            continue

        # Scan files in component directory
        try:
            with os.scandir(component_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(weight_file_extensions):
                        total_size += entry.stat().st_size
        except PermissionError:
            logger.error(f"Permission denied scanning directory: {component_path}")
            raise
        except OSError as e:
            logger.error(f"Error scanning directory {component_path}: {e}")
            raise

    return total_size


def calculate_local_model_weight_size(path: str, is_diffusion: bool = False) -> int:
    """
    Calculate model weight size based on model type.

    Unified entry point for calculating model weight sizes.

    Args:
        path: Directory path to scan
        is_diffusion: Whether this is a diffusion model (default: False)

    Returns:
        Total size in bytes of weight files

    Raises:
        FileNotFoundError: If path doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If permission denied
        json.JSONDecodeError: If model_index.json is invalid (diffusion only)
    """
    if is_diffusion:
        return calculate_diffusion_model_weight_size(path)
    else:
        return calculate_llm_model_weight_size(path)


def hf_model_filename(
    repo_id: str, filename: Optional[str] = None, token: Optional[str] = None
) -> Optional[str]:
    if filename is None:
        return None
    else:
        matching_files = match_hugging_face_files(repo_id, filename, None, token)
        if len(matching_files) == 0:
            raise ValueError(f"File {filename} not found in {repo_id}")

        return matching_files[0]


def hf_mmproj_filename(model: Model, token: Optional[str] = None) -> Optional[str]:
    mmproj_filename = get_mmproj_filename(model)
    matching_files = match_hugging_face_files(
        model.huggingface_repo_id, mmproj_filename, None, token
    )
    if len(matching_files) == 0:
        return None

    matching_files = sorted(matching_files, reverse=True)

    return matching_files[0]


def model_scope_file_path(model_id: str, file_path: str) -> str:
    file_paths = match_model_scope_file_paths(model_id, file_path)
    if len(file_paths) == 0:
        raise ValueError(f"File {file_path} not found in {model_id}")
    return file_paths[0]


def model_scope_mmproj_file_path(model: Model) -> Optional[str]:
    mmproj_filename = get_mmproj_filename(model)
    file_paths = match_model_scope_file_paths(
        model.model_scope_model_id, mmproj_filename
    )
    if len(file_paths) == 0:
        return None

    file_paths = sorted(file_paths, reverse=True)

    return file_paths[0]
