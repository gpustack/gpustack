import argparse
import asyncio
import dataclasses
from enum import Enum
import logging
import os
import subprocess
from dataclasses import dataclass
import time
from typing import List, Optional
from dataclasses_json import dataclass_json


from gpustack.config.config import get_global_config
from gpustack.schemas.models import (
    Model,
    SourceEnum,
    get_mmproj_filename,
)
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.hub import match_hugging_face_files, match_model_scope_file_paths
from gpustack.utils import platform

logger = logging.getLogger(__name__)
fetch_file_timeout_in_seconds = 15


class GPUOffloadEnum(str, Enum):
    Full = "full"
    Partial = "partial"
    Disable = "disable"


@dataclass_json
@dataclass
class layerMemoryEstimate:
    uma: int
    nonuma: int
    handleLayers: Optional[int] = None


@dataclass_json
@dataclass
class memoryEstimate:
    fullOffloaded: bool
    ram: layerMemoryEstimate
    vrams: List[layerMemoryEstimate]
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
class estimate:
    items: List[memoryEstimate]
    architecture: str
    embeddingOnly: bool = False
    imageOnly: bool = False
    distributable: bool = False
    reranking: bool = False
    contextSize: Optional[int] = None


@dataclass_json
@dataclass
class ggufParserOutput:
    estimate: estimate


@dataclass
class ModelResourceClaim:
    model: Model
    resource_claim_estimate: estimate

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        if self.model.id and self.model.updated_at:
            return hash((self.model.id, self.model.updated_at))
        return hash(self.model.model_source_index)

    def __eq__(self, other):
        if isinstance(other, ModelResourceClaim):
            return self.__hash__() == other.model.__hash__()
        return False


def _get_empty_estimate(n_gpu: int = 1) -> estimate:
    empty_layer_memory_estimate = layerMemoryEstimate(
        uma=0, nonuma=0, handleLayers=None
    )
    memory_estimate = memoryEstimate(
        offloadLayers=999,
        fullOffloaded=True,
        ram=empty_layer_memory_estimate,
        vrams=[empty_layer_memory_estimate for _ in range(n_gpu)],
    )
    return estimate(
        items=[memory_estimate],
        contextSize=0,
        architecture="",
        embeddingOnly=False,
        imageOnly=False,
        distributable=False,
        reranking=False,
    )


def _gguf_parser_env(model: Model) -> dict:
    env = os.environ.copy()
    if model.source == SourceEnum.HUGGING_FACE:
        global_config = get_global_config()
        if global_config.huggingface_token:
            env["HF_TOKEN"] = str(global_config.huggingface_token)
    return env


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
    cache_expiration: str = "168h0m0s"
    skip_cache: Optional[bool] = None
    skip_dns_cache: Optional[bool] = None
    skip_proxy: Optional[bool] = None
    skip_range_download_detect: Optional[bool] = None
    skip_tls_verify: Optional[bool] = None

    def from_args(self, args: List[str]):
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False)

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

        if self.backend_version:
            # Parser v0.18.0+ supports estimating Sliding Window Attention (SWA) usage,
            # however, llama-box treats `--batch-size` as the same as `--ctx-size` within [v0.0.140, v0.0.148],
            # so we need to set `--batch-size` to `--ctx-size` to avoid wrong RAM/VRAM estimation.
            if "v0.0.139" < self.backend_version < "v0.0.149":
                command.append(f"--batch-size={self.ctx_size}")


async def _gguf_parser_command(  # noqa: C901
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    bin_path = pkg_resources.files("gpustack.third_party.bin.gguf-parser").joinpath(
        "gguf-parser" + (".exe" if platform.system() == "windows" else "")
    )

    # Preset the command with immutable arguments.
    command = [
        bin_path,
        "--skip-tokenizer",
        "--skip-architecture",
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

    source_args = await _gguf_parser_command_args_from_source(model, **kwargs)
    command.extend(source_args)

    return command


async def calculate_model_resource_claim(
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    **kwargs,
) -> ModelResourceClaim:
    """
    Calculate the resource claim of the model.
    Args:
        model: Model to calculate the resource claim for.
        offload: GPU offload strategy.
        kwargs: Additional arguments to pass to the GGUF parser.
    """

    if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(model.local_path):
        # Skip the calculation if the model is not available, policies like spread strategy still apply.
        # TODO Support user provided resource claim for better scheduling.
        estimate = _get_empty_estimate()
        tensor_split = kwargs.get("tensor_split")
        if tensor_split:
            estimate = _get_empty_estimate(n_gpu=len(tensor_split))
        return ModelResourceClaim(model, estimate)

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
        claim: ggufParserOutput = ggufParserOutput.from_json(cmd_output)
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

        return ModelResourceClaim(model, claim.estimate)

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to execute {command}, error: {e}, ",
            f"stderr: {e.stderr.decode()}, ",
            f"stdout: {e.stdout.decode()}",
        )
    except Exception as e:
        raise Exception(
            f"Failed to parse the output of {command}, error: {e}",
        )


def clear_vram_claim(claim: ggufParserOutput):
    for item in claim.estimate.items:
        # gguf-parser provides vram claim when offloadLayers is 0 due to current llama.cpp behavior, but llama-box won't allocate such vram.
        if item.offloadLayers == 0:
            item.vrams = [
                layerMemoryEstimate(uma=0, nonuma=0, handleLayers=0) for _ in item.vrams
            ]


async def _gguf_parser_command_args_from_source(  # noqa: C901
    model: Model, **kwargs
) -> List[str]:
    """
    Get the model url based on the model source.
    Args:
        model: Model to get the url for.
    """

    if model.source not in [
        SourceEnum.OLLAMA_LIBRARY,
        SourceEnum.HUGGING_FACE,
        SourceEnum.MODEL_SCOPE,
        SourceEnum.LOCAL_PATH,
    ]:
        raise ValueError(f"Unsupported source: {model.source}")

    try:
        if model.source == SourceEnum.OLLAMA_LIBRARY:
            args = ["-ol-model", model.ollama_library_model_name]
            ol_base_url = kwargs.get("ollama_library_base_url")
            if ol_base_url:
                args.extend(["-ol-base-url", ol_base_url])
            return args
        elif model.source == SourceEnum.HUGGING_FACE:
            global_config = get_global_config()

            args = ["-hf-repo", model.huggingface_repo_id]
            if model.huggingface_filename:
                model_filename = await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_model_filename,
                        model.huggingface_repo_id,
                        model.huggingface_filename,
                        global_config.huggingface_token,
                    ),
                    timeout=fetch_file_timeout_in_seconds,
                )
                args.extend(["-hf-file", model_filename])

                mmproj_filename = await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_mmproj_filename,
                        model,
                        global_config.huggingface_token,
                    ),
                    timeout=fetch_file_timeout_in_seconds,
                )
                if mmproj_filename:
                    args.extend(["--hf-mmproj-file", mmproj_filename])

            return args
        elif model.source == SourceEnum.MODEL_SCOPE:
            file_path = await asyncio.wait_for(
                asyncio.to_thread(
                    model_scope_file_path,
                    model.model_scope_model_id,
                    model.model_scope_file_path,
                ),
                timeout=fetch_file_timeout_in_seconds,
            )
            args = ["-ms-repo", model.model_scope_model_id, "-ms-file", file_path]

            mmproj_file_path = await asyncio.wait_for(
                asyncio.to_thread(
                    model_scope_mmproj_file_path,
                    model,
                ),
                timeout=fetch_file_timeout_in_seconds,
            )
            if mmproj_file_path:
                args.extend(["--ms-mmproj-file", mmproj_file_path])

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
