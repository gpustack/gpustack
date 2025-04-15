import asyncio
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
from gpustack.utils.command import find_bool_parameter, find_parameter
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


def add_bool_flag(
    parameters: Optional[List[str]],
    keys: List[str],
    command_list: List[str],
    flag_name: str,
) -> None:
    """
    Adds a boolean flag to the command list if the parameter is found.

    :param parameters: A list of parameters, or None.
    :param keys: A list of keys to search for in the parameters.
    :param command_list: The list of commands to which the flag should be added.
    :param flag_name: The name of the flag to add to the command list.
    """
    if parameters and find_bool_parameter(parameters, keys):
        command_list.append(flag_name)


def add_parameter_with_value(
    parameters: Optional[List[str]],
    keys: List[str],
    command_list: List[str],
    flag_name: str,
) -> None:
    """
    Adds a parameter and its value to the command list if the parameter is found.

    :param parameters: A list of parameters, or None.
    :param keys: A list of keys to search for in the parameters.
    :param command_list: The list of commands to which the parameter and value should be added.
    :param flag_name: The name of the parameter to add to the command list.
    """
    if parameters:
        value = find_parameter(parameters, keys)
        if value:
            command_list.extend([flag_name, value])


def _gguf_parser_env(model: Model) -> dict:
    env = os.environ.copy()
    if model.source == SourceEnum.HUGGING_FACE:
        global_config = get_global_config()
        if global_config.huggingface_token:
            env["HF_TOKEN"] = str(global_config.huggingface_token)
    return env


async def _gguf_parser_command(  # noqa: C901
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    command = "gguf-parser"
    if platform.system() == "windows":
        command += ".exe"

    command_path = pkg_resources.files("gpustack.third_party.bin.gguf-parser").joinpath(
        command
    )
    executable_command = [
        command_path,
        "--in-max-ctx-size",
        "--skip-tokenizer",
        "--skip-architecture",
        "--skip-metadata",
        "--image-vae-tiling",
        "--cache-expiration",
        "168h0m0s",
        "--platform-footprint",
        "150,500",
        "--no-mmap",
        "--json",
    ]

    ctx_size = find_parameter(model.backend_parameters, ["ctx-size", "c"])
    if ctx_size is None:
        ctx_size = "8192"

    executable_command.extend(["--ctx-size", ctx_size])

    add_bool_flag(
        model.backend_parameters,
        ["image-no-text-encoder-model-offload"],
        executable_command,
        "--image-no-text-encoder-model-offload",
    )
    add_bool_flag(
        model.backend_parameters,
        ["image-no-vae-model-offload"],
        executable_command,
        "--image-no-vae-model-offload",
    )
    add_bool_flag(
        model.backend_parameters,
        ["image-no-vae-tiling"],
        executable_command,
        "--image-no-vae-tiling",
    )
    add_bool_flag(
        model.backend_parameters,
        ["flash-attention", "flash-attn", "fa", "diffusion-fa"],
        executable_command,
        "--flash-attention",
    )

    add_parameter_with_value(
        model.backend_parameters,
        ["image-max-height"],
        executable_command,
        "--image-max-height",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["image-max-width"],
        executable_command,
        "--image-max-width",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["visual-max-image-size"],
        executable_command,
        "--visual-max-image-size",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["cache-type-k", "ctk"],
        executable_command,
        "--cache-type-k",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["cache-type-v", "ctv"],
        executable_command,
        "--cache-type-v",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["batch-size", "b"],
        executable_command,
        "--batch-size",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["ubatch-size", "ub"],
        executable_command,
        "--ubatch-size",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["split-mode", "sm"],
        executable_command,
        "--split-mode",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["platform-footprint"],
        executable_command,
        "--platform-footprint",
    )

    cache_dir = kwargs.get("cache_dir")
    if cache_dir:
        executable_command.extend(["--cache-path", cache_dir])

    if offload == GPUOffloadEnum.Full:
        executable_command.extend(["--gpu-layers", "-1"])
    elif offload == GPUOffloadEnum.Partial:
        executable_command.extend(["--gpu-layers-step", "1"])
    elif offload == GPUOffloadEnum.Disable:
        executable_command.extend(["--gpu-layers", "0"])

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
        executable_command.extend(["--tensor-split", tensor_split_str])

    rpc = kwargs.get("rpc")
    if rpc:
        rpc_str = ",".join([v for v in rpc])
        executable_command.extend(["--rpc", rpc_str])

    source_args = await _gguf_parser_command_args_from_source(model, **kwargs)
    executable_command.extend(source_args)
    return executable_command


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
        raise Exception(f"Timeout when getting the file for model {model.name}")
    except Exception as e:
        raise Exception(f"Failed to get the file for model {model.name}, error: {e}")


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
