import asyncio
from enum import Enum
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json


from gpustack.config.config import get_global_config
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    SourceEnum,
    get_mmproj_filename,
)
from gpustack.utils.command import find_bool_parameter, find_parameter
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.hub import match_hugging_face_files, match_model_scope_file_paths
from gpustack.utils import platform

logger = logging.getLogger(__name__)
fetch_file_timeout_in_seconds = 5


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
class modelResoruceClaim:
    estimate: estimate


@dataclass
class ModelInstanceResourceClaim:
    model_instance: ModelInstance
    resource_claim_estimate: estimate

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        return self.model_instance.id

    # compare the model instance id
    def __eq__(self, other):
        if isinstance(other, ModelInstanceResourceClaim):
            return self.model_instance.id == other.model_instance.id
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


async def _gguf_parser_command(  # noqa: C901
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    command = "gguf-parser"
    if platform.system() == "windows":
        command += ".exe"

    command_path = pkg_resources.files("gpustack.third_party.bin.gguf-parser").joinpath(
        command
    )
    execuable_command = [
        command_path,
        "--in-max-ctx-size",
        "--skip-tokenizer",
        "--skip-architecture",
        "--skip-metadata",
        "--image-vae-tiling",
        "--cache-expiration",
        "168h0m0s",
        "--no-mmap",
        "--json",
    ]

    ctx_size = find_parameter(model.backend_parameters, ["ctx-size", "c"])
    if ctx_size is None:
        ctx_size = "8192"

    execuable_command.extend(["--ctx-size", ctx_size])

    add_bool_flag(
        model.backend_parameters,
        ["image-no-text-encoder-model-offload"],
        execuable_command,
        "--image-no-text-encoder-model-offload",
    )
    add_bool_flag(
        model.backend_parameters,
        ["image-no-vae-model-offload"],
        execuable_command,
        "--image-no-vae-model-offload",
    )
    add_bool_flag(
        model.backend_parameters,
        ["image-no-vae-tiling"],
        execuable_command,
        "--image-no-vae-tiling",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["image-max-height"],
        execuable_command,
        "--image-max-height",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["image-max-width"],
        execuable_command,
        "--image-max-width",
    )
    add_parameter_with_value(
        model.backend_parameters,
        ["visual-max-image-size"],
        execuable_command,
        "--visual-max-image-size",
    )

    cache_dir = kwargs.get("cache_dir")
    if cache_dir:
        execuable_command.extend(["--cache-path", cache_dir])

    if offload == GPUOffloadEnum.Full:
        execuable_command.extend(["--gpu-layers", "-1"])
    elif offload == GPUOffloadEnum.Partial:
        execuable_command.extend(["--gpu-layers-step", "1"])
    elif offload == GPUOffloadEnum.Disable:
        execuable_command.extend(["--gpu-layers", "0"])

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
        execuable_command.extend(["--tensor-split", tensor_split_str])

    rpc = kwargs.get("rpc")
    if rpc:
        rpc_str = ",".join([v for v in rpc])
        execuable_command.extend(["--rpc", rpc_str])

    source_args = await _gguf_parser_command_args_from_source(model, **kwargs)
    execuable_command.extend(source_args)
    return execuable_command


async def calculate_model_resource_claim(
    model_instance: ModelInstance,
    model: Model,
    offload: GPUOffloadEnum = GPUOffloadEnum.Full,
    **kwargs,
) -> ModelInstanceResourceClaim:
    """
    Calculate the resource claim of the model instance.
    Args:
        model_instance: Model instance to calculate the resource claim for.
        model: Model to calculate the resource claim for.
    """

    if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(model.local_path):
        # Skip the calculation if the model is not available, policies like spread strategy still apply.
        # TODO Support user provided resource claim for better scheduling.
        estimate = _get_empty_estimate()
        tensor_split = kwargs.get("tensor_split")
        if tensor_split:
            estimate = _get_empty_estimate(n_gpu=len(tensor_split))
        return ModelInstanceResourceClaim(model_instance, estimate)

    logger.info(f"Calculating resource claim for model instance {model_instance.name}")

    command = await _gguf_parser_command(model, offload, **kwargs)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )

        cmd_output = stdout.decode()
        claim: modelResoruceClaim = modelResoruceClaim.from_json(cmd_output)

        if offload == GPUOffloadEnum.Full:
            logger.info(
                f"Calculated resource claim for full offload model instance {model_instance.name}, "
                f"claim: {claim.estimate.items[0]}"
            )
        elif offload == GPUOffloadEnum.Partial:
            logger.info(
                f"Calculated resource claim for partial offloading model instance {model_instance.name}, "
                f"least claim: {claim.estimate.items[1]}, "
                f"most claim: {claim.estimate.items[len(claim.estimate.items) - 2]}"
            )
        elif offload == GPUOffloadEnum.Disable:
            logger.info(
                f"Calculated resource claim for disabled offloading model instance {model_instance.name}, "
                f"claim: {claim.estimate.items[0]}"
            )
            clear_vram_claim(claim)

        return ModelInstanceResourceClaim(model_instance, claim.estimate)

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to execution {command}, error: {e}, ",
            f"stderr: {e.stderr.decode()}, ",
            f"stdout: {e.stdout.decode()}",
        )
    except Exception as e:
        raise Exception(
            f"Failed to parse the output of {command}, error: {e}",
        )


def clear_vram_claim(claim: modelResoruceClaim):
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
            args = ["-hf-repo", model.huggingface_repo_id]
            if model.huggingface_filename:
                model_filename = await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_model_filename,
                        model.huggingface_repo_id,
                        model.huggingface_filename,
                    ),
                    timeout=fetch_file_timeout_in_seconds,
                )
                args.extend(["-hf-file", model_filename])

                mmproj_filename = await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_mmproj_filename,
                        model,
                    ),
                    timeout=fetch_file_timeout_in_seconds,
                )
                if mmproj_filename:
                    args.extend(["--hf-mmproj-file", mmproj_filename])

            global_config = get_global_config()
            if global_config.huggingface_token:
                args.extend(["-hf-token", global_config.huggingface_token])

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


def hf_model_filename(repo_id: str, filename: Optional[str] = None) -> Optional[str]:
    if filename is None:
        return None
    else:
        matching_files = match_hugging_face_files(repo_id, filename)
        if len(matching_files) == 0:
            raise ValueError(f"File {filename} not found in {repo_id}")

        return matching_files[0]


def hf_mmproj_filename(model: Model) -> Optional[str]:
    mmproj_filename = get_mmproj_filename(model)
    matching_files = match_hugging_face_files(
        model.huggingface_repo_id, mmproj_filename
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
