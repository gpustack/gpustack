import asyncio
from enum import Enum
import logging
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json
import platform


from gpustack.schemas.models import Model, ModelInstance, SourceEnum
from gpustack.utils.command import get_platform_command
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.hugging_face import match_hf_files


logger = logging.getLogger(__name__)


class GPUOffloadEnum(str, Enum):
    Full = "full"
    Partial = "partial"
    Disable = "disable"


@dataclass_json
@dataclass
class layerMemoryEstimate:
    uma: int
    nonuma: int
    handleLayers: Optional[int]


@dataclass_json
@dataclass
class memoryEstimate:
    offloadLayers: int
    fullOffloaded: bool
    ram: layerMemoryEstimate
    vrams: List[layerMemoryEstimate]


@dataclass_json
@dataclass
class estimate:
    items: List[memoryEstimate]
    contextSize: int
    architecture: str
    embeddingOnly: bool
    distributable: bool


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


def _gguf_parser_command(
    model: Model, offload: GPUOffloadEnum = GPUOffloadEnum.Full, **kwargs
):
    command_map = {
        ("Windows", "amd64"): "gguf-parser-windows-amd64.exe",
        ("Darwin", "amd64"): "gguf-parser-darwin-universal",
        ("Darwin", "arm64"): "gguf-parser-darwin-universal",
        ("Linux", "amd64"): "gguf-parser-linux-amd64",
        ("Linux", "arm64"): "gguf-parser-linux-arm64",
    }

    command = get_platform_command(command_map)
    if command == "":
        raise Exception(
            f"No supported gguf-parser command found for "
            f"{platform.system()} {platform.machine()}."
        )

    command_path = pkg_resources.files("gpustack.third_party.bin.gguf-parser").joinpath(
        command
    )
    execuable_command = [
        command_path,
        "--ctx-size",
        "8192",
        "--in-max-ctx-size",
        "--skip-tokenizer",
        "--skip-architecture",
        "--skip-metadata",
        "--cache-expiration",
        "168h0m0s",
        "--no-mmap",
        "--json",
    ]

    cache_dir = kwargs.get("cache_dir")
    if cache_dir:
        execuable_command.append("--cache-path")
        execuable_command.append(cache_dir)

    if offload == GPUOffloadEnum.Full:
        execuable_command.append("--gpu-layers")
        execuable_command.append("-1")
    elif offload == GPUOffloadEnum.Partial:
        execuable_command.append("--gpu-layers-step")
        execuable_command.append("1")
    elif offload == GPUOffloadEnum.Disable:
        execuable_command.append("--gpu-layers")
        execuable_command.append("0")

    tensor_split = kwargs.get("tensor_split")
    if tensor_split:
        tensor_split_str = ",".join(
            [str(int(i / (1024 * 1024))) for i in tensor_split]
        )  # convert to MiB to prevent overflow
        execuable_command.append("--tensor-split")
        execuable_command.append(tensor_split_str)

    rpc = kwargs.get("rpc")
    if rpc:
        rpc_str = ",".join([v for v in rpc])
        execuable_command.append("--rpc")
        execuable_command.append(rpc_str)

    source_args = _gguf_parser_command_args_from_source(model, **kwargs)
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

    logger.info(f"Calculating resource claim for model instance {model_instance.name}")

    command = _gguf_parser_command(model, offload, **kwargs)
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )

        cmd_output = stdout.decode()
        claim = modelResoruceClaim.from_json(cmd_output)

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

        return ModelInstanceResourceClaim(model_instance, claim.estimate)

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to execution {command.__str__()}, " f"error: {e}, ",
            f"stderr: {e.stderr.decode()}, ",
            f"stdout: {e.stdout.decode()}",
        )
    except Exception as e:
        raise Exception(
            f"Failed to parse the output of {command.__str__()}, " f"error: {e}, ",
            f"stderr: {e.stderr.decode()}, ",
            f"stdout: {e.stdout.decode()}",
        )


def _gguf_parser_command_args_from_source(model: Model, **kwargs) -> List[str]:
    """
    Get the model url based on the model source.
    Args:
        model: Model to get the url for.
    """
    if model.source == SourceEnum.HUGGING_FACE:
        model_url = hf_model_url(
            repo_id=model.huggingface_repo_id, filename=model.huggingface_filename
        )

        return ["-url", model_url]
    elif model.source == SourceEnum.OLLAMA_LIBRARY:
        args = ["-ol-model", model.ollama_library_model_name]
        ol_base_url = kwargs.get("ollama_library_base_url")
        if ol_base_url:
            args.extend(["-ol-base-url", ol_base_url])
        return args
    else:
        raise ValueError(f"Unsupported source: {model.source}")


def hf_model_url(repo_id: str, filename: Optional[str] = None) -> str:
    _registry_url = "https://huggingface.co"
    if filename is None:
        return f"{_registry_url}/{repo_id}"
    else:
        matching_files = match_hf_files(repo_id, filename)
        if len(matching_files) == 0:
            raise ValueError(f"File {filename} not found in {repo_id}")

        filename = Path(matching_files[0]).name
        return f"{_registry_url}/{repo_id}/resolve/main/{filename}"
