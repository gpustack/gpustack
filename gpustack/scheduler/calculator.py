import asyncio
import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json
import platform


from gpustack.schemas.models import Model, ModelInstance, SourceEnum
from gpustack.utils.command import get_platform_command
from gpustack.utils.compat_importlib import pkg_resources


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class memoryResource:
    ram: int
    vram: int


@dataclass_json
@dataclass
class memoryEstimate:
    offloadLayers: int
    uma: memoryResource
    nonUMA: memoryResource


@dataclass_json
@dataclass
class estimate:
    memory: List[memoryEstimate]
    contextSize: int
    architecture: str
    embeddingOnly: bool


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


def _gguf_parser_command(model: Model, **kwargs):
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
        "-ctx-size",
        "8192",
        "-in-max-ctx-size",
        "-gpu-layers-step",
        "1",
        "-skip-tokenizer",
        "-skip-architecture",
        "-skip-model",
        "-json",
    ]

    source_args = _gguf_parser_command_args_from_source(model, **kwargs)
    execuable_command.extend(source_args)
    return execuable_command


async def calculate_model_resource_claim(
    model_instance: ModelInstance, model: Model, **kwargs
) -> ModelInstanceResourceClaim:
    """
    Calculate the resource claim of the model instance.
    Args:
        model_instance: Model instance to calculate the resource claim for.
        model: Model to calculate the resource claim for.
    """

    logger.info(f"Calculating resource claim for model instance {model_instance.name}")

    command = _gguf_parser_command(model, **kwargs)
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

        logger.info(
            f"Calculated resource claim for model instance {model_instance.name}, "
            f"least: {claim.estimate.memory[0]}, "
            f"most: {claim.estimate.memory[len(claim.estimate.memory) - 1]}"
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
        return f"{_registry_url}/{repo_id}/resolve/main/{filename}"
