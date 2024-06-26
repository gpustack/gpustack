import subprocess
from dataclasses import dataclass
from typing import List
from dataclasses_json import dataclass_json
import platform
import importlib.resources as pkg_resources


from gpustack.schemas.models import Model, ModelInstance, SourceEnum
from gpustack.worker.downloaders import OllamaLibraryDownloader, HfDownloader


@dataclass_json
@dataclass
class nonUMA:
    ram: int
    vram: int


@dataclass_json
@dataclass
class memoryEstimate:
    offloadLayers: int
    uma: int
    nonUMA: nonUMA


@dataclass_json
@dataclass
class estimate:
    memory: List[memoryEstimate]
    contextSize: int
    architecture: str


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


def _gguf_parser_command(model_url):
    command = ""
    command_path = "gpustack.third_party.gguf-parser"
    match platform.system():
        case "Darwin":
            command = "gguf-parser-darwin-universal"
        case "Linux":
            if "amd64" in platform.machine() or "x86_64" in platform.machine():
                command = "gguf-parser-linux-amd64"
            elif "arm" in platform.machine() or "aarch64" in platform.machine():
                command = "gguf-parser-linux-arm64"

    command_path = pkg_resources.files("gpustack.third_party.gguf-parser").joinpath(
        command
    )
    execuable_command = [
        command_path,
        "-url",
        model_url,
        "-ctx-size",
        "-1",
        "-flash-attention",
        "-offload-layers-step",
        "1",
        "-skip-tokenizer",
        "-skip-architecture",
        "-skip-model",
        "-json",
    ]
    return execuable_command


async def calculate_model_resource_claim(
    model_instance: ModelInstance, model: Model
) -> ModelInstanceResourceClaim:
    """
    Calculate the resource claim of the model instance.
    Args:
        model_instance: Model instance to calculate the resource claim for.
        model: Model to calculate the resource claim for.
    """

    model_url = get_model_url(model)
    command = _gguf_parser_command(model_url)
    try:
        cmd_result = subprocess.run(command, capture_output=True, text=True, check=True)
        cmd_output = cmd_result.stdout

        claim = modelResoruceClaim.from_json(cmd_output)
        return ModelInstanceResourceClaim(model_instance, claim.estimate)
    except subprocess.CalledProcessError as e:
        e.add_note(command.__str__() + "execution failed")
        raise
    except Exception as e:
        e.add_note(
            "error occurred when trying execute and parse the output of "
            + command.__str__()
        )
        raise e


# arr is a sorted list from smallest to largest
def binary_search(arr, target):
    """
    Binary search the target in the arr.
    """
    if len(arr) == 0:
        return -1

    if arr[0] > target:
        return -1

    if arr[-1] < target:
        return len(arr) - 1

    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return high


def get_model_url(model: Model) -> str:
    """
    Get the model url based on the model source.
    Args:
        model: Model to get the url for.
    """
    if model.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.model_url(
            repo_id=model.huggingface_repo_id, filename=model.huggingface_filename
        )
    elif model.source == SourceEnum.OLLAMA_LIBRARY:
        return OllamaLibraryDownloader.model_url(
            model_name=model.ollama_library_model_name
        )
