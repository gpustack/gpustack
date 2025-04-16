import logging
from typing import List, Optional
from pathlib import Path
import fnmatch
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id
from modelscope.hub.api import HubApi
from transformers import PretrainedConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from requests.exceptions import HTTPError

from gpustack.config.config import get_global_config
from gpustack.schemas.models import Model, SourceEnum

logger = logging.getLogger(__name__)


MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN = [
    '*.json',
    '*.py',
]


def match_hugging_face_files(
    repo_id: str,
    filename: str,
    extra_filename: Optional[str] = None,
    token: Optional[str] = None,
) -> List[str]:
    validate_repo_id(repo_id)

    hffs = HfFileSystem(token=token)

    files = [
        file["name"] if isinstance(file, dict) else file
        for file in hffs.ls(repo_id, recursive=True)
    ]

    file_list: List[str] = []
    for file in files:
        rel_path = Path(file).relative_to(repo_id)
        file_list.append(rel_path.as_posix())

    matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore
    matching_files = sorted(matching_files)

    if extra_filename is None:
        return matching_files

    extra_matching_files = [
        file for file in file_list if fnmatch.fnmatch(file, extra_filename)
    ]
    extra_matching_files = sorted(extra_matching_files, reverse=True)
    if extra_matching_files:
        # Add the first element of the extra matching files to the matching files
        # For example, when matches f16 and f32 mmproj files, prefer f32 over f16
        matching_files.append(extra_matching_files[0])

    return matching_files


def match_model_scope_file_paths(
    model_id: str, file_path: str, extra_file_path: Optional[str] = None
) -> List[str]:
    if '/' in file_path:
        root, _ = file_path.rsplit('/', 1)
    else:
        root = None

    api = HubApi()
    files = api.get_model_files(model_id, root=root, recursive=True)

    file_paths = [file["Path"] for file in files]
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]
    matching_paths = sorted(matching_paths)

    if extra_file_path is None:
        return matching_paths

    extra_matching_paths = [
        p for p in file_paths if fnmatch.fnmatch(p, extra_file_path)
    ]
    extra_matching_paths = sorted(extra_matching_paths, reverse=True)
    if extra_matching_paths:
        # Add the first element of the extra matching paths to the matching paths
        # For example, when matches f16 and f32 mmproj files, prefer f32 over f16
        matching_paths.append(extra_matching_paths[0])

    return matching_paths


def get_model_weight_size(model: Model, token: Optional[str] = None) -> int:
    """
    Get the size of the model weights. This is the sum of all the weight files with extensions
    .safetensors, .bin, .pt, .pth.
    Args:
        model: Model to get the weight size for
        token: Optional Hugging Face API token
    Returns:
        int: The size of the model weights
    """
    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    if model.source == SourceEnum.HUGGING_FACE:
        api = HfApi(token=token)
        repo_info = api.repo_info(model.huggingface_repo_id, files_metadata=True)
        total_size = sum(
            sibling.size
            for sibling in repo_info.siblings
            if sibling.size is not None
            and sibling.rfilename.endswith(weight_file_extensions)
        )
        return total_size
    elif model.source == SourceEnum.MODEL_SCOPE:
        api = HubApi()
        files = api.get_model_files(model.model_scope_model_id, recursive=True)

        return sum(
            file["Size"]
            for file in files
            if file["Name"].endswith(weight_file_extensions)
        )


def get_pretrained_config(model: Model, **kwargs):
    """
    Get the pretrained config of the model from Hugging Face or ModelScope.
    Args:
        model: Model to get the pretrained config for.
    """

    trust_remote_code = False
    if (
        model.backend_parameters and "--trust-remote-code" in model.backend_parameters
    ) or kwargs.get("trust_remote_code"):
        trust_remote_code = True

    global_config = get_global_config()
    pretrained_config = None
    if model.source == SourceEnum.HUGGING_FACE:
        from transformers import AutoConfig

        pretrained_config = AutoConfig.from_pretrained(
            model.huggingface_repo_id,
            token=global_config.huggingface_token,
            trust_remote_code=trust_remote_code,
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        from modelscope import AutoConfig, snapshot_download

        try:
            # Download first then load config locally.
            # A temporary workaround for the issue:
            # https://github.com/modelscope/modelscope/issues/1302
            config_dir = snapshot_download(
                model.model_scope_model_id,
                allow_file_pattern=MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN,
            )
            pretrained_config = AutoConfig.from_pretrained(
                config_dir,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            if config_dir in str(e):
                # Make the message not confusing.
                raise ValueError(str(e).replace(config_dir, model.model_scope_model_id))
            else:
                raise e
    elif model.source == SourceEnum.LOCAL_PATH:
        from transformers import AutoConfig

        pretrained_config = AutoConfig.from_pretrained(
            model.local_path,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Unsupported model source: {model.source}")

    return pretrained_config


# Simplified from vllm.config._get_and_verify_max_len
# Keep in our codebase to avoid dependency on vllm's internal
# APIs which may change unexpectedly.
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/config.py#L2453
def get_max_model_len(pretrained_config) -> int:  # noqa: C901
    """Get the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(pretrained_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(pretrained_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError("rope_scaling must have a 'type' or 'rope_type' key.")

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type not in ("su", "longrope", "llama3"):
            scaling_factor = 1
            if "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    logger.debug(f"Derived max model length: {derived_max_model_len}")
    return int(derived_max_model_len)


# Similar to https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/transformers_utils/config.py#L700,
# But we don't assert and fail if num_attention_heads is missing.
def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "num_attention_heads"
    ):
        return config.text_config
    else:
        return config


quantization_list = [
    "-IQ1_",
    "-IQ2_",
    "-IQ3_",
    "-IQ4_",
    "-Q2_",
    "-Q3_",
    "-Q4_",
    "-Q5_",
    "-Q6_",
    "-Q8_",
]


def get_hugging_face_model_min_gguf_path(
    model_id: str,
    token: Optional[str] = None,
) -> Optional[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(model_id)

    gguf_files = sorted([f for f in files if f.endswith(".gguf")])
    if not gguf_files:
        return None

    for quantization in quantization_list:
        for gguf_file in gguf_files:
            if quantization in gguf_file.upper():
                return gguf_file

    return gguf_files[0]


def auth_check(
    model: Model,
    huggingface_token: Optional[str] = None,
):
    if model.source == SourceEnum.HUGGING_FACE:
        api = HfApi(token=huggingface_token)
        try:
            api.auth_check(model.huggingface_repo_id)
        except GatedRepoError:
            raise Exception(
                "Access to the model is restricted. Please set a valid Huggingface token with proper permissions in the GPUStack server configuration."
            )
        except HfHubHTTPError as e:
            if e.response.status_code in [401, 403]:
                raise Exception(
                    "Access to the model is restricted. Please set a valid Huggingface token with proper permissions in the GPUStack server configuration."
                )
    if model.source == SourceEnum.MODEL_SCOPE:
        api = HubApi()
        try:
            api.get_model_files(model.model_scope_model_id)
        except HTTPError as e:
            if e.response.status_code in [401, 403, 404]:
                raise Exception("Access to the model is restricted.")


def get_model_scope_model_min_gguf_path(
    model_id: str,
) -> Optional[str]:
    api = HubApi()
    files = api.get_model_files(model_id, recursive=True)
    file_paths: List[str] = [file["Path"] for file in files]

    gguf_files = sorted([f for f in file_paths if f.endswith(".gguf")])
    if not gguf_files:
        return None

    for quantization in quantization_list:
        for gguf_file in gguf_files:
            if quantization in gguf_file.upper():
                return gguf_file

    return gguf_files[0]
