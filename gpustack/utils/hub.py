import json
import logging
import gzip
import os
import tempfile
from typing import Dict, List, Optional
from pathlib import Path
import fnmatch
from threading import Lock
from functools import cache
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import (
    snapshot_download as modelscope_snapshot_download,
)
from transformers import PretrainedConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from requests.exceptions import HTTPError

from gpustack.config.config import get_global_config
from gpustack.schemas import ModelFile
from gpustack.schemas.models import CategoryEnum, Model, SourceEnum, get_mmproj_filename
from gpustack.utils.cache import is_cached, load_cache, save_cache

logger = logging.getLogger(__name__)

LIST_REPO_CACHE_DIR = "repo-skeleton"

MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN = [
    '*.json',
    '*.py',
]


@cache
def get_model_lock(model_id: str) -> Lock:
    """Get or create a lock for the given model_id. The model_id is used as the key to store Lock in cache."""
    return Lock()


class FileEntry:
    def __init__(self, rfilename: str, size: Optional[int] = None):
        self.rfilename = rfilename
        self.size = size


def get_model_path_and_name(model: ModelFile) -> (str, str):
    if model.source == SourceEnum.HUGGING_FACE:
        return model.huggingface_repo_id, model.huggingface_filename
    elif model.source == SourceEnum.MODEL_SCOPE:
        return model.model_scope_model_id, model.model_scope_file_path
    elif model.source == SourceEnum.LOCAL_PATH:
        return model.local_path, ""
    else:
        return "", ""


def match_file_and_calculate_size(
    files: List[FileEntry],
    model: ModelFile,
    cache_dir: str,
) -> (int, List[str]):
    """
    Match the files and calculate the total size.
    Also return the selected files.
    """

    selected_files = []
    match_files = []
    extra_files = []

    file_path, filename = get_model_path_and_name(model)
    extra_filename = get_mmproj_filename(model)

    if file_path and not filename:
        base_dir = model.local_dir or f"{cache_dir}/{model.source.value}/{file_path}"
        return (
            sum(f.size for f in files if getattr(f, 'size', None) is not None),
            [base_dir],
        )

    for sibling in files:
        if sibling.size is None:
            continue

        rfilename = sibling.rfilename

        if filename and fnmatch.fnmatch(rfilename, filename):
            selected_files.append(rfilename)
            match_files.append(sibling)
        elif extra_filename and fnmatch.fnmatch(rfilename, extra_filename):
            extra_files.append(rfilename)
            match_files.append(sibling)

    best_extra = select_most_suitable_extra_file(extra_files)
    if best_extra:
        selected_files.append(best_extra)

    sum_size = sum(
        f.size
        for f in match_files
        if getattr(f, 'rfilename', '') in selected_files
        and getattr(f, 'size', None) is not None
    )

    if selected_files and model.source in [
        SourceEnum.HUGGING_FACE,
        SourceEnum.MODEL_SCOPE,
    ]:
        base_dir = model.local_dir or f"{cache_dir}/{model.source.value}/{file_path}"
        selected_files = [os.path.join(base_dir, f) for f in selected_files]

    return sum_size, selected_files


def select_most_suitable_extra_file(file_list: List[str]) -> str:
    """
    Select the most suitable extra file from the list of files.
    For example, when matches f16 and f32 mmproj files, prefer f32 over f16
    """
    if not file_list or len(file_list) == 0:
        return ""
    _file_list = sorted(file_list, reverse=True)
    return _file_list[0]


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
    extra_file = select_most_suitable_extra_file(extra_matching_files)
    if extra_file:
        matching_files.append(extra_file)

    return matching_files


def is_repo_cached(repo_id: str, source: str) -> bool:
    if not repo_id or not source:
        return False
    cache_key = f"{source}:{repo_id}"
    return is_cached(LIST_REPO_CACHE_DIR, cache_key)


def list_repo(
    repo_id: str,
    source: str,
    token: Optional[str] = None,
    cache_expiration: Optional[int] = None,
    root_dir_only: bool = False,
) -> List[Dict[str, any]]:
    cache_key = f"{source}:{repo_id}:{root_dir_only}"
    cached_result, is_succ = load_cache(
        LIST_REPO_CACHE_DIR, cache_key, cache_expiration
    )
    if is_succ:
        result = json.loads(cached_result)
        if isinstance(result, list):
            return result

    if source == SourceEnum.HUGGING_FACE:
        validate_repo_id(repo_id)
        hffs = HfFileSystem(token=token)
        file_info = []
        for file in hffs.ls(repo_id, recursive=not root_dir_only):
            if not isinstance(file, dict):
                continue
            relative_path = Path(file["name"]).relative_to(repo_id).as_posix()
            # If root_only is True, skip files in subdirectories
            if root_dir_only and "/" in relative_path:
                continue
            file_info.append(
                {
                    "name": relative_path,
                    "size": file["size"],
                }
            )
    elif source == SourceEnum.MODEL_SCOPE:
        msapi = HubApi()
        files = msapi.get_model_files(repo_id, recursive=not root_dir_only)
        file_info = []
        for file in files:
            file_path = file["Path"]
            # If root_only is True, skip files in subdirectories
            if root_dir_only and "/" in file_path:
                continue
            file_info.append(
                {
                    "name": file_path,
                    "size": file["Size"],
                }
            )
    else:
        raise ValueError(f"Invalid source: {source}")

    if not save_cache(LIST_REPO_CACHE_DIR, cache_key, json.dumps(file_info)):
        logger.info(f"Saved cache {LIST_REPO_CACHE_DIR} {cache_key} fail")

    return file_info


def filter_filename(file_path: str, file_paths: List[str]):
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]
    matching_paths = sorted(matching_paths)

    return matching_paths


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


def read_repo_file_content(
    model: Model,
    file_path: str,
    token: Optional[str] = None,
) -> Optional[bytes]:
    """
    Read a file's raw bytes from the model's source.

    - Hugging Face: uses HfFileSystem to open `{repo_id}/{file_path}`.
    - ModelScope: downloads a snapshot matching `file_path` and cleaned automatically after reading locally.
    - Local Path: reads from the local directory.

    Returns None if the file cannot be found or read.
    """
    try:
        if model.source == SourceEnum.HUGGING_FACE:
            hffs = HfFileSystem(token=token)
            repo_path = f"{model.huggingface_repo_id}/{file_path}"
            with hffs.open(repo_path, "rb") as f:
                content = f.read()
                if (
                    content
                    and content.startswith(b"\x1f\x8b")
                    and not file_path.endswith(".gz")
                ):
                    try:
                        content = gzip.decompress(content)
                    except Exception as e:
                        logger.warning(
                            f"Failed to decompress gzip content for {file_path}: {e}"
                        )
                return content

        elif model.source == SourceEnum.MODEL_SCOPE:
            _cfg = get_global_config()
            base_tmp = os.path.join(
                (_cfg.cache_dir if _cfg and _cfg.cache_dir else "/tmp"),
                "modelscope",
                "tempfile",
            )
            os.makedirs(base_tmp, exist_ok=True)
            safe_id = (model.model_scope_model_id or "").replace("/", "__")
            with tempfile.TemporaryDirectory(
                dir=base_tmp, prefix=f"{safe_id}__"
            ) as tmp_dir:
                model_dir = modelscope_snapshot_download(
                    model_id=model.model_scope_model_id,
                    local_dir=tmp_dir,
                    allow_patterns=[file_path],
                )

                candidate = os.path.join(model_dir, file_path)
                fp = candidate if os.path.exists(candidate) else None
                if not fp:
                    # Search recursively by base filename for robustness
                    base_name = os.path.basename(file_path)
                    for root, _dirs, files in os.walk(model_dir):
                        if base_name in files:
                            fp = os.path.join(root, base_name)
                            break
                if not fp:
                    return None
                with open(fp, "rb") as f:
                    return f.read()

        elif model.source == SourceEnum.LOCAL_PATH:
            local_path = model.local_path or ""
            if not local_path or not os.path.isdir(local_path):
                return None
            fp = os.path.join(local_path, file_path)
            if not os.path.exists(fp):
                return None
            with open(fp, "rb") as f:
                return f.read()

        else:
            return None
    except Exception as e:
        source_key = (
            model.huggingface_repo_id
            or model.model_scope_model_id
            or model.local_path
            or "<unknown>"
        )
        logger.error(f"Failed to read '{file_path}' for source '{source_key}': {e}")
        return None


def get_model_weight_size(model: Model, token: Optional[str] = None) -> int:
    """
    Get the size of the model weights. This is the sum of all the weight files with extensions
    .safetensors, .bin, .pt, .pth in the root directory only.
    Args:
        model: Model to get the weight size for
        token: Optional Hugging Face API token
    Returns:
        int: The size of the model weights
    """
    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    if model.source == SourceEnum.HUGGING_FACE:
        repo_id = model.huggingface_repo_id
    elif model.source == SourceEnum.MODEL_SCOPE:
        repo_id = model.model_scope_model_id
    else:
        raise ValueError(f"Unknown source {model.source}")
    repo_file_infos = list_repo(repo_id, model.source, token=token, root_dir_only=True)
    return sum(
        file.get("size", 0)
        for file in repo_file_infos
        if file.get("name", "").endswith(weight_file_extensions)
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
            cache_dir=os.path.join(global_config.cache_dir, "huggingface"),
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        from modelscope import AutoConfig

        model_scope_cache_dir = os.path.join(global_config.cache_dir, "model_scope")
        repo_cache_dir = os.path.join(
            model_scope_cache_dir, *model.model_scope_model_id.split('/')
        )
        local_files_only = False
        pretrained_model_name_or_path = model.model_scope_model_id
        if os.path.exists(repo_cache_dir):
            local_files_only = True
            pretrained_model_name_or_path = repo_cache_dir
        with get_model_lock(model.model_scope_model_id):
            pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                allow_file_pattern=MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN,
                cache_dir=model_scope_cache_dir,
                local_files_only=local_files_only,
            )
    elif model.source == SourceEnum.LOCAL_PATH:
        if not os.path.exists(model.local_path):
            logger.warning(
                f"Local Path: {model.readable_source} is not local to the server node and may reside on a worker node."
            )
            # Return an empty dict here to facilitate special handling by upstream methods.
            return {}

        from transformers import AutoConfig

        pretrained_config = AutoConfig.from_pretrained(
            model.local_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    else:
        raise ValueError(f"Unsupported model source: {model.source}")

    return pretrained_config


def get_pretrained_config_with_fallback(model: Model, **kwargs):
    pretrained_config = None
    try:
        pretrained_config = get_pretrained_config(model, **kwargs)
    except Exception as e:
        logger.debug(
            "Fallback to load config.json after AutoConfig.from_pretrained failed"
        )

        if model.backend_version is not None or isinstance(e, ImportError):
            # Fallback:
            # AutoConfig.from_pretrained performs strict architecture validation and may fail in several cases, like:
            #   1. Models using custom or backend-specific architectures not recognized by the current Transformers version.
            #   2. Newly released models whose architectures are not yet supported in older AutoConfig implementations.
            #   3. Import-time failures caused by missing or conflicting dependencies
            #      (e.g., LlamaFlashAttention2 import errors — see: https://github.com/deepseek-ai/DeepSeek-OCR/issues/7).
            # In all such cases, fallback to loading config.json directly to avoid blocking model startup.
            try:
                # try to read config.json and ensure num_attention_heads not None.
                config_content = read_repo_file_content(
                    model,
                    "config.json",
                    token=get_global_config().huggingface_token,
                )
                if config_content:
                    try:
                        try:
                            content = (config_content or b"").decode("utf-8")
                        except Exception:
                            content = (config_content or b"").decode()
                        config_dict = json.loads(content)
                        pretrained_config = PretrainedConfig.from_dict(config_dict)
                    except Exception as ce:
                        logger.warning(f"read_repo_file_content failed: {ce}")
            except Exception as ce:
                logger.warning(f"Fallback to load config.json failed: {ce}")

        if (
            pretrained_config is None
            and CategoryEnum.LLM in model.categories
            and (not model.env or not model.env.get("GPUSTACK_SKIP_MODEL_EVALUATION"))
        ):
            # For LLM models: empty config is unacceptable → raise original error
            raise e

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


def has_diffusers_model_index(  # noqa: C901
    model: Model, token: Optional[str] = None
) -> bool:
    """Check whether the model source contains a model_index.json with
    the key "_diffusers_version".

    Supported sources:
    - Hugging Face: checks via HfFileSystem
    - ModelScope: downloads only model_index.json via snapshot_download and inspects
    - Local Path: reads model_index.json in the provided directory
    """
    try:
        content_bytes: Optional[bytes] = None

        # Read model_index.json content based on model source
        content_bytes = read_repo_file_content(model, "model_index.json", token=token)
        if content_bytes is None:
            return False

        # Decode and parse JSON
        try:
            content = (content_bytes or b"").decode("utf-8")
        except Exception:
            content = (content_bytes or b"").decode()

        try:
            data = json.loads(content)
        except Exception:
            return False

        # The typical structure is a dict containing _diffusers_version
        if isinstance(data, dict) and "_diffusers_version" in data:
            return True
        # Some repos might have a list structure; check items for the key
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "_diffusers_version" in item:
                    return True

        return False
    except Exception as e:
        # Best-effort detection; do not raise on error
        try:
            source_key = (
                model.huggingface_repo_id
                or model.model_scope_model_id
                or model.local_path
                or "<unknown>"
            )
            logger.error(f"Failed to check model_index.json for {source_key}: {e}")
        except Exception:
            pass
        return False
