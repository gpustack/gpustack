import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
import fnmatch
from threading import Lock
from functools import cache

import transformers.utils
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.utils import validate_repo_id
from modelscope import snapshot_download
from modelscope.hub.api import HubApi
from transformers import PretrainedConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from requests.exceptions import HTTPError

from enum import Enum

from gpustack.config.config import get_global_config
from gpustack.schemas.models import Model, SourceEnum, get_mmproj_filename
from gpustack.utils.cache import is_cached, load_cache, save_cache


class FileEntry:
    def __init__(self, rfilename: str, size: Optional[int] = None):
        self.rfilename = rfilename
        self.size = size

    def __eq__(self, other):
        if not isinstance(other, FileEntry):
            return False
        return self.rfilename == other.rfilename and self.size == other.size

    def __repr__(self):
        return f"FileEntry(rfilename='{self.rfilename}', size={self.size})"


class FilterPurpose(str, Enum):
    """Enum for model file filtering purpose."""

    EVALUATE = "evaluate"
    DOWNLOAD = "download"


logger = logging.getLogger(__name__)

LIST_REPO_CACHE_DIR = "repo-skeleton"

MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN = [
    '*.json',
    '*.py',
]

# Default format preference order (priority: first = highest)
DEFAULT_FORMAT_PREFERENCE = ['safetensors', 'bin', 'pt', 'pth', 'gguf']


class ModelFileFilter:
    def __init__(self, model: Model, purpose: FilterPurpose = FilterPurpose.EVALUATE):
        """
        Initialize the ModelFileFilter.
        Args:
            model: Model configuration object
            purpose: Purpose of filtering (FilterPurpose.EVALUATE or FilterPurpose.DOWNLOAD)
        """
        self.model = model
        self.purpose = purpose

        # Extract model preferences and repository information
        self.format_preference = self._get_format_preference()
        self.subdir_preference = self._get_subdir_preference()
        self.repo_id, self.source = self._extract_repo_info()

    def filter_files(self, repo_file_infos: List[FileEntry]) -> List[FileEntry]:
        """
        Filter model files based on the configured preferences.
        Args:
            repo_file_infos: List of FileEntry objects from repository
        Returns:
            List of filtered FileEntry objects
        """
        if not repo_file_infos:
            return []

        # Filter files by subdirectory preference if evaluating
        filtered_files = self._apply_subdir_filter(repo_file_infos)

        # Categorize files to a dict by format
        categorized_files = self._categorize_files_by_format(filtered_files)

        # Select files based on .index.json detection and preferences
        selected_files = self._select_files_with_index_detection(categorized_files)

        if not selected_files:
            raise ValueError(
                "No valid model files were found in the repository, please verify the repository structure, network connectivity, and environment variable configuration during deployment."
            )

        return selected_files

    def _categorize_files_by_format(
        self,
        files: List[FileEntry],
    ) -> Dict[str, List[FileEntry]]:
        """Categorize files by their format (safetensors, pytorch, etc.)."""

        # Use defaultdict(list) to automatically create empty list for new keys
        # This avoids manually checking if key exists before appending
        categorized = defaultdict(list)

        # Check if this is an index file for any format
        index_file_set = {}
        for format_ext in self.format_preference:
            index_file_set[self._get_index_filename(format_ext)] = format_ext

        for file in files:
            if not file.rfilename:
                continue

            extension = file.rfilename.split(".")[-1].lower()
            if (
                extension in self.format_preference
                or self.purpose == FilterPurpose.DOWNLOAD
            ):
                categorized[extension].append(file)
            elif file.rfilename in index_file_set.keys():
                categorized[index_file_set[file.rfilename]].append(file)
        return categorized

    def _get_format_preference(self) -> List[str]:
        """Get format preference from environment variables or use default."""
        # Check for user-defined preference
        if (
            hasattr(self.model, 'env')
            and self.model.env
            and "GPUSTACK_MODEL_FORMAT_PREF" in self.model.env
        ):
            pref_str = self.model.env["GPUSTACK_MODEL_FORMAT_PREF"]
            return [fmt.strip() for fmt in pref_str.split(",")]

        # Use default preference order
        return DEFAULT_FORMAT_PREFERENCE

    def _get_subdir_preference(self) -> bool:
        """
        Get subdirectory preference from model environment.

        Returns:
            bool: True to include subdirectories, False to only include root files
        """
        if (
            hasattr(self.model, 'env')
            and self.model.env
            and "GPUSTACK_MODEL_INCLUDE_SUBDIRS" in self.model.env
        ):
            return self.model.env["GPUSTACK_MODEL_INCLUDE_SUBDIRS"].lower() in [
                "true",
                "1",
            ]

        return False  # Default to root directory only

    def _extract_repo_info(self) -> tuple[Optional[str], Optional[SourceEnum]]:
        """Extract repository information for index file parsing."""
        if self.model.source == SourceEnum.HUGGING_FACE:
            return self.model.huggingface_repo_id, SourceEnum.HUGGING_FACE
        elif self.model.source == SourceEnum.MODEL_SCOPE:
            return self.model.model_scope_model_id, SourceEnum.MODEL_SCOPE
        return None, None

    def _apply_subdir_filter(self, repo_file_infos: List[FileEntry]) -> List[FileEntry]:
        """Apply subdirectory filtering based on purpose and preferences."""
        if self.purpose == FilterPurpose.EVALUATE and not self.subdir_preference:
            # Only include files in root directory
            return [f for f in repo_file_infos if '/' not in f.rfilename]
        return repo_file_infos

    def _select_files_with_index_detection(
        self, categorized_files: Dict[str, List[FileEntry]]
    ) -> List[FileEntry]:
        """
        Select files based on .index.json detection and format preferences.
        This method implements the core logic:
        - Look for .index.json files to determine if model is sharded
        - If sharded, parse index.json file to get exact shard filenames
        - If not sharded, select consolidated weight files
        - Apply format preferences

        Args:
            categorized_files: Files categorized by format
        Returns:
            List of selected FileEntry objects
        """
        if not categorized_files:
            return []

        selected_files = []

        # Try each format in preference order
        for format_ext, format_files in categorized_files.items():
            # Check if this format has an index file (indicating sharding)
            index_filename = self._get_index_filename(format_ext)
            has_index = any(f.rfilename == index_filename for f in format_files)

            if has_index:
                # Model is sharded, get files from index
                shard_files = self._get_sharded_files_from_index(
                    format_files, format_ext
                )
                selected_files.extend(shard_files)
            else:
                # Model is not sharded, look for consolidated files
                consolidated_files = self._get_consolidated_files(
                    format_files, format_ext
                )
                selected_files.extend(consolidated_files)

            # For evaluation, usually one format is enough to avoid redundancy
            if self.purpose == FilterPurpose.EVALUATE and selected_files:
                break

        return selected_files

    def _get_consolidated_files(
        self, files: List[FileEntry], format_ext: str
    ) -> List[FileEntry]:
        """Get consolidated (single) model files for a format."""
        consolidated_files = []

        # Define standard consolidated filenames for different formats
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/__init__.py#L291-L307
        standard_filenames = []
        if format_ext == 'safetensors':
            standard_filenames = ['model.safetensors']
        elif format_ext in ['bin', 'pt', 'pth']:
            standard_filenames = [f'pytorch_model.{format_ext}', f'model.{format_ext}']

        # Look for standard consolidated files in order of preference
        for f in files:
            if f.rfilename in standard_filenames:
                consolidated_files.append(f)
                break

        # If no standard consolidated file found, apply different strategies based on purpose
        if not consolidated_files:
            non_index_files = [
                f
                for f in files
                if f.rfilename.endswith(f'.{format_ext}')
                and not f.rfilename.endswith('.index.json')
            ]

            if self.purpose == FilterPurpose.DOWNLOAD:
                # For download, be more permissive - include all non-index files
                # This ensures we download potentially valuable model files even if they don't follow standard naming
                consolidated_files = non_index_files
            else:
                # For evaluation, be more conservative - only include the largest file
                # This helps ensure we get the main model file for inference
                if non_index_files:
                    # Sort by size (descending) and take the largest file
                    sorted_files = sorted(
                        non_index_files, key=lambda f: f.size or 0, reverse=True
                    )
                    consolidated_files = [sorted_files[0]]

        return consolidated_files

    def _get_sharded_files_from_index(
        self, files: List[FileEntry], format_ext: str
    ) -> List[FileEntry]:
        """
        Get list of sharded files by reading the index file content.

        This method parses the index.json file to get the exact list of shard files
        referenced in the weight_map, ensuring we only return files that are actually
        part of the sharded model.

        Args:
            files: List of FileEntry objects from repository
            format_ext: File extension to check

        Returns:
            List of sharded files for the format
        """
        # Find the index file
        index_filename = self._get_index_filename(format_ext)
        index_file = None

        for file_entry in files:
            if file_entry.rfilename == index_filename:
                index_file = file_entry
                break

        if not index_file:
            # Fallback: return regex-matched files
            return [f for f in files if self._is_sharded_file_name(f.rfilename)]

        # Try to get and parse the index file content if repo info is available
        if self.repo_id and self.source:
            index_content = self._get_index_file_content(
                self.repo_id, index_filename, self.source
            )
            if index_content:
                # Parse the index file to get exact shard filenames
                shard_file_set = self._parse_index_file_content(index_content)
                # Filter files to only include those referenced in weight_map
                sharded_files = [
                    file_entry
                    for file_entry in files
                    if file_entry.rfilename in shard_file_set
                ]
                return sharded_files

        # Fallback: return regex-matched filename
        # This happens when we can't download/parse the index file
        sharded_files = []
        for file_entry in files:
            filename = file_entry.rfilename
            if self._is_sharded_file_name(filename):
                sharded_files.append(file_entry)

        return sharded_files

    @staticmethod
    def _get_index_filename(format_ext: str) -> str:
        """Get the expected index filename for a given format."""
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/__init__.py#L291-L307
        format_to_index_name = {
            'safetensors': transformers.utils.SAFE_WEIGHTS_INDEX_NAME,
            'bin': transformers.utils.WEIGHTS_INDEX_NAME,
            'h5': transformers.utils.TF2_WEIGHTS_INDEX_NAME,
            'msgpack': transformers.utils.FLAX_WEIGHTS_INDEX_NAME,
        }
        return format_to_index_name.get(format_ext, f'model.{format_ext}.index.json')

    @staticmethod
    def _parse_index_file_content(index_content: str) -> set:
        """Parse index.json content to extract shard filenames from weight_map.

        Args:
            index_content: JSON content of the index file

        Returns:
            Set of unique shard filenames referenced in weight_map
        """
        try:
            index_data = json.loads(index_content)
            weight_map = index_data.get('weight_map', {})

            # Extract unique shard filenames from weight_map values
            shard_filenames = set(weight_map.values())
            return shard_filenames

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse index file content: {e}")
            return set()

    @staticmethod
    def _is_sharded_file_name(filename: str) -> bool:
        """Check if filename matches sharded file pattern."""
        # Standard xx-of-xx format (model-00001-of-00002.safetensors)
        match = re.search(r'^(.+?)[-_]\d+[-_]of[-_]\d+', filename)
        return match is not None

    @staticmethod
    def _get_index_file_content(
        repo_id: str, index_filename: str, source: SourceEnum
    ) -> Optional[str]:
        """Get the content of an index file from the repository.

        This function downloads and reads index files (like model.safetensors.index.json)
        from model repositories with proper error handling, timeout, and retry mechanisms.

        Args:
            repo_id: Repository ID
            index_filename: Name of the index file
            source: Source type (HuggingFace or ModelScope)

        Returns:
            Content of the index file as string, or None if failed
        """

        def _download_huggingface_index():
            global_config = get_global_config()
            token = (
                global_config.huggingface_token
                if hasattr(global_config, 'huggingface_token')
                else None
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download with timeout settings
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=index_filename,
                    token=token,
                    local_dir=temp_dir,
                    resume_download=True,  # Resume partial downloads
                )

                with open(downloaded_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content

        def _download_modelscope_index():
            # Create a temporary directory for downloading
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the specific file using ModelScope's snapshot_download
                downloaded_path = snapshot_download(
                    model_id=repo_id,
                    local_dir=temp_dir,
                    allow_file_pattern=[index_filename],  # Only download the index file
                )

                # Read the index file content
                index_file_path = os.path.join(downloaded_path, index_filename)
                if os.path.exists(index_file_path):
                    with open(index_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return content
                else:
                    raise FileNotFoundError(
                        f"Index file {index_filename} not found in ModelScope repository {repo_id}"
                    )

        try:
            if source == SourceEnum.HUGGING_FACE:
                return _download_huggingface_index()
            elif source == SourceEnum.MODEL_SCOPE:
                return _download_modelscope_index()
        except Exception as e:
            logger.warning(f"Failed to get index file content: {e}")
            return None

        return None


@cache
def get_model_lock(model_id: str) -> Lock:
    """Get or create a lock for the given model_id. The model_id is used as the key to store Lock in cache."""
    return Lock()


def get_model_path_and_name(model: Model) -> (str, str):
    if model.source == SourceEnum.HUGGING_FACE:
        return model.huggingface_repo_id, model.huggingface_filename
    elif model.source == SourceEnum.MODEL_SCOPE:
        return model.model_scope_model_id, model.model_scope_file_path
    elif model.source == SourceEnum.OLLAMA_LIBRARY:
        return model.ollama_library_model_name, ""
    elif model.source == SourceEnum.LOCAL_PATH:
        return model.local_path, ""
    else:
        return "", ""


def match_file_and_calculate_size(
    files: List[FileEntry],
    model: Model,
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
        return (
            sum(f.size for f in files if getattr(f, 'size', None) is not None),
            [f"{cache_dir}/{model.source.value}/{file_path}"],
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
        selected_files = [
            f"{cache_dir}/{model.source.value}/{file_path}/{f}" for f in selected_files
        ]

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
    model: Optional[Model] = None,
) -> List[str]:
    """
    Match files in a Hugging Face repository with intelligent filtering.

    Args:
        repo_id: Repository ID
        filename: Filename pattern to match
        extra_filename: Optional extra filename pattern (e.g., for mmproj files)
        token: Optional Hugging Face token
        model: Optional model configuration for intelligent filtering

    Returns:
        List of matched file paths
    """
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

    # Apply intelligent filtering if model is provided and we're dealing with weight files
    if model and matching_files:
        # Convert file paths to FileEntry objects for filtering
        file_entries = []
        for file_path in matching_files:
            # We don't have size info here, but that's OK for filtering logic
            file_entries.append(FileEntry(file_path, 0))

        # Apply filtering to remove duplicates and select optimal versions
        model_file_filter = ModelFileFilter(model, purpose=FilterPurpose.DOWNLOAD)
        filtered_infos = model_file_filter.filter_files(file_entries)
        matching_files = [info.rfilename for info in filtered_infos]

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
    include_subdirs: bool = True,
) -> List[FileEntry]:
    """
    List repository files with optional subdirectory filtering.

    Args:
        repo_id: Repository ID
        source: Source type (HUGGING_FACE or MODEL_SCOPE)
        token: Optional authentication token
        cache_expiration: Cache expiration time
        include_subdirs: Whether to include subdirectories

    Returns:
        List of FileEntry objects
    """
    cache_key = f"{source}:{repo_id}:{include_subdirs}"
    cached_result, is_succ = load_cache(
        LIST_REPO_CACHE_DIR, cache_key, cache_expiration
    )
    if is_succ:
        result = json.loads(cached_result)
        if isinstance(result, list):
            # Convert cached dictionaries back to FileEntry objects
            return [FileEntry(item["name"], item["size"]) for item in result]

    if source == SourceEnum.HUGGING_FACE:
        validate_repo_id(repo_id)
        hffs = HfFileSystem(token=token)
        file_info = []
        for file in hffs.ls(repo_id, recursive=include_subdirs):
            if not isinstance(file, dict):
                continue
            relative_path = Path(file["name"]).relative_to(repo_id).as_posix()
            if file["size"] == 0:
                continue
            if not include_subdirs and "/" in relative_path:
                continue
            file_info.append(FileEntry(relative_path, file["size"]))
    elif source == SourceEnum.MODEL_SCOPE:
        msapi = HubApi()
        files = msapi.get_model_files(repo_id, recursive=include_subdirs)
        file_info = []
        for file in files:
            if file["Size"] == 0:
                continue
            file_info.append(FileEntry(file["Path"], file["Size"]))
    else:
        raise ValueError(f"Invalid source: {source}")

    # Convert FileEntry objects to dictionaries for caching
    cache_data = [{"name": entry.rfilename, "size": entry.size} for entry in file_info]
    if not save_cache(LIST_REPO_CACHE_DIR, cache_key, json.dumps(cache_data)):
        logger.info(f"Saved cache {LIST_REPO_CACHE_DIR} {cache_key} fail")

    return file_info


def filter_filename(file_path: str, file_paths: List[str]):
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]
    matching_paths = sorted(matching_paths)

    return matching_paths


def match_model_scope_file_paths(
    model_id: str,
    file_path: str,
    extra_file_path: Optional[str] = None,
    model: Optional[Model] = None,
) -> List[str]:
    """
    Match files in a ModelScope repository with intelligent filtering.

    Args:
        model_id: Repository ID
        file_path: Filename pattern to match
        extra_file_path: Optional extra filename pattern (e.g., for mmproj files)
        model: Optional model configuration for intelligent filtering

    Returns:
        List of matched file paths
    """
    if '/' in file_path:
        root, _ = file_path.rsplit('/', 1)
    else:
        root = None

    api = HubApi()
    files = api.get_model_files(model_id, root=root, recursive=True)

    file_paths = [file["Path"] for file in files]
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]

    # Apply intelligent filtering if model is provided and we're dealing with weight files
    if model and matching_paths:
        # Convert file paths to file info format for filtering
        file_infos = []
        for file_path_item in matching_paths:
            # Find the corresponding file info with size
            file_info = next((f for f in files if f["Path"] == file_path_item), None)
            if file_info:
                file_infos.append(FileEntry(file_path_item, file_info.get("Size")))
            else:
                file_infos.append(FileEntry(file_path_item, None))

        # Apply filtering to remove duplicates and select optimal versions
        model_file_filter = ModelFileFilter(model, purpose=FilterPurpose.DOWNLOAD)
        filtered_infos = model_file_filter.filter_files(file_infos)
        matching_paths = [info.rfilename for info in filtered_infos]

    matching_paths = sorted(matching_paths)

    if extra_file_path is None:
        return matching_paths

    extra_matching_paths = [
        p for p in file_paths if fnmatch.fnmatch(p, extra_file_path)
    ]
    extra_file = select_most_suitable_extra_file(extra_matching_paths)
    if extra_file:
        matching_paths.append(extra_file)

    return matching_paths


def get_model_weight_size(model: Model, token: Optional[str] = None) -> int:
    """
    Get the size of the model weights using intelligent file filtering to avoid duplicates.
    Args:
        model: Model to get the weight size for
        token: Optional Hugging Face API token
    Returns:
        int: The size of the model weights in bytes
    """
    if model.source == SourceEnum.HUGGING_FACE:
        repo_id = model.huggingface_repo_id
    elif model.source == SourceEnum.MODEL_SCOPE:
        repo_id = model.model_scope_model_id
    else:
        raise ValueError(f"Unknown source {model.source}")

    # Get all repository files (subdirectory filtering will be handled in filter_model_files)
    repo_file_infos = list_repo(
        repo_id, model.source, token=token, include_subdirs=True
    )

    # Apply intelligent filtering to select optimal files
    model_file_filter = ModelFileFilter(model, purpose=FilterPurpose.EVALUATE)
    filtered_files = model_file_filter.filter_files(repo_file_infos)

    total_size = sum(file.size for file in filtered_files if file.size is not None)

    if filtered_files:
        logger.debug(
            f"Weight calculation for {model.readable_source}: "
            f"{len(filtered_files)} files, {total_size / (1024**3):.2f} GB"
        )

    return total_size


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
