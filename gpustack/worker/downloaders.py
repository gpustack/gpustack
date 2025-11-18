import logging
import os
from typing import List, Optional, Union
from pathlib import Path
from tqdm.contrib.concurrent import thread_map

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import (
    snapshot_download as modelscope_snapshot_download,
)
from modelscope.hub.utils.utils import model_id_to_group_owner_name

from gpustack.schemas.models import Model, ModelSource, SourceEnum, get_mmproj_filename
from gpustack.utils import file
from gpustack.utils.hub import (
    match_hugging_face_files,
    match_model_scope_file_paths,
    FileEntry,
)
from gpustack.utils.locks import HeartbeatSoftFileLock

logger = logging.getLogger(__name__)


def download_model(
    model: ModelSource,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> List[str]:
    if model.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.download(
            repo_id=model.huggingface_repo_id,
            filename=model.huggingface_filename,
            extra_filename=get_mmproj_filename(model),
            token=huggingface_token,
            local_dir=local_dir,
            cache_dir=os.path.join(cache_dir, "huggingface"),
            owner_worker_id=getattr(model, "worker_id", None),
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.download(
            model_id=model.model_scope_model_id,
            file_path=model.model_scope_file_path,
            extra_file_path=get_mmproj_filename(model),
            local_dir=local_dir,
            cache_dir=os.path.join(cache_dir, "model_scope"),
            owner_worker_id=getattr(model, "worker_id", None),
        )
    elif model.source == SourceEnum.LOCAL_PATH:
        return file.get_sharded_file_paths(model.local_path)


def get_model_file_info(
    model: Model,
    huggingface_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[FileEntry]:
    if model.source == SourceEnum.HUGGING_FACE:
        return HfDownloader.get_model_file_info(
            model=model,
            token=huggingface_token,
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        return ModelScopeDownloader.get_model_file_info(
            model=model,
        )
    elif model.source == SourceEnum.LOCAL_PATH:
        sharded_or_original_file_paths = file.get_sharded_file_paths(model.local_path)
        file_list = [
            FileEntry(f, file.getsize(f)) for f in sharded_or_original_file_paths
        ]
        return file_list

    raise ValueError(f"Unsupported model source: {model.source}")


class HfDownloader:
    _registry_url = "https://huggingface.co"

    @classmethod
    def get_model_file_info(cls, model: Model, token: Optional[str]) -> List[FileEntry]:

        api = HfApi(token=token)
        repo_info = api.repo_info(model.huggingface_repo_id, files_metadata=True)
        file_list = [FileEntry(f.rfilename, f.size) for f in repo_info.siblings]
        return file_list

    @classmethod
    def download(
        cls,
        repo_id: str,
        filename: Optional[str],
        extra_filename: Optional[str],
        token: Optional[str] = None,
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        max_workers: int = 8,
        owner_worker_id: Optional[int] = None,
    ) -> List[str]:
        """Download a model from the Hugging Face Hub.

        Args:
            repo_id:
                The model repo id.
            filename:
                A filename or glob pattern to match the model file in the repo.
            token:
                The Hugging Face API token.
            local_dir:
                The local directory to save the model to.
            local_dir_use_symlinks:
                Whether to use symlinks when downloading the model.
            max_workers (`int`, *optional*):
                Number of concurrent threads to download files (1 thread = 1 file download).
                Defaults to 8.

        Returns:
            The paths to the downloaded model files.
        """

        group_or_owner, name = model_id_to_group_owner_name(repo_id)
        lock_filename = os.path.join(cache_dir, group_or_owner, f"{name}.lock")

        if local_dir is None:
            local_dir = os.path.join(cache_dir, group_or_owner, name)

        logger.info(f"Retrieving file lock: {lock_filename}")
        with HeartbeatSoftFileLock(lock_filename, owner_worker_id=owner_worker_id):
            if filename:
                return cls.download_file(
                    repo_id=repo_id,
                    filename=filename,
                    token=token,
                    local_dir=local_dir,
                    extra_filename=extra_filename,
                )

            snapshot_download(
                repo_id=repo_id,
                token=token,
                local_dir=local_dir,
            )
            return [local_dir]

    @classmethod
    def download_file(
        cls,
        repo_id: str,
        filename: Optional[str],
        token: Optional[str] = None,
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        max_workers: int = 8,
        extra_filename: Optional[str] = None,
    ) -> List[str]:
        """Download a model from the Hugging Face Hub.
        Args:
            repo_id: The model repo id.
            filename: A filename or glob pattern to match the model file in the repo.
            token: The Hugging Face API token.
            local_dir: The local directory to save the model to.
            local_dir_use_symlinks: Whether to use symlinks when downloading the model.
        Returns:
            The path to the downloaded model.
        """

        matching_files = match_hugging_face_files(
            repo_id, filename, extra_filename, token
        )

        if len(matching_files) == 0:
            raise ValueError(f"No file found in {repo_id} that match {filename}")

        logger.info(f"Downloading model {repo_id}/{filename}")

        subfolder = (
            None
            if (subfolder := str(Path(matching_files[0]).parent)) == "."
            else subfolder
        )

        unfolder_matching_files = [Path(file).name for file in matching_files]
        downloaded_files = []

        def _inner_hf_hub_download(repo_file: str):
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                token=token,
                subfolder=subfolder,
                local_dir=local_dir,
            )
            downloaded_files.append(downloaded_file)

        thread_map(
            _inner_hf_hub_download,
            unfolder_matching_files,
            desc=f"Fetching {len(unfolder_matching_files)} files",
            max_workers=max_workers,
        )

        logger.info(f"Downloaded model {repo_id}/{filename}")
        return sorted(downloaded_files)

    def __call__(self):
        return self.download()


class ModelScopeDownloader:

    @classmethod
    def get_model_file_info(cls, model: Model) -> List[FileEntry]:
        api = HubApi()
        repo_files = api.get_model_files(model.model_scope_model_id, recursive=True)
        file_list = [FileEntry(f.get("Path"), f.get("Size")) for f in repo_files]
        return file_list

    @classmethod
    def download(
        cls,
        model_id: str,
        file_path: Optional[str],
        extra_file_path: Optional[str],
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        owner_worker_id: Optional[int] = None,
    ) -> List[str]:
        """Download a model from Model Scope.

        Args:
            model_id:
                The model id.
            file_path:
                A filename or glob pattern to match the model file in the repo.
            cache_dir:
                The cache directory to save the model to.

        Returns:
            The path to the downloaded model.
        """

        group_or_owner, name = model_id_to_group_owner_name(model_id)
        lock_filename = os.path.join(cache_dir, group_or_owner, f"{name}.lock")

        if local_dir is None:
            local_dir = os.path.join(cache_dir, group_or_owner, name)

        logger.info(f"Retrieving file lock: {lock_filename}")
        with HeartbeatSoftFileLock(lock_filename, owner_worker_id=owner_worker_id):
            if file_path:
                matching_files = match_model_scope_file_paths(
                    model_id, file_path, extra_file_path
                )
                if len(matching_files) == 0:
                    raise ValueError(
                        f"No file found in {model_id} that match {file_path}"
                    )

                model_dir = modelscope_snapshot_download(
                    model_id=model_id,
                    local_dir=local_dir,
                    allow_patterns=matching_files,
                )
                return [os.path.join(model_dir, file) for file in matching_files]

            modelscope_snapshot_download(
                model_id=model_id,
                local_dir=local_dir,
            )
            return [local_dir]
