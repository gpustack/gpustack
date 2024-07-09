import json
import os
import re
import requests
from typing import List, Literal, Optional, Union
import fnmatch
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfFileSystem
from huggingface_hub.utils import validate_repo_id


class HfDownloader:
    _registry_url = "https://huggingface.co"

    @classmethod
    def download(
        cls,
        repo_id: str,
        filename: Optional[str],
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> str:
        """Download a model from the Hugging Face Hub.

        Args:
            repo_id: The model repo id.
            filename: A filename or glob pattern to match the model file in the repo.
            local_dir: The local directory to save the model to.
            local_dir_use_symlinks: Whether to use symlinks when downloading the model.

        Returns:
            The path to the downloaded model.
        """

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id)
        ]

        # split each file into repo_id, subfolder, filename
        file_list: List[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(files)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
        )

        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, filename)

        return model_path

    def __call__(self):
        return self.download()


class OllamaLibraryDownloader:
    _registry_url = "https://registry.ollama.ai"
    _cache_dir = os.path.expanduser("~/.cache/gpustack/models")

    @staticmethod
    def download_blob(url: str, filename: str):
        temp_filename = filename + ".part"
        headers = {}
        if os.path.exists(temp_filename):
            existing_file_size = os.path.getsize(temp_filename)
            headers = {"Range": f"bytes={existing_file_size}-"}
        else:
            existing_file_size = 0

        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get("content-length", 0)) + existing_file_size

        mode = "ab" if existing_file_size > 0 else "wb"
        chunk_size = 10 * 1024 * 1024  # 10MB
        with (
            open(temp_filename, mode) as file,
            tqdm(total=total_size, initial=existing_file_size) as bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        os.rename(temp_filename, filename)
        print(f"Downloaded {filename}")

    @classmethod
    def download(cls, model_name: str) -> str:
        sanitized_filename = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        if not os.path.exists(cls._cache_dir):
            os.makedirs(cls._cache_dir)

        # Check if the model is already downloaded
        model_path = os.path.join(cls._cache_dir, sanitized_filename)
        if os.path.exists(model_path):
            return model_path

        blob_url = cls.model_url(model_name=model_name)
        if blob_url is not None:
            cls.download_blob(blob_url, model_path)

        return model_path
