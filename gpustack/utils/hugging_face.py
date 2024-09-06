from typing import List
from pathlib import Path
import fnmatch
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id


def match_hf_files(repo_id: str, filename: str) -> List[str]:
    validate_repo_id(repo_id)

    hffs = HfFileSystem()

    files = [
        file["name"] if isinstance(file, dict) else file for file in hffs.ls(repo_id)
    ]

    file_list: List[str] = []
    for file in files:
        rel_path = Path(file).relative_to(repo_id)
        file_list.append(str(rel_path))

    matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore
    matching_files = sorted(matching_files)
    return matching_files
