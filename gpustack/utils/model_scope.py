import fnmatch
from typing import List
from modelscope.hub.api import HubApi


def match_model_scope_file_paths(model_id: str, file_path: str) -> List[str]:
    if '/' in file_path:
        root, _ = file_path.rsplit('/', 1)
    else:
        root = None

    api = HubApi()
    files = api.get_model_files(model_id, root=root)

    file_paths = [file["Path"] for file in files]
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]
    matching_paths = sorted(matching_paths)
    return matching_paths
