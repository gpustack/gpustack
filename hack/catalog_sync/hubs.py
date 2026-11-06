"""Collect weight manifests without downloading or executing model weights."""

import fnmatch
import json
from pathlib import PurePosixPath
from typing import List, Optional
from urllib.parse import quote, urlencode

from hack.catalog_sync.http import PublicClient
from hack.catalog_sync.models import REPO_ID, digest, require

INDEX_LIMIT = 64 * 1024 * 1024
SAFETENSORS_INDEXES = (
    "model.safetensors.index.json",
    "consolidated.safetensors.index.json",
)


def file_url(source: str, repo: str, revision: str, path: str) -> str:
    if source == "huggingface":
        return (
            f"https://huggingface.co/{repo}/resolve/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )
    query = urlencode({"Revision": revision, "FilePath": path})
    return f"https://modelscope.cn/api/v1/models/{repo}/repo?{query}"


def _manifest(client, source, repo, revision):
    if source == "huggingface":
        url = f"https://huggingface.co/api/models/{repo}"
        url += f"/revision/{quote(revision or 'main', safe='')}?blobs=true"
        info = client.json(url)
        revision = info["sha"]
        files = {
            f["rfilename"]: {
                "size": f.get("size"),
                "oid": f.get("lfs") or f.get("blobId"),
            }
            for f in info["siblings"]
        }
        return revision, files, (info.get("safetensors") or {}).get("total")
    require(source == "model_scope", "Unsupported hub")
    revision = revision or "master"
    query = urlencode({"Revision": revision, "Recursive": "true"})
    body = client.json(f"https://modelscope.cn/api/v1/models/{repo}/repo/files?{query}")
    require(
        isinstance(body, dict) and body.get("Code") == 200,
        "ModelScope did not return a successful listing",
    )
    data = body.get("Data")
    require(isinstance(data, dict), "ModelScope listing is missing a Data object")
    rows = data.get("Files")
    require(isinstance(rows, list), "ModelScope listing is missing a Files array")
    require(
        all(
            isinstance(row, dict) and isinstance(row.get("Path"), str) and row["Path"]
            for row in rows
        ),
        "ModelScope listing contains an invalid file entry",
    )
    files = {
        f["Path"]: {"size": f.get("Size"), "oid": f.get("Sha256") or f.get("Revision")}
        for f in rows
        if f.get("Type") != "tree"
    }
    return revision, files, None


def select_weights(files: dict, pattern: Optional[str], indexes: dict) -> List[str]:
    """Select one representation, requiring an explicit choice for ambiguity."""
    if pattern:
        selected = sorted(name for name in files if fnmatch.fnmatchcase(name, pattern))
    elif any(name in indexes for name in SAFETENSORS_INDEXES):
        index = next(indexes[name] for name in SAFETENSORS_INDEXES if name in indexes)
        selected = sorted(set(index["weight_map"].values()))
    elif "model.safetensors" in files:
        selected = ["model.safetensors"]
    elif "pytorch_model.bin.index.json" in indexes:
        selected = sorted(
            set(indexes["pytorch_model.bin.index.json"]["weight_map"].values())
        )
    elif "pytorch_model.bin" in files:
        selected = ["pytorch_model.bin"]
    else:
        selected = [
            name
            for name in files
            if PurePosixPath(name).suffix
            in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
        ]
        require(len(selected) == 1, "Ambiguous weights: supply a filename selection")
    require(bool(selected), "No matching weight files")
    suffixes = {PurePosixPath(name).suffix for name in selected}
    require(len(suffixes) == 1, "Do not sum duplicate weight representations")
    require(
        suffixes <= {".safetensors", ".bin", ".pt", ".pth", ".gguf"},
        "Selection must contain weight files only",
    )
    if not pattern:
        extras = {
            name
            for name in files
            if PurePosixPath(name).suffix in suffixes
            and name not in selected
            and name != "consolidated.safetensors"
        }
        require(not extras, "Additional weights need an explicit complete selection")
    for index in indexes.values():
        if not isinstance(index, dict) or "weight_map" not in index:
            continue
        shards = set(index["weight_map"].values())
        if shards.intersection(selected):
            require(shards <= set(selected), "Selection omits indexed weight shards")
    for name in selected:
        require(name in files, f"Missing weight shard: {name}")
        size = files[name]["size"]
        require(type(size) is int and size > 0, f"Missing weight size: {name}")
    return selected


def _documents(client, source, repo, revision, files):
    documents = {}
    if "config.json" in files:
        documents["config.json"] = client.json(
            file_url(source, repo, revision, "config.json")
        )
    for path in (*SAFETENSORS_INDEXES, "pytorch_model.bin.index.json"):
        if path not in files:
            continue
        size = files[path]["size"]
        require(type(size) is int and 0 < size <= INDEX_LIMIT, "Invalid index size")
        index = client.json(file_url(source, repo, revision, path), limit=INDEX_LIMIT)
        weights = index.get("weight_map") if isinstance(index, dict) else None
        require(
            isinstance(weights, dict)
            and bool(weights)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in weights.items()
            ),
            "Invalid weight index mapping",
        )
        documents[path] = index
    return documents


def _compact_documents(documents):
    # Bind every tensor mapping into the digest without copying large indexes
    # into proposals and validation artifacts. Shard selection uses the full map.
    return {
        path: (
            {
                "sha256": digest(document),
                "metadata": document.get("metadata", {}),
                "shards": sorted(set(document["weight_map"].values())),
                "tensor_count": len(document["weight_map"]),
            }
            if path.endswith(".index.json")
            else document
        )
        for path, document in documents.items()
    }


def collect(
    client: PublicClient,
    source: str,
    repo: str,
    revision: Optional[str] = None,
    pattern: Optional[str] = None,
) -> dict:
    """Snapshot the exact selected files, config and index metadata.

    ModelScope branch snapshots carry file hashes as well as the branch name;
    validation reacquires the same snapshot and rejects changed content.
    """
    require(bool(REPO_ID.fullmatch(repo)), "Invalid repository ID")
    revision, files, parameters = _manifest(client, source, repo, revision)
    documents = _documents(client, source, repo, revision, files)
    selected = select_weights(files, pattern, documents)
    require(
        all(files[name]["oid"] for name in selected), "Weights need content identifiers"
    )
    result = {
        "source": source,
        "repo_id": repo,
        "revision": revision,
        "files": {name: files[name] for name in selected},
        "weight_bytes": sum(files[name]["size"] for name in selected),
        "documents": _compact_documents(documents),
        "safetensors_parameters": parameters,
    }
    result["snapshot"] = digest(result)
    # The model card is research context; its dates are not release-date defaults.
    for path in ("README.md", "readme.md"):
        if path in files:
            result["model_card_url"] = file_url(source, repo, revision, path)
            result["model_card"] = client.get(result["model_card_url"]).decode("utf-8")
            break
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["huggingface", "model_scope"])
    parser.add_argument("repo")
    parser.add_argument("--revision")
    parser.add_argument("--pattern")
    args = parser.parse_args()
    print(json.dumps(collect(PublicClient(), **vars(args)), indent=2))


if __name__ == "__main__":
    main()
