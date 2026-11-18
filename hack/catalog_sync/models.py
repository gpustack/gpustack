"""The data contract shared by research, validation and publication."""

import hashlib
import json
import re
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

CATALOGS = {
    "huggingface": "gpustack/assets/model-catalog.yaml",
    "model_scope": "gpustack/assets/model-catalog-modelscope.yaml",
}
REPO_FIELDS = {
    "huggingface": "huggingface_repo_id",
    "model_scope": "model_scope_model_id",
}
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_targets(value: str) -> List[str]:
    """Normalize a bounded batch without interpolating input into commands."""
    targets = []
    for item in re.split(r"[\s,]+", value.strip()):
        if not item:
            continue
        source = "hf"
        if item.startswith("https://"):
            url = urlparse(item)
            require(not url.query and not url.fragment, "Use a model repository URL")
            hosts = {"huggingface.co": "hf", "modelscope.cn": "ms"}
            require(url.netloc in hosts, "Unsupported model URL")
            source = hosts[url.netloc]
            item = url.path.strip("/")
            if source == "ms":
                item = item.removeprefix("models/")
        elif ":" in item:
            source, item = item.split(":", 1)
        require(source in {"hf", "ms"}, "Use hf: or ms: for model IDs")
        require(bool(REPO_ID.fullmatch(item)), f"Invalid model ID: {item}")
        target = f"{source}:{item}"
        if target not in targets:
            targets.append(target)
    require(0 < len(targets) <= 10, "Supply between 1 and 10 distinct model IDs")
    return targets


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Citation(Record):
    url: str
    quote: str = Field(min_length=12, max_length=1500)
    explanation: str = Field(min_length=12, max_length=2000)


class HubAssessment(Record):
    status: Literal["available", "unavailable", "unverified"]
    repo_id: Optional[str] = None
    reason: str = Field(min_length=12, max_length=2000)
    evidence: List[Citation] = Field(default_factory=list)


class SpecProof(Record):
    snapshot: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: str = Field(min_length=1, max_length=128)
    deployment: List[Citation] = Field(min_length=1, max_length=8)
    quantization: List[Citation] = Field(min_length=1, max_length=8)
    # A packaged runner or an explicit image from the official backend publisher.
    runner_image: str
    hardware: Literal["nvidia", "ascend:910b", "ascend:910c"]
    compatibility: str = Field(min_length=20, max_length=3000)


class Icon(Record):
    filename: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*\.png$")
    url: str
    github_org: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9-]+$")
    evidence: List[Citation] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=12, max_length=1000)


class Entry(Record):
    source: Literal["huggingface", "model_scope"]
    card: dict
    parameter_method: Literal["official", "safetensors_metadata"] = "official"
    parameter_notes: str = Field(min_length=12, max_length=2000)
    evidence: Dict[str, List[Citation]]
    specs: List[SpecProof] = Field(min_length=1, max_length=12)
    icon: Optional[Icon] = None


class Candidate(Record):
    target: str
    status: Literal["ready", "blocked", "unchanged"]
    reason: str = Field(min_length=12, max_length=2000)
    hubs: Dict[str, HubAssessment]
    ascend_910b: str = Field(min_length=20, max_length=2000)
    entries: List[Entry] = Field(default_factory=list, max_length=2)


class Proposal(Record):
    schema_version: Literal[1]
    candidates: List[Candidate] = Field(min_length=1, max_length=10)
