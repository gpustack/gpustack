"""Translate common video inputs after the concrete inference target is selected."""

import json
import math
import re
from typing import Any, Dict, Optional, Tuple

import aiohttp

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import Model
from gpustack.utils.video import get_video_capabilities

_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
_REFERENCE_FIELDS = (
    "input_reference",
    "reference_url",
    "image_reference",
    "video_reference",
    "audio_reference",
)


def _object(value: Any, field: str) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _canvas(size: Any) -> Tuple[int, str]:
    match = re.fullmatch(r"([1-9]\d*)x([1-9]\d*)", str(size))
    if match is None:
        raise ValueError("size must use positive WIDTHxHEIGHT dimensions")
    width, height = map(int, match.groups())
    ratios = [(name, tuple(map(int, name.split(":")))) for name in _ASPECT_RATIOS]
    name, (x, y) = min(
        ratios, key=lambda item: abs(width / height - item[1][0] / item[1][1])
    )
    # Published H3 canvases are aligned to 32 pixels, including 1344x768.
    if min(abs(width - height * x / y), abs(height - width * y / x)) > 32:
        raise ValueError("MiniMax H3 size must use a supported aspect ratio")
    return min(width, height), name


def _duration(value: Any) -> float:
    try:
        duration = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("seconds must be between 4 and 15 for MiniMax H3") from exc
    if (
        isinstance(value, bool)
        or not math.isfinite(duration)
        or not 4 <= duration <= 15
        or not duration.is_integer()
    ):
        raise ValueError(
            "seconds must be between 4 and 15 in whole seconds for MiniMax H3"
        )
    return duration


def _extension_parameters(values: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = _object(values.pop("extra_body", None), "extra_body")
    extra_params = _object(values.pop("extra_params", None), "extra_params")
    for key in extra_body.keys() & extra_params.keys():
        if extra_body[key] != extra_params[key]:
            raise ValueError(f"Conflicting video parameter: {key}")
    extras = {**extra_body, **extra_params}
    if {"model", "prompt"} & extras.keys():
        raise ValueError("model and prompt must be top-level video parameters")
    for key in (
        "task",
        "target",
        "conditions",
        "aspect_ratio",
        "duration",
        "duration_seconds",
    ):
        if key not in values:
            continue
        value = values.pop(key)
        if key in ("target", "conditions") and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be valid JSON") from exc
        if key in extras and extras[key] != value:
            raise ValueError(f"Conflicting video parameter: {key}")
        extras[key] = value
    return extras


def adapt_h3_parameters(model: Model, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Fill common text-to-video fields while preserving native reference requests."""
    capabilities = get_video_capabilities(model)
    if capabilities is None:
        return parameters
    values = dict(parameters)
    extras = _extension_parameters(values)
    task = extras.get("task")
    has_reference = any(
        values.get(key) is not None or extras.get(key) is not None
        for key in _REFERENCE_FIELDS
    ) or bool(extras.get("conditions"))
    if task is None and has_reference:
        if capabilities["request_profile"] == "vllm_omni_h3":
            return {**values, "extra_params": extras}
        raise ValueError(
            "MiniMax H3 references require an explicit task and model-specific reference parameters"
        )
    task = capabilities["default_task"] if task is None else task
    if task not in capabilities["tasks"]:
        raise ValueError(f"MiniMax H3 task must be one of {capabilities['tasks']}")
    if task == "t2va" and has_reference:
        raise ValueError("MiniMax H3 t2va accepts text only; remove reference inputs")
    extras["task"] = task
    if task != "t2va":
        return (
            {**extras, **values}
            if capabilities["request_profile"] == "sglang_h3"
            else {**values, "extra_params": extras}
        )
    if capabilities["request_profile"] == "sglang_h3":
        return _sglang_parameters(values, extras, capabilities)
    return _vllm_parameters(values, extras, capabilities)


def _sglang_parameters(values, extras, capabilities):
    target = _object(extras.get("target"), "target")
    if "short_edge" not in target or "aspect_ratio" not in target:
        short_edge, ratio = _canvas(values.get("size") or capabilities["default_size"])
        target.setdefault("short_edge", short_edge)
        target.setdefault(
            "aspect_ratio", "auto" if extras["task"] == "fl2va" else ratio
        )
    if "duration_seconds" not in target:
        target["duration_seconds"] = _duration(values.get("seconds", 4))
    extras["target"] = target
    if extras["task"] == "t2va":
        extras.setdefault("conditions", [])
    # Explicit top-level engine options win, matching SGLang's multipart parser.
    return {**extras, **values}


def _vllm_parameters(values, extras, capabilities):
    target = _object(extras.get("target"), "target")
    if not target.get("aspect_ratio") and "aspect_ratio" not in extras:
        _, ratio = _canvas(values.get("size") or capabilities["default_size"])
        extras["aspect_ratio"] = ratio
    if (
        not any(key in extras for key in ("duration", "duration_seconds"))
        and "duration_seconds" not in target
        and (values.get("seconds") is not None or values.get("num_frames") is None)
    ):
        extras["duration"] = _duration(values.get("seconds", 4))
    return {**values, "extra_params": extras}


def adapt_video_request(
    model: Model,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
) -> Tuple[Optional[dict], Optional[aiohttp.FormData]]:
    """Preserve upload bytes and filenames when translating multipart fields."""
    capabilities = get_video_capabilities(model)
    if capabilities is None:
        return body_json, form_data
    # FormData has not been serialized yet. Retain its upload descriptors so
    # request translation never rereads or copies the uploaded file contents.
    fields = form_data._fields if form_data is not None else []
    values = (
        {options["name"]: value for options, _, value in fields}
        if form_data is not None
        else body_json or {}
    )
    try:
        adapted = adapt_h3_parameters(model, values)
    except ValueError as exc:
        raise BadRequestException(message=str(exc), is_openai_exception=True) from exc
    if form_data is None and capabilities["request_profile"] == "sglang_h3":
        return adapted, None
    result = aiohttp.FormData(default_to_multipart=True)
    uploaded = set()
    for options, headers, value in fields:
        if "filename" in options:
            name = options["name"]
            uploaded.add(name)
            result.add_field(
                name,
                value,
                filename=options["filename"],
                content_type=headers.get("Content-Type"),
            )
    for name, value in adapted.items():
        if name in uploaded or value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        result.add_field(name, str(value))
    return None, result
