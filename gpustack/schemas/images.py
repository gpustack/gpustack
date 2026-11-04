import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

from pydantic import BaseModel
from openai.types.image import Image

# Inference endpoints (path after ``/v1/``) whose usage is metered in images.
IMAGE_GENERATION_ENDPOINTS = frozenset({"images/generations", "images/edits"})

# ``1024x1024`` / ``1792X1024``. OpenAI-compatible backends also accept
# ``auto`` and backend-specific presets, which carry no pixel dimensions.
_IMAGE_SIZE_PATTERN = re.compile(r"^\s*(\d{1,6})\s*[xX*]\s*(\d{1,6})\s*$")


class ImageData(BaseModel):
    index: int
    object: str
    progress: float
    b64_json: Optional[str] = None
    finish_reason: Optional[str] = None


class ImageGenerationChunk(BaseModel):
    created: int
    object: str
    model: Optional[str] = None
    data: List[ImageData] = []
    usage: Optional[dict] = None


class ImagesResponse(BaseModel):
    created: int

    data: Optional[List[Image]] = None

    usage: Optional[dict] = None


@dataclass(frozen=True)
class ImageRequestParams:
    """The billing-relevant parameters of an image generation request.

    Captured by the inference proxy before the request is forwarded, so the
    usage middleware can meter the response without re-reading (possibly
    multipart) request bodies.
    """

    n: int = 1
    size: Optional[str] = None

    @classmethod
    def from_mapping(cls, params: Optional[Mapping[str, Any]]) -> "ImageRequestParams":
        """Build from a JSON body or a form, tolerating missing / malformed values.

        Args:
            params: The request's JSON body or form fields.

        Returns:
            The parsed parameters; ``n`` falls back to the OpenAI default of 1.
        """
        if not params:
            return cls()
        try:
            n = int(params.get("n") or 1)
        except (TypeError, ValueError):
            n = 1
        size = params.get("size")
        return cls(n=max(n, 1), size=size if isinstance(size, str) else None)


def parse_image_size(size: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a ``WIDTHxHEIGHT`` image size.

    Args:
        size: The ``size`` request parameter.

    Returns:
        ``(width, height)``, or ``None`` when the value is absent, symbolic
        (``auto``) or not a positive pixel size.
    """
    if not size:
        return None
    match = _IMAGE_SIZE_PATTERN.match(size)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def compute_image_usage(
    params: Optional[ImageRequestParams], generated_count: Optional[int] = None
) -> Tuple[int, int]:
    """Derive the metered image quantities for one image generation request.

    Args:
        params: Parameters of the request, or ``None`` when they were not captured.
        generated_count: Number of images the backend actually returned, when
            the response exposes it. It takes precedence over the requested
            ``n`` because a backend may return fewer images than asked for.

    Returns:
        ``(image_count, image_pixel_count)``. The pixel count is 0 when the
        size is not a concrete ``WIDTHxHEIGHT``.
    """
    params = params or ImageRequestParams()
    count = generated_count if generated_count is not None else params.n
    count = max(count, 0)
    dimensions = parse_image_size(params.size)
    if dimensions is None:
        return count, 0
    width, height = dimensions
    return count, count * width * height
