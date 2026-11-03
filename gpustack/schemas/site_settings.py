import base64
from datetime import datetime
from typing import Optional

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import CheckConstraint, Column, Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlmodel import SQLModel, Field as SQLField

from gpustack.schemas.common import UTCDateTime

_MAX_ASSET_SIZE = 512 * 1024


def _is_local_path(value: str) -> bool:
    return value.startswith("/") and not value.startswith("//")


def _validate_url_characters(value: str) -> None:
    if "\\" in value or any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise ValueError(
            "URLs must not contain whitespace, control characters or backslashes."
        )


def _validate_asset(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    if len(value) > 4 * ((_MAX_ASSET_SIZE + 2) // 3) + 32:
        raise ValueError("Brand assets must not exceed 512 KiB.")
    header, _, payload = value.partition(",")
    signatures = {
        "data:image/png;base64": b"\x89PNG\r\n\x1a\n",
        "data:image/jpeg;base64": b"\xff\xd8\xff",
        "data:image/webp;base64": b"RIFF",
    }
    if header in signatures:
        try:
            content = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise ValueError(
                "Brand assets must contain valid base64 image data."
            ) from exc
        if len(content) > _MAX_ASSET_SIZE:
            raise ValueError("Brand assets must not exceed 512 KiB.")
        if not content.startswith(signatures[header]) or (
            header == "data:image/webp;base64" and content[8:12] != b"WEBP"
        ):
            raise ValueError("Brand asset content must match its image type.")
        return value
    raise ValueError("Brand assets must be a PNG, JPEG or WebP data URL.")


def _validate_link(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    _validate_url_characters(value)
    if _is_local_path(value):
        return value
    parsed = urlsplit(value)
    # urlsplit defers malformed/out-of-range port validation to this property.
    _ = parsed.port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
    ):
        raise ValueError("Links must be an HTTP(S) URL or a same-origin path.")
    return value


class SiteSettingsUpdate(BaseModel):
    """Replace the instance-wide public branding configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    site_name: str = Field(
        min_length=1,
        max_length=128,
    )
    primary_color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    logo: Optional[str] = Field(default=None)
    logo_dark: Optional[str] = Field(default=None)
    mini_logo: Optional[str] = Field(default=None)
    mini_logo_dark: Optional[str] = Field(default=None)
    favicon: Optional[str] = Field(default=None)
    footer_name: Optional[str] = Field(
        default=None,
        max_length=128,
    )
    footer_url: Optional[str] = Field(
        default=None,
        max_length=2048,
    )

    _validate_logo = field_validator(
        "logo",
        "logo_dark",
        "mini_logo",
        "mini_logo_dark",
        "favicon",
    )(_validate_asset)

    _validate_links = field_validator("footer_url")(_validate_link)

    @field_validator("footer_name")
    @classmethod
    def normalize_footer_name(cls, value: Optional[str]) -> Optional[str]:
        return value or None


class SiteSettingsPublic(BaseModel):
    """Explicit allowlist of configuration visible before authentication."""

    model_config = ConfigDict(from_attributes=True)

    site_name: str = "GPUStack"
    primary_color: str = "#007BFF"
    logo: Optional[str] = None
    logo_dark: Optional[str] = None
    mini_logo: Optional[str] = None
    mini_logo_dark: Optional[str] = None
    favicon: Optional[str] = None
    footer_name: Optional[str] = None
    footer_url: Optional[str] = None
    updated_at: Optional[datetime] = None


class SiteSettings(SQLModel, table=True):
    """One configuration shared by all users and server replicas."""

    __tablename__ = "site_settings"
    __table_args__ = (CheckConstraint("id = 1", name="ck_site_settings_singleton"),)

    id: int = SQLField(
        default=1, primary_key=True, sa_column_kwargs={"autoincrement": False}
    )
    site_name: str = SQLField(default="GPUStack", max_length=128)
    primary_color: str = SQLField(default="#007BFF", max_length=7)
    logo: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(MEDIUMTEXT(), "mysql")
    )
    logo_dark: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(MEDIUMTEXT(), "mysql")
    )
    mini_logo: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(MEDIUMTEXT(), "mysql")
    )
    mini_logo_dark: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(MEDIUMTEXT(), "mysql")
    )
    favicon: Optional[str] = SQLField(
        default=None, sa_type=Text().with_variant(MEDIUMTEXT(), "mysql")
    )
    footer_name: Optional[str] = SQLField(default=None, max_length=128)
    footer_url: Optional[str] = SQLField(default=None, max_length=2048)
    updated_at: Optional[datetime] = SQLField(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
