"""Graphic CAPTCHA generation and opaque challenge tokens."""

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass

from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
from cryptography.fernet import Fernet, InvalidToken

# Avoid characters that are easy to confuse in the generated image.
CAPTCHA_ALPHABET = "23456789ACDEFGHJKMNPRTUVWXY"
CAPTCHA_MIN_LENGTH = 4
CAPTCHA_MAX_LENGTH = 6

_TOKEN_PURPOSE = "login_captcha"
_TOKEN_VERSION = 2
_TOKEN_KEY_CONTEXT = b"gpustack-login-captcha-token-v1"
_IMAGE_WIDTH = 160
_IMAGE_HEIGHT = 60
_IMAGE_FONT_SIZES = (36, 42, 48)


class InvalidCaptchaToken(ValueError):
    """Raised when a CAPTCHA challenge token is invalid or expired."""


@dataclass(frozen=True)
class CaptchaChallenge:
    code: str
    nonce: str
    binding: str


def generate_code(length: int = 4) -> str:
    """Return a random code drawn from the unambiguous alphabet."""
    if not CAPTCHA_MIN_LENGTH <= length <= CAPTCHA_MAX_LENGTH:
        raise ValueError(
            f"CAPTCHA length must be between {CAPTCHA_MIN_LENGTH} "
            f"and {CAPTCHA_MAX_LENGTH}"
        )
    return "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(length))


def generate_captcha(length: int = 4) -> tuple[str, bytes]:
    """Generate a CAPTCHA and return its code with PNG image bytes."""
    code = generate_code(length)
    generator = ImageCaptcha(
        width=_IMAGE_WIDTH,
        height=_IMAGE_HEIGHT,
        font_sizes=_IMAGE_FONT_SIZES,
    )
    image = generator.generate(code, format="png")
    return code, image.getvalue()


def generate_audio(code: str) -> bytes:
    """Generate a WAV rendering of a validated CAPTCHA code."""
    if not CAPTCHA_MIN_LENGTH <= len(code) <= CAPTCHA_MAX_LENGTH:
        raise ValueError("Invalid CAPTCHA code length")
    if any(char.upper() not in CAPTCHA_ALPHABET for char in code):
        raise ValueError("Invalid CAPTCHA code")
    return bytes(AudioCaptcha().generate(code.upper()))


def encrypt_challenge(secret_key: str, code: str, nonce: str, binding: str) -> str:
    """Encrypt a short-lived CAPTCHA challenge for delivery to the browser."""
    payload = {
        "purpose": _TOKEN_PURPOSE,
        "version": _TOKEN_VERSION,
        "code": code.lower(),
        "nonce": nonce,
        "binding": binding,
    }
    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _build_cipher(secret_key).encrypt(plaintext).decode("ascii")


def decrypt_challenge(
    secret_key: str, token: str, ttl_seconds: int
) -> CaptchaChallenge:
    """Decrypt and validate a CAPTCHA challenge token."""
    try:
        plaintext = _build_cipher(secret_key).decrypt(
            token.encode("ascii"), ttl=ttl_seconds
        )
        payload = json.loads(plaintext)
    except (InvalidToken, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise InvalidCaptchaToken from exc

    if not isinstance(payload, dict):
        raise InvalidCaptchaToken
    if (
        payload.get("purpose") != _TOKEN_PURPOSE
        or payload.get("version") != _TOKEN_VERSION
    ):
        raise InvalidCaptchaToken

    code = payload.get("code")
    nonce = payload.get("nonce")
    binding = payload.get("binding")
    if not isinstance(code, str) or not (
        CAPTCHA_MIN_LENGTH <= len(code) <= CAPTCHA_MAX_LENGTH
    ):
        raise InvalidCaptchaToken
    if any(char.upper() not in CAPTCHA_ALPHABET for char in code):
        raise InvalidCaptchaToken
    if not isinstance(nonce, str) or not 16 <= len(nonce) <= 128:
        raise InvalidCaptchaToken
    if not isinstance(binding, str) or not 32 <= len(binding) <= 128:
        raise InvalidCaptchaToken
    return CaptchaChallenge(code=code, nonce=nonce, binding=binding)


def _build_cipher(secret_key: str) -> Fernet:
    if not secret_key:
        raise ValueError("CAPTCHA token secret must not be empty")
    # Domain-separate CAPTCHA encryption from the JWT use of the same root key.
    derived_key = hmac.new(
        secret_key.encode("utf-8"), _TOKEN_KEY_CONTEXT, hashlib.sha256
    ).digest()
    return Fernet(base64.urlsafe_b64encode(derived_key))
