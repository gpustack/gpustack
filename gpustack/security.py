import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Tuple
from functools import lru_cache
import jwt
from argon2 import PasswordHasher
import hashlib

from gpustack import envs

ph = PasswordHasher()

API_KEY_PREFIX = "gpustack"


@lru_cache(maxsize=2048)
def verify_hashed_secret(hashed: Union[str, bytes], plain: Union[str, bytes]) -> bool:
    try:
        return ph.verify(hashed, plain)
    except Exception:
        return False


def get_secret_hash(plain: Union[str, bytes]):
    return ph.hash(plain)


def generate_secure_password(length=12):
    if length < 8:
        raise ValueError("Password length should be at least 8 characters")

    special_characters = "!@#$%^&*_+"
    characters = string.ascii_letters + string.digits + special_characters
    while True:
        password = ''.join(secrets.choice(characters) for i in range(length))
        if (
            any(c.islower() for c in password)
            and any(c.isupper() for c in password)
            and any(c.isdigit() for c in password)
            and any(c in special_characters for c in password)
        ):
            return password


def custom_key_hash(secret_key: str) -> str:
    return hashlib.blake2b(secret_key.encode(), digest_size=16).hexdigest()


def is_valid_format(key: str) -> Tuple[bool, str, str]:
    if not key.startswith(f"{API_KEY_PREFIX}_"):
        return False, "", ""
    parts = key.split("_", 2)
    if len(parts) != 3:
        return False, "", ""
    access_key, secret_key = parts[1], parts[2]
    return True, access_key, secret_key


def get_key_pair(key: str) -> Tuple[str, str]:
    """
    Parse and validate an API key.

    Scenarios:
    1. Standard format key: "gpustack_{access_key}_{secret_key}"
       - access_key: 8 characters (hex string, e.g. "3192253c")
       - secret_key: 16 characters (hex string, e.g. "c11c75ed6334ea9505da4ad9")
       - Used for normal API authentication via /v2/* routes

    2. Legacy UUID format key: standard UUID format with dashes
       - Example: access_key: "3192253c-c11c-75ed-6334-ea9505da4ad9", the secret_key can be any string
       - Used by legacy worker tokens that use UUID as identifier
       - Falls back to custom_key_hash for backward compatibility

    3. Custom/unrecognized format key:
       - Example: "any_random_string_here", "sk-xxx"
       - Any other string format that doesn't match standard format
       - Returns hashed value for storage, original value for lookup
       - Used for backward compatibility with non-standard API keys

    Returns:
        Tuple of (access_key, secret_key):
        - For standard format: returns the parsed access_key and secret_key
        - For non-standard format: returns (custom_key_hash(key), key)
    """
    valid, access_key, secret_key = is_valid_format(key)
    if not valid:
        return custom_key_hash(key), key
    return access_key, secret_key


class JWTManager:
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        expires_delta: Optional[timedelta] = None,
    ):
        if expires_delta is None:
            expires_delta = timedelta(minutes=envs.JWT_TOKEN_EXPIRE_MINUTES)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expires_delta = expires_delta

    def create_jwt_token(self, username: str):
        to_encode = {"sub": username}
        expire = datetime.now(timezone.utc) + self.expires_delta
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def decode_jwt_token(self, token: str):
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
