import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from functools import lru_cache
import jwt
from argon2 import PasswordHasher

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
