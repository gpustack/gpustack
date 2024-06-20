from typing import Union
from datetime import datetime, timedelta, timezone
import jwt
from argon2 import PasswordHasher

ph = PasswordHasher()

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(hashed_password, plain_password) -> bool:
    try:
        return ph.verify(hashed_password, plain_password)
    except Exception:
        return False


def get_password_hash(password):
    return ph.hash(password)


def create_access_token(username: str, expires_delta: Union[timedelta, None] = None):
    to_encode = {"sub": username}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
