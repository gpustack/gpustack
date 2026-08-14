import re
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Tuple
from functools import lru_cache
import jwt
from argon2 import PasswordHasher
import base64
import hashlib

from gpustack import envs

ph = PasswordHasher()

API_KEY_PREFIX = "gpustack"

# Number of random bytes behind a secret this server generates itself. 16 bytes
# of CSPRNG output (32 hex characters) is what makes the fast digest below sound:
# there is no dictionary to try, so a work factor buys nothing.
GENERATED_SECRET_KEY_BYTES = 16
GENERATED_ACCESS_KEY_BYTES = 8

# ``secret_key_digest`` value format, self-describing like argon2's own: the
# algorithm name is the prefix, the salt is inlined. Changing construction means
# changing the prefix, so the stored string always says how to verify it.
SECRET_KEY_DIGEST_ALGORITHM = "sha256"
_DIGEST_SALT_BYTES = 16
_GENERATED_SECRET_KEY_RE = re.compile(f"^[0-9a-f]{{{GENERATED_SECRET_KEY_BYTES * 2}}}$")

# The characters that count as "special" for the password policy. The set is
# the one the UI's client-side check offers, so a password its form accepts is
# a password this API accepts:
#
#   /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*_+.])[a-zA-Z\d!@#$%^&*_+.]{6,64}$/
#
# The length bound and the allowlist of permitted characters in that regex are
# the UI's own. A client may be stricter than the server; the server is the
# authority, and NIST SP 800-63B argues against capping length or restricting
# the alphabet there.
#
# Both the validator and ``generate_secure_password`` read this, and they have
# to agree: ``gpustack reset-admin-password`` submits a generated password
# through the regular UserUpdate path, so anything the generator can emit that
# the validator would reject turns that command into a 422.
PASSWORD_SPECIAL_CHARACTERS = "!@#$%^&*_+."


def _as_text(value: Union[str, bytes]) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _as_bytes(value: Union[str, bytes]) -> bytes:
    return value if isinstance(value, bytes) else value.encode()


# argon2 is deliberately expensive and the credentials left on this path -- custom
# keys and the legacy cluster token -- have no digest to fall back on, so the memo
# is what keeps a frequently polling worker off a ~30 ms verify per request.
# ``lru_cache`` is thread-safe and shared across the ``asyncio.to_thread`` workers
# that call this, at the cost of holding the plaintext as part of the key. That
# residency is bounded to the credentials above: anything carrying a
# ``secret_key_digest`` never reaches here (see get_user_from_api_token).
@lru_cache(maxsize=2048)
def verify_hashed_secret(hashed: Union[str, bytes], plain: Union[str, bytes]) -> bool:
    try:
        return ph.verify(hashed, plain)
    except Exception:
        return False


def get_secret_hash(plain: Union[str, bytes]):
    return ph.hash(plain)


def generate_access_key() -> str:
    return secrets.token_hex(GENERATED_ACCESS_KEY_BYTES)


def generate_secret_key() -> str:
    return secrets.token_hex(GENERATED_SECRET_KEY_BYTES)


def secret_key_digest_eligible(
    *, secret_key: Union[str, bytes], is_custom: bool, access_key: str
) -> bool:
    """Whether ``secret_key`` may be stored under the fast digest.

    The decision lives here rather than at the call sites because getting it
    wrong stores a secret behind a hash that is cheap to search, and a call site
    cannot see that.

    A secret this server generated always qualifies, on two conditions that have
    to hold together: a non-empty ``access_key``, which excludes the legacy
    cluster token (its row stores the deployment-provided ``config.token`` under
    an empty access key and never goes through the custom flag), and the exact
    hex shape of :func:`generate_secret_key`. Neither suffices alone -- a custom
    key may well be 32 hex characters, an md5 of a dictionary word being one.

    A custom secret qualifies only when the deployment says so, via
    ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS``. The shape test does not apply to it:
    a custom secret is whatever the user typed, so there is nothing to check it
    against. That is also why the flag exists rather than a rule -- nothing in
    this system knows how much entropy a given custom key has, and
    ``ApiKeyCreate.custom`` imposes none, so the choice is the operator's to
    make with knowledge this code does not have. See the flag's own comment for
    what it costs.
    """
    if not access_key:
        return False
    if is_custom:
        return envs.GATEWAY_AUTH_ALLOW_CUSTOM_KEYS
    return bool(_GENERATED_SECRET_KEY_RE.match(_as_text(secret_key)))


def new_secret_key_digest(
    *, secret_key: Union[str, bytes], is_custom: bool, access_key: str
) -> Optional[str]:
    """A fresh ``sha256$<salt>$<hash>`` digest, or None if not eligible.

    The salt blocks precomputation and keeps two keys that happen to share a
    secret from producing the same digest.
    """
    if not secret_key_digest_eligible(
        secret_key=secret_key, is_custom=is_custom, access_key=access_key
    ):
        return None
    salt = secrets.token_hex(_DIGEST_SALT_BYTES)
    return f"{SECRET_KEY_DIGEST_ALGORITHM}${salt}${_digest_hash(salt, secret_key)}"


# The form the gateway config carries. Same salt, same hashed input, same
# digest -- only the expected hash is truncated to its leading 16 bytes and
# re-encoded. A derivation rather than a second construction, which is what
# lets an existing key shrink: rewriting a *stored* digest would need the
# plaintext, and the plaintext stops reaching this server once its key
# authenticates at the gateway.
GATEWAY_DIGEST_ALGORITHM = "s128"
_GATEWAY_DIGEST_BYTES = 16


def gateway_digest(digest: Optional[str]) -> Optional[str]:
    """``digest`` rewritten for the gateway's key table, or None if unusable.

    Every byte here is multiplied by the number of keys: the table ships in a
    WasmPlugin CR, etcd caps an object at ~1.5 MiB, and the digest is most of an
    entry. 60 characters instead of 104 is the difference between roughly 6000
    and 8600 keys authenticating at the gateway rather than at the server.

    Truncating to 128 bits is the matching length, not a compromise. The secret
    being verified is 128 bits of CSPRNG output (:func:`generate_secret_key`),
    so a second preimage costs 2^128 -- exactly what guessing the secret
    outright costs, and the discarded half was protecting nothing. That
    reasoning is tied to ``GENERATED_SECRET_KEY_BYTES`` and does not survive
    being applied to a lower-entropy secret, which is one more reason
    :func:`secret_key_digest_eligible` refuses those.

    The algorithm prefix changes with the encoding, which is what keeps a
    rollback safe: an older gateway cannot name ``s128`` so it falls through to
    the server, whereas one that still read ``sha256`` and compared a truncated
    hash against a full one would reject the request outright.
    """
    parsed = _parse_secret_key_digest(digest)
    if parsed is None:
        return None
    salt, expected = parsed
    try:
        raw = bytes.fromhex(expected)
    except ValueError:
        return None
    truncated = base64.urlsafe_b64encode(raw[:_GATEWAY_DIGEST_BYTES])
    return f"{GATEWAY_DIGEST_ALGORITHM}${salt}${truncated.decode().rstrip('=')}"


def secret_key_digest_usable(digest: Optional[str]) -> bool:
    """Whether a stored digest is one this build can check.

    Callers must consult this before treating a False from
    :func:`verify_secret_key_digest` as a rejection. A missing, malformed or
    unknown-algorithm value says nothing about the secret, and the argon2 hash --
    which every key keeps permanently -- stays authoritative in that case.
    Rejecting on an unusable digest would lock out a valid key on bad column
    data, and would also make introducing a new digest algorithm a breaking
    change for anything an older server reads.
    """
    return _parse_secret_key_digest(digest) is not None


def verify_secret_key_digest(
    digest: Optional[str], secret_key: Union[str, bytes]
) -> bool:
    """Constant-time check of a plaintext against a stored digest.

    True only on a match. False covers both a real mismatch and a value this
    build cannot check, so it is not on its own grounds for rejection -- see
    :func:`secret_key_digest_usable`.
    """
    parsed = _parse_secret_key_digest(digest)
    if parsed is None:
        return False
    salt, expected = parsed
    return secrets.compare_digest(_digest_hash(salt, secret_key), expected)


def _parse_secret_key_digest(digest: Optional[str]) -> Optional[Tuple[str, str]]:
    """Split a stored value into (salt, hash), or None if unusable.

    Single source of the value's layout: nothing outside this module needs to
    know that it is ``<algorithm>$<salt>$<hash>``, or which algorithm names this
    build understands.
    """
    if not digest:
        return None
    parts = digest.split("$")
    if len(parts) != 3:
        return None
    algorithm, salt, expected = parts
    if algorithm != SECRET_KEY_DIGEST_ALGORITHM or not salt or not expected:
        return None
    return salt, expected


def _digest_hash(salt: str, secret_key: Union[str, bytes]) -> str:
    # The salt is fixed-length and comes from the entry being checked, so the
    # trailing secret is the only variable and no separator is needed.
    return hashlib.sha256(_as_bytes(salt) + _as_bytes(secret_key)).hexdigest()


def generate_secure_password(length=12):
    if length < 8:
        raise ValueError("Password length should be at least 8 characters")

    special_characters = PASSWORD_SPECIAL_CHARACTERS
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
       - access_key: 8 random bytes, i.e. 16 hex characters (e.g. "3192253c1f4a9b7e")
       - secret_key: 16 random bytes, i.e. 32 hex characters
         (e.g. "c11c75ed6334ea9505da4ad9c11c75ed")
       - Used for normal API authentication via /v2/* routes
       - The counts are byte counts doubled -- see GENERATED_ACCESS_KEY_BYTES /
         GENERATED_SECRET_KEY_BYTES, which ``secret_key_digest_eligible`` keys
         its shape test on

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


AUTH_CACHE_HEADER = "x-gpustack-auth-cache"


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

    def create_token(self, payload: dict, expires_delta: Optional[timedelta] = None):
        delta = expires_delta if expires_delta is not None else self.expires_delta
        to_encode = {"data": payload, "exp": datetime.now(timezone.utc) + delta}
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_jwt_token(self, token: str):
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

    def decode_jwt_data(self, token: str) -> dict:
        return self.decode_jwt_token(token)["data"]
