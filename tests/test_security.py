import hashlib

import pytest

from gpustack import security
from gpustack.security import (
    GENERATED_SECRET_KEY_BYTES,
    SECRET_KEY_DIGEST_ALGORITHM,
    generate_access_key,
    generate_secret_key,
    new_secret_key_digest,
    secret_key_digest_eligible,
    secret_key_digest_usable,
    verify_hashed_secret,
    verify_secret_key_digest,
)


def test_generated_secret_key_shape():
    assert len(generate_secret_key()) == GENERATED_SECRET_KEY_BYTES * 2
    assert len(generate_access_key()) == 16


def test_digest_round_trip():
    secret = generate_secret_key()
    access_key = generate_access_key()
    digest = new_secret_key_digest(
        secret_key=secret, is_custom=False, access_key=access_key
    )

    algorithm, salt, _ = digest.split("$")
    assert algorithm == SECRET_KEY_DIGEST_ALGORITHM
    assert salt  # inlined so the stored value is self-describing
    assert verify_secret_key_digest(digest, secret)
    assert not verify_secret_key_digest(digest, generate_secret_key())


def test_both_verifiers_accept_the_same_plaintext():
    """The unit-level half of the downgrade guarantee: a key carrying a digest
    still verifies through argon2 alone, which is all an older version has.

    The end-to-end version — create on this version, let the backfill run, then
    start an older one against the same database — needs two builds and stays a
    manual check.
    """
    secret = generate_secret_key()
    hashed = security.get_secret_hash(secret)
    digest = new_secret_key_digest(
        secret_key=secret, is_custom=False, access_key=generate_access_key()
    )

    assert verify_secret_key_digest(digest, secret)
    assert verify_hashed_secret(hashed, secret)  # the only path an old build has


def test_digest_is_salted_per_key():
    """Two keys that happen to share a secret must not share a digest."""
    secret = generate_secret_key()
    first = new_secret_key_digest(
        secret_key=secret, is_custom=False, access_key=generate_access_key()
    )
    second = new_secret_key_digest(
        secret_key=secret, is_custom=False, access_key=generate_access_key()
    )

    assert first != second
    assert verify_secret_key_digest(first, secret)
    assert verify_secret_key_digest(second, secret)


@pytest.mark.parametrize(
    "digest",
    [
        None,
        "",
        "sha256$onlytwo",
        "sha256$$abc",
        "sha256$deadbeef$",
        "$argon2id$v=19$m=65536,t=3,p=4$c2FsdA$aGFzaA",
        "blake2b$deadbeef$abc",
    ],
)
def test_malformed_or_unknown_digest_never_verifies(digest):
    """Anything unparseable must not pass and must not raise."""
    assert not verify_secret_key_digest(digest, generate_secret_key())


@pytest.mark.parametrize(
    "digest",
    [
        None,
        "",
        "sha256$onlytwo",
        "sha256$$abc",
        "sha256$deadbeef$",
        "$argon2id$v=19$m=65536,t=3,p=4$c2FsdA$aGFzaA",
        "blake2b$deadbeef$abc",
    ],
)
def test_unusable_digest_is_reported_as_such(digest):
    """A value this build cannot check says nothing about the secret, so it must
    be distinguishable from a genuine mismatch -- otherwise bad column data would
    lock out a key whose argon2 hash is still perfectly good.
    """
    assert not secret_key_digest_usable(digest)


def test_a_real_digest_is_usable_whether_or_not_it_matches():
    digest = new_secret_key_digest(
        secret_key=generate_secret_key(),
        is_custom=False,
        access_key=generate_access_key(),
    )
    assert secret_key_digest_usable(digest)
    # Usable but not matching: an authoritative rejection, unlike the cases above.
    assert not verify_secret_key_digest(digest, generate_secret_key())


# --- Eligibility: getting this wrong stores a weak secret behind a fast hash ---


def test_custom_key_never_gets_a_digest():
    """``ApiKeyCreate.custom`` has no entropy requirement, so ``custom: "123456"``
    is a key that can exist.
    """
    assert (
        new_secret_key_digest(
            secret_key="123456", is_custom=True, access_key=generate_access_key()
        )
        is None
    )


def test_custom_key_shaped_like_a_generated_one_is_still_refused():
    """The shape test alone is not enough: an md5 of a dictionary word is exactly
    32 lowercase hex characters, and brute-forcing it is trivial.
    """
    md5_of_weak_word = hashlib.md5(b"123456").hexdigest()
    assert len(md5_of_weak_word) == GENERATED_SECRET_KEY_BYTES * 2
    assert (
        new_secret_key_digest(
            secret_key=md5_of_weak_word,
            is_custom=True,
            access_key=generate_access_key(),
        )
        is None
    )


def test_deployment_token_never_gets_a_digest():
    """The legacy cluster token stores the deployment-provided ``config.token``
    under an empty access key, and never goes through the custom flag — so the
    empty access key is the only thing standing between it and a fast hash.
    """
    assert (
        new_secret_key_digest(
            secret_key=generate_secret_key(), is_custom=False, access_key=""
        )
        is None
    )
    assert (
        new_secret_key_digest(
            secret_key="a-deployment-chosen-token", is_custom=False, access_key=""
        )
        is None
    )


@pytest.mark.parametrize(
    "secret_key",
    [
        "",
        "short",
        "gpustack_3192253c1f4a9b7e_c11c75ed6334ea9505da4ad9",
        "3192253c1f4a9b7e-c11c75ed6334ea95",  # right length, not hex
        "C11C75ED6334EA9505DA4AD9C11C75ED",  # uppercase hex is not our output
        "c11c75ed6334ea9505da4ad9c11c75e",  # one char short
        "c11c75ed6334ea9505da4ad9c11c75ed9",  # one char long
    ],
)
def test_secret_not_shaped_like_our_output_is_refused(secret_key):
    assert not secret_key_digest_eligible(
        secret_key=secret_key, is_custom=False, access_key=generate_access_key()
    )


def test_generated_secret_is_eligible():
    assert secret_key_digest_eligible(
        secret_key=generate_secret_key(),
        is_custom=False,
        access_key=generate_access_key(),
    )


# --- The argon2 fallback path keeps its memo ---


def test_repeated_argon2_verify_is_served_from_the_memo():
    """Custom keys and the legacy cluster token have no digest, so this path
    stays hot for them and a second verify must not recompute argon2.
    """
    secret = "a-custom-secret-that-stays-on-argon2"
    hashed = security.get_secret_hash(secret)
    verify_hashed_secret.cache_clear()

    assert verify_hashed_secret(hashed, secret)
    before = verify_hashed_secret.cache_info().hits
    assert verify_hashed_secret(hashed, secret)
    assert verify_hashed_secret.cache_info().hits == before + 1
    assert not verify_hashed_secret(hashed, "wrong-secret")


def test_verify_memo_is_bounded(monkeypatch):
    # A rejected argon2 verify costs as much as an accepted one, so stub the
    # hasher: this test is about the bound, not about argon2. ``PasswordHasher``
    # uses slots, so replace the instance rather than its method.
    class _StubHasher:
        def verify(self, hashed, plain):
            return False

    monkeypatch.setattr(security, "ph", _StubHasher())
    verify_hashed_secret.cache_clear()
    try:
        maxsize = verify_hashed_secret.cache_info().maxsize
        for index in range(maxsize + 5):
            verify_hashed_secret("stored-hash", f"secret-{index}")
        assert verify_hashed_secret.cache_info().currsize <= maxsize
    finally:
        # The memo is process-wide; leaving thousands of stubbed entries behind
        # would follow later tests around.
        verify_hashed_secret.cache_clear()
