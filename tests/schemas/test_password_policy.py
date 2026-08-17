"""Password policy on the user DTOs.

The policy has to stay in step with the check the UI runs client-side. Anything
the UI accepts and the server refuses reaches the user as a 422 on a password
the form told them was fine.
"""

import inspect

import pytest
from pydantic import BaseModel, ValidationError

import gpustack.schemas.users as users_schemas
from gpustack.schemas.users import (
    UpdatePassword,
    UserCreate,
    UserSelfUpdate,
    UserUpdate,
)
from gpustack.security import (
    PASSWORD_SPECIAL_CHARACTERS,
    generate_secure_password,
)

PASSWORD_FIELD_NAMES = {"password", "new_password"}

# A value that trips the first rule in the policy, whatever else changes.
VIOLATING_PASSWORD = "a"

# DTOs whose password field doubles as "leave the password alone".
OPTIONAL_PASSWORD_DTOS = [UserCreate, UserUpdate, UserSelfUpdate]


def create_with(password):
    return UserCreate(username="alice", password=password)


def build(dto, **kwargs):
    """Construct ``dto`` with only the fields it actually declares.

    ``UserSelfUpdate`` has no ``username``, and pydantic drops unknown keys
    silently, so passing one everywhere would quietly test something else.
    """
    accepted = {key: value for key, value in kwargs.items() if key in dto.model_fields}
    return dto(**accepted)


@pytest.mark.parametrize("special", list(PASSWORD_SPECIAL_CHARACTERS))
def test_accepts_every_special_character_the_ui_offers(special):
    # Pins the set to what the UI's regex offers, character by character.
    password = f"Aa1x{special}y"
    assert create_with(password).password == password


def test_length_is_not_part_of_the_policy():
    # The server is the authority and does not cap length; the UI's own 6-64
    # bound is a client-side courtesy. NIST SP 800-63B asks for at least 64
    # characters to be accepted, and argon2 hashes the whole string -- there is
    # no bcrypt-style truncation to defend against.
    short = "Aa1!"
    long = "Aa1!" + "a" * 200
    assert create_with(short).password == short
    assert create_with(long).password == long


@pytest.mark.parametrize(
    "password, expected",
    [
        ("passw0rd.", "uppercase"),
        ("PASSW0RD.", "lowercase"),
        ("Password.", "digit"),
        ("Passw0rdx", "special character"),
        # Outside the set the UI accepts, so it stays rejected on purpose --
        # accepting it here would just move the mismatch to the other side.
        ("Passw0rd-", "special character"),
    ],
)
def test_rejects(password, expected):
    with pytest.raises(ValidationError, match=expected):
        create_with(password)


def test_error_message_names_the_accepted_characters():
    # "must contain at least one special character" reads as a lie to someone
    # whose password does contain one, just not from this set — so the message
    # has to spell the set out.
    with pytest.raises(ValidationError) as excinfo:
        create_with("Passw0rdx")
    for special in PASSWORD_SPECIAL_CHARACTERS:
        assert special in str(excinfo.value)


@pytest.mark.parametrize("dto", OPTIONAL_PASSWORD_DTOS)
def test_optional_password_dtos_enforce_the_policy(dto):
    # All three reach `set_password`, so the policy has to hold on every one:
    # a route that skips it becomes a way to set a password the create route
    # would refuse.
    with pytest.raises(ValidationError, match="uppercase"):
        build(dto, username="alice", password=VIOLATING_PASSWORD)


@pytest.mark.parametrize("dto", OPTIONAL_PASSWORD_DTOS)
@pytest.mark.parametrize("absent", [None, ""])
def test_absent_password_leaves_the_password_alone(dto, absent):
    # The write paths act on `if user_in.password`, so an empty string has to
    # arrive as None rather than as a policy violation.
    assert build(dto, username="alice", password=absent).password is None


def test_update_password_enforces_the_policy():
    with pytest.raises(ValidationError, match="uppercase"):
        UpdatePassword(current_password="whatever", new_password=VIOLATING_PASSWORD)


def test_update_password_rejects_an_empty_new_password():
    # `new_password` is required and names the thing being set, so blank is a
    # violation here rather than "no change".
    with pytest.raises(ValidationError, match="uppercase"):
        UpdatePassword(current_password="whatever", new_password="")


def _password_dtos():
    for name, cls in vars(users_schemas).items():
        if not (inspect.isclass(cls) and issubclass(cls, BaseModel)):
            continue
        fields = set(getattr(cls, "model_fields", {})) & PASSWORD_FIELD_NAMES
        if fields:
            yield name, cls, sorted(fields)


def test_every_password_field_on_the_api_surface_is_validated():
    """Any DTO carrying a password on the API surface must enforce the policy.

    Asserted behaviourally rather than by looking for a validator: a validator
    that accepts everything would satisfy the declaration and nothing else.

    The check covers construction, which is where request bodies are parsed.
    Assigning to the attribute afterwards does not run validators -- e.g.
    ``cmd/reset_admin_password.py`` sets ``user_update.password`` on an already
    built model -- so a client can skip the policy locally and the server-side
    parse of the request body is what actually holds the line.
    """
    unguarded = {}
    for name, cls, fields in _password_dtos():
        for field in fields:
            kwargs = {
                other: "x"
                for other, spec in cls.model_fields.items()
                if spec.is_required() and other not in PASSWORD_FIELD_NAMES
            }
            kwargs[field] = VIOLATING_PASSWORD
            try:
                cls(**kwargs)
            except ValidationError:
                continue
            unguarded.setdefault(name, []).append(field)

    assert (
        not unguarded
    ), f"password fields that accept {VIOLATING_PASSWORD!r}: {unguarded}"


def test_the_guard_above_covers_the_known_dtos():
    # Guards the guard: a rename that makes _password_dtos() yield nothing
    # would leave the scan vacuously green.
    assert {name for name, _, _ in _password_dtos()} == {
        "UserCreate",
        "UserUpdate",
        "UserSelfUpdate",
        "UpdatePassword",
    }


def test_generated_passwords_satisfy_the_policy():
    # `gpustack reset-admin-password` submits a generated password through
    # UserUpdate, so a generator that drifts from the validator turns that
    # command into a 422.
    for _ in range(50):
        password = generate_secure_password()
        assert create_with(password).password == password
