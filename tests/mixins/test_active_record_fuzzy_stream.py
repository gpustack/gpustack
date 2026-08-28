"""A NULL column must not match the watch stream's fuzzy predicate.

``_match_fuzzy_fields`` is the Python twin of the ``fuzzy_fields`` LIKE the read
builds, and it read ``str(getattr(event.data, key, "")).lower()`` — so a NULL
column yielded the string ``"none"``. Confirmed against PostgreSQL 17: with a
nullable ``display_name`` in the dict, ``search=none`` returns ``[]`` from the
read while the stream accepts every row whose display name is NULL. A user
searching it sees an empty list, then rows appearing in it one by one as they
change.

SQL is the reference: ``NULL LIKE '%none%'`` is NULL, never true. So the key is
skipped when the attribute is None, which can only narrow the stream toward what
the read returns.

``GPUInstanceSSHPublicKey`` stands in for the family — it is one of the routes
that passes a nullable ``display_name`` — but the fix is in the mixin, so every
route that passes ``fuzzy_fields`` inherits it.
"""

from gpustack.schemas.gpu_instance_ssh_public_keys import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeySpec,
)
from gpustack.server.bus import Event, EventType

BOTH_NAMES = {"name": "none", "display_name": "none"}


def _event(name, display_name):
    return Event(
        type=EventType.UPDATED,
        data=GPUInstanceSSHPublicKey(
            owner_principal_id=1,
            name=name,
            display_name=display_name,
            spec=GPUInstanceSSHPublicKeySpec(data="ssh-ed25519 AAAA"),
        ),
    )


def test_a_null_display_name_does_not_match_the_word_none():
    # The regression: str(None) is "none", so this returned True while the read
    # returned no rows at all.
    event = _event("res-000002", None)
    assert GPUInstanceSSHPublicKey._match_fuzzy_fields(event, BOTH_NAMES) is False


def test_a_null_display_name_still_matches_by_name():
    # Positive control — skipping the NULL key must not drop the other arm.
    event = _event("res-000002", None)
    assert (
        GPUInstanceSSHPublicKey._match_fuzzy_fields(
            event, {"name": "000002", "display_name": "000002"}
        )
        is True
    )


def test_a_set_display_name_still_matches():
    event = _event("res-7f3a91", "Team A Pool")
    assert (
        GPUInstanceSSHPublicKey._match_fuzzy_fields(
            event, {"name": "team a", "display_name": "team a"}
        )
        is True
    )


def test_an_empty_display_name_matches_nothing_but_an_empty_term():
    # An empty string is a value, not an absence: it stays a literal match, as
    # SQL's ``'' LIKE '%none%'`` does.
    event = _event("res-000003", "")
    assert GPUInstanceSSHPublicKey._match_fuzzy_fields(event, BOTH_NAMES) is False


def test_an_id_only_payload_is_rejected():
    # A DELETED event can carry an id-only dict. Every key resolves to None, so
    # nothing matches — and nothing raises.
    event = Event(type=EventType.DELETED, data={"id": 7})
    assert GPUInstanceSSHPublicKey._match_fuzzy_fields(event, BOTH_NAMES) is False


def test_no_fuzzy_fields_accepts_everything():
    # The unsearched stream: an empty filter is not a filter.
    event = _event("res-000002", None)
    assert GPUInstanceSSHPublicKey._match_fuzzy_fields(event, None) is True
    assert GPUInstanceSSHPublicKey._match_fuzzy_fields(event, {}) is True
