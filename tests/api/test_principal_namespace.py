"""Unit tests for the per-kind principal name namespace.

USER / ORG / SYSTEM / GROUP each have their own ``name`` partition, so a
User and an Org may share a name. The behaviours that keep that safe are
that every USER-facing lookup is kind-scoped to USER — these tests pin
that scoping at the query-construction level (sessions are mocked, in
line with the rest of the suite; DB-level uniqueness is Postgres-only and
out of scope here).
"""

import logging
from unittest.mock import AsyncMock

import pytest

from gpustack.gpu_instances.cluster_apis_util import (
    get_namespace_name,
    principal_namespace_identifier,
)
from gpustack.logging import TRACE_LEVEL, trace
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.server import services
from gpustack.server.services import UserService

# ``get_by_username`` goes through the ``locked_cached`` layer, which logs
# at TRACE — a custom level only registered during app startup. Register
# it here so the cached call path doesn't blow up under the test runner.
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.Logger.trace = trace


@pytest.mark.asyncio
async def test_get_by_username_scopes_to_user_kind(monkeypatch):
    # A same-named ORG must never be returned by a login lookup, else it
    # would shadow a real user login. Assert the query is scoped to USER.
    one_by_fields = AsyncMock(return_value=None)
    monkeypatch.setattr(services.User, "one_by_fields", one_by_fields)

    # Unique name avoids the in-memory locked_cached cache returning a
    # value memoised by another test.
    result = await UserService(object()).get_by_username("namespace-probe-login-user")

    assert result is None
    one_by_fields.assert_awaited_once()
    _, fields = one_by_fields.await_args.args[:2]
    assert fields == {
        "name": "namespace-probe-login-user",
        "kind": PrincipalType.USER,
    }


def test_user_namespace_keys_off_id_not_name():
    # A USER's name is a login identifier (here an email) that is not a
    # valid k8s label and may collide with a same-named Org. The k8s
    # namespace must key off ``user-<id>`` instead.
    user = Principal(id=42, kind=PrincipalType.USER, name="alice@example.com")
    assert principal_namespace_identifier(user) == "user-42"
    assert get_namespace_name(principal_namespace_identifier(user)) == (
        "gpustack-user-42"
    )


def test_org_namespace_keys_off_name():
    org = Principal(id=7, kind=PrincipalType.ORG, name="acme")
    assert principal_namespace_identifier(org) == "acme"
    assert get_namespace_name(principal_namespace_identifier(org)) == "gpustack-acme"


def test_user_and_org_sharing_a_name_get_distinct_namespaces():
    # The whole point: User "acme" and Org "acme" coexist (per-kind name
    # partitions) and must not land in the same k8s namespace.
    user = Principal(id=5, kind=PrincipalType.USER, name="acme")
    org = Principal(id=9, kind=PrincipalType.ORG, name="acme")
    user_ns = get_namespace_name(principal_namespace_identifier(user))
    org_ns = get_namespace_name(principal_namespace_identifier(org))
    assert user_ns == "gpustack-user-5"
    assert org_ns == "gpustack-acme"
    assert user_ns != org_ns
