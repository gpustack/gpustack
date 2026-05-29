"""Unit tests for the unified ``/access`` principals surface.

Principal grants (any kind) are managed through ``POST /access`` with a
``principals`` list and read back via ``GET /access``; there is no longer
a separate ``/principals`` endpoint.
"""

from unittest.mock import AsyncMock, MagicMock
from typing import Optional

import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.routes import model_routes
from gpustack.schemas.model_routes import ModelPrincipalRef
from gpustack.schemas.principals import Principal, PrincipalType


def _principal(
    id: int = 5,
    kind: PrincipalType = PrincipalType.ORG,
    name: str = "principal",
    display_name: Optional[str] = None,
):
    p = MagicMock(spec=Principal)
    p.id = id
    p.kind = kind
    p.name = name
    p.display_name = display_name
    p.deleted_at = None
    return p


def _exec_returning(*results):
    queue = []
    for value in results:
        result = MagicMock()
        result.all = MagicMock(return_value=value if isinstance(value, list) else [])
        result.first = MagicMock(
            return_value=(
                (value[0] if value else None) if isinstance(value, list) else value
            )
        )
        queue.append(result)
    return AsyncMock(side_effect=queue)


def _session(*results):
    s = MagicMock()
    s.exec = _exec_returning(*results)
    s.commit = AsyncMock()
    s.rollback = AsyncMock()
    s.delete = AsyncMock()
    s.add = MagicMock()
    return s


# ---- list -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_route_principals_returns_grants():
    link = MagicMock()
    link.principal_id = 5
    # Two exec() calls: the grant rows, then the bulk principal lookup.
    session = MagicMock()
    session.exec = _exec_returning(
        [link], [_principal(id=5, kind=PrincipalType.ORG, name="acme")]
    )
    result = await model_routes._list_route_principals(session, 1)
    assert len(result) == 1
    assert result[0].principal_type == PrincipalType.ORG
    assert result[0].principal_id == 5
    assert result[0].principal_name == "acme"


@pytest.mark.asyncio
async def test_list_route_principals_empty():
    session = MagicMock()
    session.exec = _exec_returning([])
    assert await model_routes._list_route_principals(session, 1) == []


# ---- validate ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_principals_not_found(monkeypatch):
    monkeypatch.setattr(
        model_routes.Principal, "one_by_id", AsyncMock(return_value=None)
    )
    with pytest.raises(InvalidException):
        await model_routes._validate_principals(
            MagicMock(),
            [ModelPrincipalRef(principal_type=PrincipalType.ORG, principal_id=999)],
        )


@pytest.mark.asyncio
async def test_validate_principals_rejects_kind_mismatch(monkeypatch):
    # Caller declared GROUP, but the principal row is actually an ORG.
    monkeypatch.setattr(
        model_routes.Principal,
        "one_by_id",
        AsyncMock(return_value=_principal(id=5, kind=PrincipalType.ORG)),
    )
    with pytest.raises(InvalidException):
        await model_routes._validate_principals(
            MagicMock(),
            [ModelPrincipalRef(principal_type=PrincipalType.GROUP, principal_id=5)],
        )


@pytest.mark.asyncio
async def test_validate_principals_rejects_system(monkeypatch):
    # System actors live in kind=SYSTEM rows; requesting one as a USER
    # grant fails the kind check.
    monkeypatch.setattr(
        model_routes.Principal,
        "one_by_id",
        AsyncMock(return_value=_principal(id=2, kind=PrincipalType.SYSTEM)),
    )
    with pytest.raises(InvalidException):
        await model_routes._validate_principals(
            MagicMock(),
            [ModelPrincipalRef(principal_type=PrincipalType.USER, principal_id=2)],
        )


@pytest.mark.asyncio
async def test_validate_principals_ok(monkeypatch):
    monkeypatch.setattr(
        model_routes.Principal,
        "one_by_id",
        AsyncMock(return_value=_principal(id=5, kind=PrincipalType.ORG)),
    )
    # Should not raise.
    await model_routes._validate_principals(
        MagicMock(),
        [ModelPrincipalRef(principal_type=PrincipalType.ORG, principal_id=5)],
    )


# ---- replace ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_route_principals_adds_and_removes():
    keep = MagicMock()
    keep.principal_id = 5
    drop = MagicMock()
    drop.principal_id = 6
    session = _session([keep, drop])  # existing grants

    # Desired set keeps 5, drops 6, adds 7.
    await model_routes._replace_route_principals(session, 1, [5, 7])

    session.delete.assert_awaited_once_with(drop)
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_replace_route_principals_empty_clears_all():
    a = MagicMock()
    a.principal_id = 5
    b = MagicMock()
    b.principal_id = 6
    session = _session([a, b])

    await model_routes._replace_route_principals(session, 1, [])

    assert session.delete.await_count == 2
    session.add.assert_not_called()
