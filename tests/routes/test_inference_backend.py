from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.tenant import TenantContext
from gpustack.routes import inference_backend as ib_route
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    InferenceBackendUpdate,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendSourceEnum
from gpustack.schemas.principals import Principal, PrincipalType


def _org_ctx(current_principal_id: int) -> TenantContext:
    """An Org-scoped context (member or admin acting-as an Org)."""
    user = Principal(id=1, name="u1", kind=PrincipalType.USER, is_admin=False)
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=current_principal_id,
        org_role=None,
        current_is_personal_scope=False,
    )


def _platform_community_backend() -> InferenceBackend:
    """A Platform (owner_principal_id=None) community row that is disabled
    but carries the catalog versions, tagged with built_in_frameworks."""
    return InferenceBackend(
        id=10,
        backend_name="my-community",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=False,
        default_version="v1",
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1", built_in_frameworks=["vllm"]),
                "v2": VersionConfig(image_name="img:v2", built_in_frameworks=["vllm"]),
            }
        ),
    )


@pytest.mark.asyncio
async def test_redirect_seeds_org_row_with_platform_versions_on_bare_enable(
    monkeypatch,
):
    """Bare "enable" of a community backend in an Org scope carries no
    versions in the payload. The redirected Org row must inherit the
    Platform catalog versions so its version isn't blank in the
    uncollapsed "All" view."""
    platform = _platform_community_backend()
    # A bare enable: the community catalog endpoint clears version_configs,
    # so the payload arrives with none.
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        version_configs=VersionConfigDict(root={}),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_fields", AsyncMock(return_value=None))
    monkeypatch.setattr(
        InferenceBackend,
        "create",
        AsyncMock(side_effect=lambda session, obj: obj),
    )

    created = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert created.owner_principal_id == 42
    assert created.backend_source == BackendSourceEnum.COMMUNITY
    assert set(created.version_configs.root.keys()) == {"v1", "v2"}
    assert created.version_configs.root["v1"].image_name == "img:v1"

    # The seed is a deep copy: mutating the Org row must not leak back
    # into the Platform row it inherited from.
    created.version_configs.root["v1"].image_name = "mutated"
    assert platform.version_configs.root["v1"].image_name == "img:v1"


@pytest.mark.asyncio
async def test_redirect_payload_versions_override_platform_versions(monkeypatch):
    """When the payload does carry versions, they layer on top of the
    inherited Platform versions (Org wins on key collisions)."""
    platform = _platform_community_backend()
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="org-override:v1"),
                "v3": VersionConfig(image_name="img:v3"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_fields", AsyncMock(return_value=None))
    monkeypatch.setattr(
        InferenceBackend,
        "create",
        AsyncMock(side_effect=lambda session, obj: obj),
    )

    created = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert set(created.version_configs.root.keys()) == {"v1", "v2", "v3"}
    assert created.version_configs.root["v1"].image_name == "org-override:v1"
    assert created.version_configs.root["v2"].image_name == "img:v2"


@pytest.mark.asyncio
async def test_redirect_returns_existing_org_row(monkeypatch):
    """If the Org already has its own row, the write targets that row and
    no new row is created."""
    platform = _platform_community_backend()
    existing = InferenceBackend(
        id=11,
        backend_name="my-community",
        owner_principal_id=42,
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
    )
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
    )

    monkeypatch.setattr(
        InferenceBackend, "one_by_fields", AsyncMock(return_value=existing)
    )
    create_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "create", create_mock)

    result = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert result is existing
    create_mock.assert_not_called()


@pytest.mark.asyncio
async def test_redirect_noop_in_all_mode(monkeypatch):
    """Platform admin in "All" mode (no current_principal_id) edits the
    Platform row directly — no redirect."""
    platform = _platform_community_backend()
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
    )
    ctx = _org_ctx(42)
    ctx.current_principal_id = None

    result = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), ctx, platform, backend_in
    )

    assert result is None
