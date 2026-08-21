from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.tenant import TenantContext
from gpustack.routes import inference_backend as ib_route
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    InferenceBackendPublic,
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


def _platform_admin_ctx() -> TenantContext:
    """Platform admin in "All" mode — sees Platform (NULL-owner) rows only."""
    user = Principal(id=1, name="admin", kind=PrincipalType.USER, is_admin=True)
    return TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
        current_is_personal_scope=False,
    )


def _items_by_name(response):
    return {item.backend_name: item for item in response.items}


@pytest.mark.asyncio
async def test_list_hides_disabled_versions_and_backend_for_custom(monkeypatch):
    """/list drops versions listed in disabled_versions, drops a disabled
    community backend entirely, and reroutes a default that got hidden."""
    rows = [
        InferenceBackend(
            id=1,
            backend_name="cust-a",
            owner_principal_id=None,
            backend_source=BackendSourceEnum.CUSTOM,
            enabled=True,
            default_version="v2",
            disabled_versions=["v1"],
            version_configs=VersionConfigDict(
                root={
                    "v1": VersionConfig(image_name="img:v1"),
                    "v2": VersionConfig(image_name="img:v2"),
                }
            ),
        ),
        InferenceBackend(
            id=2,
            backend_name="comm-off",
            owner_principal_id=None,
            backend_source=BackendSourceEnum.COMMUNITY,
            enabled=False,
            version_configs=VersionConfigDict(
                root={"v1": VersionConfig(image_name="img:v1")}
            ),
        ),
        InferenceBackend(
            id=3,
            backend_name="cust-b",
            owner_principal_id=None,
            backend_source=BackendSourceEnum.CUSTOM,
            enabled=True,
            default_version="v1",
            disabled_versions=["v1"],
            version_configs=VersionConfigDict(
                root={
                    "v1": VersionConfig(image_name="img:v1"),
                    "v2": VersionConfig(image_name="img:v2"),
                }
            ),
        ),
        InferenceBackend(
            id=4,
            backend_name="cust-off",
            owner_principal_id=None,
            backend_source=BackendSourceEnum.CUSTOM,
            enabled=False,
            version_configs=VersionConfigDict(
                root={"v1": VersionConfig(image_name="img:v1")}
            ),
        ),
    ]
    monkeypatch.setattr(ib_route.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(ib_route.InferenceBackend, "all", AsyncMock(return_value=rows))

    response = await ib_route.list_backend_configs(MagicMock(), _platform_admin_ctx())
    items = _items_by_name(response)

    assert "comm-off" not in items  # disabled community backend hidden
    assert "cust-off" not in items  # disabled custom backend hidden too
    assert [v.version for v in items["cust-a"].versions] == ["v2"]
    # default_version "v1" was disabled on cust-b → falls back to a live one
    assert [v.version for v in items["cust-b"].versions] == ["v2"]
    assert items["cust-b"].default_version == "v2"


@pytest.mark.asyncio
async def test_list_hides_disabled_builtin_and_versions_with_default_fallback(
    monkeypatch,
):
    """/list skips a built-in whose enabled is False, removes disabled runner
    versions from a live built-in, and reroutes its hidden default."""
    rows = [
        InferenceBackend(
            id=1,
            backend_name="vllm",
            owner_principal_id=None,
            is_built_in=True,
            enabled=None,  # NULL = enabled (backward compatible)
            default_version="v1",
            disabled_versions=["v1"],
        ),
        InferenceBackend(
            id=2,
            backend_name="sglang",
            owner_principal_id=None,
            is_built_in=True,
            enabled=False,  # explicitly disabled built-in
        ),
    ]

    def fake_merge(backend_name, workers):
        version_configs = VersionConfigDict(
            root={
                "v1": VersionConfig(built_in_frameworks=["vllm"]),
                "v2": VersionConfig(built_in_frameworks=["vllm"]),
                "v3": VersionConfig(built_in_frameworks=["vllm"]),
            }
        )
        return {}, version_configs, "v1"

    monkeypatch.setattr(ib_route.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(ib_route.InferenceBackend, "all", AsyncMock(return_value=rows))
    monkeypatch.setattr(ib_route, "merge_list_runners", fake_merge)

    response = await ib_route.list_backend_configs(MagicMock(), _platform_admin_ctx())
    items = _items_by_name(response)

    assert "sglang" not in items  # disabled built-in hidden entirely
    assert [v.version for v in items["vllm"].versions] == ["v2", "v3"]
    assert items["vllm"].default_version == "v2"  # hidden "v1" default rerouted


@pytest.mark.asyncio
async def test_list_unions_platform_and_org_disabled_versions(monkeypatch):
    """A version disabled at either the Platform or the Org layer stays hidden
    (union merge), consistent with merged version_configs."""
    platform = InferenceBackend(
        id=1,
        backend_name="shared",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.CUSTOM,
        enabled=True,
        disabled_versions=["v1"],
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1"),
                "v2": VersionConfig(image_name="img:v2"),
                "v3": VersionConfig(image_name="img:v3"),
            }
        ),
    )
    org = InferenceBackend(
        id=2,
        backend_name="shared",
        owner_principal_id=42,
        backend_source=BackendSourceEnum.CUSTOM,
        enabled=True,
        disabled_versions=["v2"],
        version_configs=VersionConfigDict(root={}),
    )
    monkeypatch.setattr(ib_route.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        ib_route.InferenceBackend, "all", AsyncMock(return_value=[platform, org])
    )

    response = await ib_route.list_backend_configs(MagicMock(), _org_ctx(42))
    items = _items_by_name(response)

    assert [v.version for v in items["shared"].versions] == ["v3"]


@pytest.mark.asyncio
async def test_list_all_versions_disabled_leaves_empty_and_no_default(monkeypatch):
    """Disabling every version of an (still-enabled) backend leaves the card
    with no versions and clears its default rather than pointing at a hidden
    one — the whole-backend axis is enabled/False, kept separate."""
    rows = [
        InferenceBackend(
            id=1,
            backend_name="cust-a",
            owner_principal_id=None,
            backend_source=BackendSourceEnum.CUSTOM,
            enabled=True,
            default_version="v1",
            disabled_versions=["v1", "v2"],
            version_configs=VersionConfigDict(
                root={
                    "v1": VersionConfig(image_name="img:v1"),
                    "v2": VersionConfig(image_name="img:v2"),
                }
            ),
        ),
    ]
    monkeypatch.setattr(ib_route.Worker, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(ib_route.InferenceBackend, "all", AsyncMock(return_value=rows))

    response = await ib_route.list_backend_configs(MagicMock(), _platform_admin_ctx())
    items = _items_by_name(response)

    assert items["cust-a"].versions == []
    assert items["cust-a"].default_version is None


@pytest.mark.asyncio
async def test_update_persists_enabled_and_disabled_versions_for_builtin(monkeypatch):
    """PUT of a built-in must persist both the enabled toggle and
    disabled_versions. They used to be dropped by the update whitelist
    (enabled was community-only; disabled_versions absent), so a built-in
    could never be hidden."""
    backend = InferenceBackend(
        id=1,
        backend_name="vllm",
        owner_principal_id=None,
        is_built_in=True,
        backend_source=BackendSourceEnum.BUILT_IN,
        enabled=True,
        version_configs=VersionConfigDict(root={}),
    )
    backend_in = InferenceBackendUpdate(
        backend_name="vllm",
        backend_source=BackendSourceEnum.BUILT_IN,
        enabled=False,
        disabled_versions=["0.10.2"],
        version_configs=VersionConfigDict(root={}),
    )

    monkeypatch.setattr(
        ib_route.InferenceBackend, "one_by_id", AsyncMock(return_value=backend)
    )
    monkeypatch.setattr(
        ib_route, "_redirect_global_edit_to_org_row", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    monkeypatch.setattr(ib_route, "_validate_version_removal", AsyncMock())
    monkeypatch.setattr(ib_route, "validate_custom_suffix", MagicMock())

    captured = {}

    async def fake_update(self, session, data):
        captured.update(data)
        return self

    monkeypatch.setattr(ib_route.InferenceBackend, "update", fake_update)

    await ib_route.update_inference_backend(
        MagicMock(), _platform_admin_ctx(), 1, backend_in
    )

    assert captured.get("enabled") is False
    assert captured.get("disabled_versions") == ["0.10.2"]


@pytest.mark.asyncio
async def test_update_builtin_ignores_round_tripped_default_version(monkeypatch):
    """The management GET enriches built-in rows with a catalog-managed
    default_version, so a client editing the row round-trips it back on PUT.
    That must not 400 (default_version is auto-managed), and must not be
    persisted onto the DB row — it stays NULL so the catalog remains the
    source of truth. This is what lets a built-in be hidden via the UI."""
    backend = InferenceBackend(
        id=1,
        backend_name="vllm",
        owner_principal_id=None,
        is_built_in=True,
        backend_source=BackendSourceEnum.BUILT_IN,
        enabled=True,
        version_configs=VersionConfigDict(root={}),
    )
    backend_in = InferenceBackendUpdate(
        backend_name="vllm",
        backend_source=BackendSourceEnum.BUILT_IN,
        default_version="0.10.2",  # echoed back from the enriched GET response
        disabled_versions=["0.10.1"],
        version_configs=VersionConfigDict(root={}),
    )

    monkeypatch.setattr(
        ib_route.InferenceBackend, "one_by_id", AsyncMock(return_value=backend)
    )
    monkeypatch.setattr(
        ib_route, "_redirect_global_edit_to_org_row", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    monkeypatch.setattr(ib_route, "_validate_version_removal", AsyncMock())
    monkeypatch.setattr(ib_route, "validate_custom_suffix", MagicMock())

    captured = {}

    async def fake_update(self, session, data):
        captured.update(data)
        return self

    monkeypatch.setattr(ib_route.InferenceBackend, "update", fake_update)

    # Must not raise for the round-tripped default_version.
    await ib_route.update_inference_backend(
        MagicMock(), _platform_admin_ctx(), 1, backend_in
    )

    assert "default_version" not in captured
    assert captured.get("disabled_versions") == ["0.10.1"]


def test_management_public_model_exposes_disabled_versions():
    """The management endpoint's response model carries disabled_versions so
    admins can see and re-enable hidden versions."""
    assert "disabled_versions" in InferenceBackendPublic.model_fields


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
