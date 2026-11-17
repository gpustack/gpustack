from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import BadRequestException, InternalServerErrorException
from gpustack.api.tenant import TenantContext
from gpustack.routes import inference_backend as ib_route
from gpustack.schemas import Worker
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    InferenceBackendUpdate,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendSourceEnum, Model
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.runner_source import RunnerOverrideEntry


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
        AsyncMock(side_effect=lambda session, obj, **kwargs: obj),
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
async def test_redirect_seed_is_flushed_without_commit(monkeypatch):
    """The seeded Org row is handed to create uncommitted: the caller
    validates the payload against it first and commits it together with the
    update, so a rejected request rolls the seed back instead of leaving a
    half-seeded row behind."""
    platform = _platform_community_backend()
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
    )

    monkeypatch.setattr(InferenceBackend, "one_by_fields", AsyncMock(return_value=None))
    create_mock = AsyncMock(side_effect=lambda session, obj, **kwargs: obj)
    monkeypatch.setattr(InferenceBackend, "create", create_mock)

    created = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert created is not None
    assert create_mock.await_args.kwargs["auto_commit"] is False


@pytest.mark.asyncio
async def test_redirect_seed_keeps_stored_version_on_collision(monkeypatch):
    """A payload entry colliding with a catalog version never wins the seed:
    the seed keeps the stored entry, matching the update merge — otherwise a
    client could plant content under a restored catalog marker and no API
    path could ever correct it."""
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
        AsyncMock(side_effect=lambda session, obj, **kwargs: obj),
    )

    created = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert set(created.version_configs.root.keys()) == {"v1", "v2", "v3"}
    assert created.version_configs.root["v1"].image_name == "img:v1"
    assert created.version_configs.root["v1"].built_in_frameworks == ["vllm"]
    assert created.version_configs.root["v2"].image_name == "img:v2"
    assert created.version_configs.root["v3"].image_name == "img:v3"
    assert created.version_configs.root["v3"].built_in_frameworks is None


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


def _all_mode_admin_ctx() -> TenantContext:
    """Platform admin in "All" mode — the only caller that may write a Global row."""
    user = Principal(id=1, name="admin", kind=PrincipalType.USER, is_admin=True)
    return TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
        current_is_personal_scope=False,
    )


def _downgraded_backend(source_name) -> InferenceBackend:
    """A community row no source publishes any more: reconcile_backend converted
    it to CUSTOM, so it keeps the source-given name — which carries no
    ``-custom`` suffix — and its versions look hand-added."""
    return InferenceBackend(
        id=12,
        backend_name="llama.cpp",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.CUSTOM,
        enabled=False,
        source_name=source_name,
        version_configs=VersionConfigDict(
            root={"v1": VersionConfig(image_name="img:v1", built_in_frameworks=None)}
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source_name, accepted", [("builtin", True), (None, False)])
async def test_a_downgraded_row_stays_writable_under_its_source_given_name(
    monkeypatch, source_name, accepted
):
    """A row downgraded from a source must stay editable: holding its
    source-given name to the ``-custom`` suffix would reject every write to it.
    A row carrying no source stamp is still held to the suffix."""
    backend = _downgraded_backend(source_name)
    backend_in = InferenceBackendUpdate(
        backend_name="llama.cpp",
        backend_source=BackendSourceEnum.CUSTOM,
        description="edited",
        version_configs=backend.version_configs,
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    if accepted:
        await ib_route.update_inference_backend(
            MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
        )
        assert update_mock.await_args.args[1]["description"] == "edited"
    else:
        with pytest.raises(BadRequestException) as e:
            await ib_route.update_inference_backend(
                MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
            )
        assert "must end with '-custom'" in str(e.value.message)
        update_mock.assert_not_called()


def _community_backend_with_custom_version(backend_id: int) -> InferenceBackend:
    """A Platform community row: catalog versions tagged with
    built_in_frameworks (one declaring no framework, which normalizes to
    the empty list) plus one user-defined '-custom' version."""
    return InferenceBackend(
        id=backend_id,
        backend_name="my-community",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        default_version="v1",
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1", built_in_frameworks=["vllm"]),
                "v2": VersionConfig(image_name="img:v2", built_in_frameworks=["vllm"]),
                "v3": VersionConfig(image_name="img:v3", built_in_frameworks=[]),
                "v1-custom": VersionConfig(image_name="custom:v1"),
            }
        ),
    )


@pytest.mark.asyncio
async def test_update_with_echoed_get_body_preserves_built_in_versions(monkeypatch):
    """GET /v2/inference-backends/{id} returns built-in versions carrying
    built_in_frameworks. A client that edits other fields and PUTs that body
    back must not strip them: the stored built-in entries survive the merge
    so models pinned to those versions stay schedulable, while user-defined
    versions still take the payload's updates."""
    backend = _community_backend_with_custom_version(backend_id=44)
    # The GET body: built-in versions echoed with their marker, the
    # user-defined version edited, a new user-defined version added.
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        default_version="v1",
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1", built_in_frameworks=["vllm"]),
                "v2": VersionConfig(image_name="img:v2", built_in_frameworks=["vllm"]),
                "v3": VersionConfig(image_name="img:v3", built_in_frameworks=[]),
                "v1-custom": VersionConfig(image_name="custom:v2"),
                "v3-custom": VersionConfig(image_name="custom:v3"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    merged = update_mock.await_args.args[1]["version_configs"].root
    assert set(merged.keys()) == {"v1", "v2", "v3", "v1-custom", "v3-custom"}
    assert merged["v1"].built_in_frameworks == ["vllm"]
    assert merged["v1"].image_name == "img:v1"
    assert merged["v2"].built_in_frameworks == ["vllm"]
    assert merged["v3"].built_in_frameworks == []
    assert merged["v3"].image_name == "img:v3"
    assert merged["v1-custom"].image_name == "custom:v2"
    assert merged["v3-custom"].image_name == "custom:v3"
    assert update_mock.await_args.args[1]["enabled"] is True
    assert update_mock.await_args.args[1]["default_version"] == "v1"


@pytest.mark.asyncio
async def test_update_without_version_configs_leaves_versions_untouched(monkeypatch):
    """A bare enable/disable carries no version_configs: the stored versions
    must survive untouched — an absent field must not read as "replace with
    nothing" and silently delete every undeployed user-defined version."""
    backend = _community_backend_with_custom_version(backend_id=51)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    in_use_mock = AsyncMock(return_value=(False, []))
    monkeypatch.setattr(ib_route, "check_backend_in_use", in_use_mock)
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    update_data = update_mock.await_args.args[1]
    assert "version_configs" not in update_data
    assert update_data["enabled"] is True
    # No version was touched, so the in-use guard had nothing to check.
    in_use_mock.assert_not_called()


@pytest.mark.asyncio
async def test_update_uses_stored_source_when_payload_omits_it(monkeypatch):
    """A PUT that echoes the GET body without backend_source (or with a
    different one) must not skip the merge or demote the row: the merge gate
    and the stored source both come from the row, not the payload."""
    backend = _community_backend_with_custom_version(backend_id=52)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        enabled=True,
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1", built_in_frameworks=["vllm"]),
                "v2": VersionConfig(image_name="img:v2", built_in_frameworks=["vllm"]),
                "v3": VersionConfig(image_name="img:v3", built_in_frameworks=[]),
                "v1-custom": VersionConfig(image_name="custom:v1"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    update_data = update_mock.await_args.args[1]
    assert update_data["backend_source"] == BackendSourceEnum.COMMUNITY
    merged = update_data["version_configs"].root
    assert set(merged.keys()) == {"v1", "v2", "v3", "v1-custom"}
    assert merged["v1"].built_in_frameworks == ["vllm"]


@pytest.mark.asyncio
async def test_update_rejects_modified_source_version(monkeypatch):
    """Editing a source-owned version must return 400, not 200 with the edit
    silently discarded by the merge."""
    backend = _community_backend_with_custom_version(backend_id=53)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="evil:v1", built_in_frameworks=["vllm"]),
                "v1-custom": VersionConfig(image_name="custom:v1"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    monkeypatch.setattr(
        ib_route, "check_backend_in_use", AsyncMock(return_value=(False, []))
    )
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    with pytest.raises(BadRequestException) as e:
        await ib_route.update_inference_backend(
            MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
        )

    assert "managed by the backend source" in str(e.value.message)
    update_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_accepts_identical_source_version_echo(monkeypatch):
    """Echoing a source-owned version back unchanged is the round trip the
    merge is built to support: it passes the edit check and the stored entry
    survives."""
    backend = _community_backend_with_custom_version(backend_id=54)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(image_name="img:v1", built_in_frameworks=["vllm"]),
                "v1-custom": VersionConfig(image_name="custom:v1"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    merged = update_mock.await_args.args[1]["version_configs"].root
    assert set(merged.keys()) == {"v1", "v2", "v3", "v1-custom"}
    assert merged["v1"].image_name == "img:v1"
    assert merged["v1"].built_in_frameworks == ["vllm"]


@pytest.mark.asyncio
async def test_update_accepts_runner_version_echo_on_built_in_backend(monkeypatch):
    """Built-in backends keep their runner versions out of version_configs
    (the list response layers them in built_in_version_configs). A payload
    that echoes those runner versions back — carrying markers for keys the
    row does not own — must succeed, not 400."""
    backend = InferenceBackend(
        id=55,
        backend_name="vllm",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.BUILT_IN,
        enabled=True,
        version_configs=VersionConfigDict(root={}),
    )
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.BUILT_IN,
        enabled=True,
        description="edited",
        version_configs=VersionConfigDict(
            root={
                "0.8.5": VersionConfig(
                    image_name="runner:v0.8.5", built_in_frameworks=["cuda"]
                ),
                "0.9.0": VersionConfig(
                    image_name="runner:v0.9.0", built_in_frameworks=["cuda"]
                ),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    update_data = update_mock.await_args.args[1]
    assert update_data["description"] == "edited"
    assert set(update_data["version_configs"].root.keys()) == set()
    assert update_data["backend_source"] == BackendSourceEnum.BUILT_IN


@pytest.mark.asyncio
async def test_update_from_yaml_rejects_modified_source_version(monkeypatch):
    """Editing a source-owned version in the YAML editor must return 400.
    Before the ownership rules read the stored row this produced a 400 for a
    different reason (marker-stripped names failing the suffix rule); the
    rejection itself is the contract."""
    backend = _community_backend_with_custom_version(backend_id=56)
    yaml_content = (
        'backend_name: "my-community"\n'
        'version_configs:\n'
        '  "v1":\n'
        '    image_name: "evil:v1"\n'
        '    built_in_frameworks: ["vllm"]\n'
        '  "v1-custom":\n'
        '    image_name: "custom:v1"\n'
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    monkeypatch.setattr(
        ib_route, "check_backend_in_use", AsyncMock(return_value=(False, []))
    )
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    with pytest.raises(BadRequestException) as e:
        await ib_route.update_inference_backend_from_yaml(
            MagicMock(), _all_mode_admin_ctx(), backend.id, {"content": yaml_content}
        )

    assert "managed by the backend source" in str(e.value.message)
    update_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_from_yaml_accepts_identical_source_version_echo(monkeypatch):
    """A YAML round trip that echoes the source-owned versions unchanged and
    edits only user-defined versions must succeed and keep the stored
    entries — this is the flow that used to 400 on every save."""
    backend = _community_backend_with_custom_version(backend_id=57)
    yaml_content = (
        'backend_name: "my-community"\n'
        'version_configs:\n'
        '  "v1":\n'
        '    image_name: "img:v1"\n'
        '    built_in_frameworks: ["vllm"]\n'
        '  "v2":\n'
        '    image_name: "img:v2"\n'
        '    built_in_frameworks: ["vllm"]\n'
        '  "v3":\n'
        '    image_name: "img:v3"\n'
        '    built_in_frameworks: []\n'
        '  "v1-custom":\n'
        '    image_name: "custom:v2"\n'
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend_from_yaml(
        MagicMock(), _all_mode_admin_ctx(), backend.id, {"content": yaml_content}
    )

    merged = update_mock.await_args.args[1]["version_configs"].root
    assert set(merged.keys()) == {"v1", "v2", "v3", "v1-custom"}
    assert merged["v1"].image_name == "img:v1"
    assert merged["v1"].built_in_frameworks == ["vllm"]
    assert merged["v3"].built_in_frameworks == []
    assert merged["v1-custom"].image_name == "custom:v2"


@pytest.mark.asyncio
async def test_seed_not_committed_when_validation_rejects_payload(monkeypatch):
    """An Org PUT to the Platform row with an invalid version must not leave
    the freshly seeded Org row behind: the seed is flushed uncommitted and
    the validation failure surfaces before anything is persisted."""
    platform = _platform_community_backend()
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        version_configs=VersionConfigDict(
            root={
                "v9": VersionConfig(image_name="unsuffixed:v9"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=platform))
    monkeypatch.setattr(InferenceBackend, "one_by_fields", AsyncMock(return_value=None))
    create_mock = AsyncMock(side_effect=lambda session, obj, **kwargs: obj)
    monkeypatch.setattr(InferenceBackend, "create", create_mock)
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    with pytest.raises(BadRequestException) as e:
        await ib_route.update_inference_backend(
            MagicMock(), _org_ctx(42), platform.id, backend_in
        )

    assert "must end with '-custom'" in str(e.value.message)
    # The seed was flushed for validation, but never committed.
    assert create_mock.await_args.kwargs["auto_commit"] is False
    update_mock.assert_not_awaited()


@pytest.mark.parametrize(
    'payload',
    [None, VersionConfigDict(root={})],
    ids=['no-version-configs', 'empty-version-configs'],
)
def test_merge_source_owned_versions_with_no_payload_keeps_built_ins(payload):
    """With nothing user-defined submitted, the stored source-owned versions
    are the entire merged result."""
    backend = _community_backend_with_custom_version(backend_id=45)

    merged = ib_route._merge_source_owned_versions(backend, payload)

    assert set(merged.root.keys()) == {"v1", "v2", "v3"}
    assert merged.root["v1"].built_in_frameworks == ["vllm"]
    assert merged.root["v3"].built_in_frameworks == []


@pytest.mark.parametrize(
    'version, marker',
    [("v1", ["vllm"]), ("v3", [])],
    ids=['framework-list-marker', 'empty-list-marker'],
)
def test_merge_source_owned_versions_drops_source_owned_collision_entries(
    version, marker
):
    """An entry whose key matches a stored source-owned version never wins
    the merge: the payload carries no trusted built_in_frameworks marker, so
    letting it through would strip the marker from the stored entry. The
    empty-list marker counts too — a source version declaring no framework
    normalizes to []."""
    backend = _community_backend_with_custom_version(backend_id=46)
    payload = VersionConfigDict(
        root={
            version: VersionConfig(image_name="echoed"),
            "v1-custom": VersionConfig(image_name="custom:v2"),
            "v3-custom": VersionConfig(image_name="custom:v3"),
        }
    )

    merged = ib_route._merge_source_owned_versions(backend, payload)

    assert set(merged.root.keys()) == {"v1", "v2", "v3", "v1-custom", "v3-custom"}
    assert merged.root[version].image_name == f"img:{version}"
    assert merged.root[version].built_in_frameworks == marker
    assert merged.root["v1-custom"].image_name == "custom:v2"
    assert merged.root["v3-custom"].image_name == "custom:v3"


@pytest.mark.asyncio
async def test_validate_version_removal_ignores_source_owned_versions(monkeypatch):
    """A payload that omits a source-owned version (empty-list marker
    included) must not count as a removal: the merge preserves it, so the
    in-use guard has nothing to protect."""
    backend = _community_backend_with_custom_version(backend_id=47)
    payload = VersionConfigDict(
        root={"v1-custom": VersionConfig(image_name="custom:v1")}
    )

    in_use_mock = AsyncMock(return_value=(True, ["deployed-model"]))
    monkeypatch.setattr(ib_route, "check_backend_in_use", in_use_mock)

    await ib_route._validate_version_removal(MagicMock(), backend, payload)

    in_use_mock.assert_not_called()


@pytest.mark.asyncio
async def test_validate_version_removal_still_guards_user_versions(monkeypatch):
    """Omitting a user-defined version from the payload is a removal and
    must hit the in-use guard."""
    backend = _community_backend_with_custom_version(backend_id=48)
    payload = VersionConfigDict(root={})

    in_use_mock = AsyncMock(return_value=(True, ["deployed-model"]))
    monkeypatch.setattr(ib_route, "check_backend_in_use", in_use_mock)

    with pytest.raises(BadRequestException) as e:
        await ib_route._validate_version_removal(MagicMock(), backend, payload)

    assert "v1-custom" in str(e.value.message)
    in_use_mock.assert_awaited_once()


@pytest.mark.parametrize(
    'version, owned, rejected',
    [
        ('v1', {'v1'}, False),
        ('v1', {'v2'}, True),
        ('v1', None, True),
    ],
    ids=['stored-source-version', 'key-not-owned', 'no-stored-ownership'],
)
def test_validate_custom_suffix_uses_stored_ownership(version, owned, rejected):
    """The '-custom' suffix rule skips only versions the stored row owns.
    The payload's own marker is never consulted: a client could forge it."""
    version_configs = VersionConfigDict(
        root={version: VersionConfig(image_name="img:v9", built_in_frameworks=["vllm"])}
    )

    if rejected:
        with pytest.raises(BadRequestException) as e:
            ib_route.validate_custom_suffix(
                None, version_configs, source_owned_versions=owned
            )
        assert "must end with '-custom'" in str(e.value.message)
    else:
        ib_route.validate_custom_suffix(
            None, version_configs, source_owned_versions=owned
        )


@pytest.mark.parametrize(
    'forged_marker', [["vllm"], []], ids=['non-empty', 'empty-list']
)
def test_validate_custom_suffix_ignores_payload_marker(forged_marker):
    """A payload marker (empty list included) cannot exempt a version from
    the '-custom' suffix rule: ownership comes from the stored row only."""
    version_configs = VersionConfigDict(
        root={
            "v9": VersionConfig(image_name="img:v9", built_in_frameworks=forged_marker)
        }
    )

    with pytest.raises(BadRequestException) as e:
        ib_route.validate_custom_suffix(None, version_configs)
    assert "must end with '-custom'" in str(e.value.message)


@pytest.mark.asyncio
async def test_update_drops_untrusted_marker_entries(monkeypatch):
    """An entry carrying a built_in_frameworks marker for a key the stored
    row does not own is dropped, not validated or stored: the marker is
    forgeable, so the entry cannot be honored as source data and must not
    continue as user data either (a forged marker would dodge the '-custom'
    suffix rule). Legitimate shapes of this entry — a runner version echoed
    from the list response, a version removed by a catalog sync between the
    client's GET and PUT — are echoed catalog data, and dropping them keeps
    the echo round trip idempotent."""
    backend = _community_backend_with_custom_version(backend_id=49)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        version_configs=VersionConfigDict(
            root={
                # Echoing the stored user version so the removal guard has
                # nothing to flag.
                "v1-custom": VersionConfig(image_name="custom:v1"),
                # Forged marker on an unknown key: dropped, not rejected.
                "forged": VersionConfig(image_name="forged:v1", built_in_frameworks=[]),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    merged = update_mock.await_args.args[1]["version_configs"].root
    assert set(merged.keys()) == {"v1", "v2", "v3", "v1-custom"}
    assert merged["v1-custom"].image_name == "custom:v1"
    assert update_mock.await_args.args[1]["backend_source"] == (
        BackendSourceEnum.COMMUNITY
    )


@pytest.mark.asyncio
async def test_update_drops_forged_marker_entries_even_with_custom_suffix(monkeypatch):
    """A '-custom'-suffixed entry carrying a marker for an unknown key is
    dropped too: user-defined versions never carry the marker, server- or
    client-side, so a marked entry is never user data."""
    backend = _community_backend_with_custom_version(backend_id=50)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        backend_source=BackendSourceEnum.COMMUNITY,
        version_configs=VersionConfigDict(
            root={
                "v1-custom": VersionConfig(image_name="custom:v1"),
                "forged-custom": VersionConfig(
                    image_name="forged:v1", built_in_frameworks=["vllm"]
                ),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    await ib_route.update_inference_backend(
        MagicMock(), _all_mode_admin_ctx(), backend.id, backend_in
    )

    merged = update_mock.await_args.args[1]["version_configs"].root
    assert "forged-custom" not in merged
    assert merged["v1-custom"].image_name == "custom:v1"


def test_process_version_configs_preserves_markers_for_validation():
    """Processing keeps the payload's markers so the ownership rules can see
    them: untrusted ones are dropped and the rest cleared after validation,
    never honored."""
    processed = ib_route._process_version_configs(
        {
            "v1": {"image_name": "img:v1", "built_in_frameworks": ["vllm"]},
            "v2": {"image_name": "img:v2", "built_in_frameworks": []},
        }
    )

    assert processed.root["v1"].built_in_frameworks == ["vllm"]
    assert processed.root["v2"].built_in_frameworks == []


@pytest.mark.asyncio
async def test_redirect_seed_rejects_payload_versions_by_ownership(monkeypatch):
    """Seeded Org rows take ownership from the Platform row, not the payload:
    an override of a catalog version keeps the stored entry (content and
    marker), and an entry carrying a marker for an unknown key is dropped as
    untrusted."""
    platform = _platform_community_backend()
    backend_in = InferenceBackendUpdate(
        backend_name="my-community",
        backend_source=BackendSourceEnum.COMMUNITY,
        enabled=True,
        version_configs=VersionConfigDict(
            root={
                # Override of a catalog version, forged marker included.
                "v1": VersionConfig(
                    image_name="org-override:v1", built_in_frameworks=[]
                ),
                # New version with a forged non-empty marker.
                "v9": VersionConfig(
                    image_name="forged:v9", built_in_frameworks=["vllm"]
                ),
                # A genuine user-defined version.
                "v9-custom": VersionConfig(image_name="custom:v9"),
            }
        ),
    )

    monkeypatch.setattr(InferenceBackend, "one_by_fields", AsyncMock(return_value=None))
    monkeypatch.setattr(
        InferenceBackend,
        "create",
        AsyncMock(side_effect=lambda session, obj, **kwargs: obj),
    )

    created = await ib_route._redirect_global_edit_to_org_row(
        MagicMock(), _org_ctx(42), platform, backend_in
    )

    assert set(created.version_configs.root.keys()) == {"v1", "v2", "v9-custom"}
    assert created.version_configs.root["v1"].image_name == "img:v1"
    assert created.version_configs.root["v1"].built_in_frameworks == ["vllm"]
    assert created.version_configs.root["v9-custom"].image_name == "custom:v9"
    assert created.version_configs.root["v9-custom"].built_in_frameworks is None
    # The Platform row's own markers must be untouched.
    assert platform.version_configs.root["v1"].built_in_frameworks == ["vllm"]
    assert platform.version_configs.root["v1"].image_name == "img:v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", [Worker, RunnerOverrideEntry, InferenceBackend])
async def test_list_backend_configs_reports_database_failures(monkeypatch, failing):
    """Any failing read must surface as a 500, not a 200 with an empty list the
    UI renders as "no backends available". The cause stays in the log: a
    database error carries SQL and schema details."""
    db_error = 'relation "inference_backends" does not exist'

    for model in (Worker, RunnerOverrideEntry, InferenceBackend):
        monkeypatch.setattr(model, "all", AsyncMock(return_value=[]))
    monkeypatch.setattr(failing, "all", AsyncMock(side_effect=RuntimeError(db_error)))

    with pytest.raises(InternalServerErrorException) as e:
        await ib_route.list_backend_configs(session=None, ctx=None)

    assert e.value.status_code == 500
    assert "inference_backends" not in e.value.message
    assert str(e.value.__cause__) == db_error


def _custom_backend_with_version(backend_id: int) -> InferenceBackend:
    return InferenceBackend(
        id=backend_id,
        backend_name="custom-backend",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.CUSTOM,
        enabled=True,
        default_version="v1-custom",
        version_configs=VersionConfigDict(
            root={
                "v1-custom": VersionConfig(
                    image_name="custom:v1", built_in_frameworks=None
                )
            }
        ),
    )


@pytest.mark.asyncio
async def test_delete_backend_fails_closed_when_usage_query_fails(monkeypatch):
    """A failed usage lookup must not be treated as proof that deletion is safe."""
    backend = _custom_backend_with_version(backend_id=41)
    db_error = "usage lookup unavailable"

    monkeypatch.setattr(
        Model,
        "all_by_field",
        AsyncMock(side_effect=RuntimeError(db_error)),
    )
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    delete_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "delete", delete_mock)

    with pytest.raises(InternalServerErrorException) as exc_info:
        await ib_route.delete_inference_backend(MagicMock(), _org_ctx(66), backend.id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == (
        "Failed to verify whether the inference backend is in use"
    )
    assert str(exc_info.value.__cause__) == db_error
    delete_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_version_fails_closed_when_usage_query_fails(monkeypatch):
    """A failed usage lookup must abort the update before a version is removed."""
    backend = _custom_backend_with_version(backend_id=42)
    backend_in = InferenceBackendUpdate(
        backend_name=backend.backend_name,
        version_configs=VersionConfigDict(root={}),
    )
    db_error = "usage lookup unavailable"

    monkeypatch.setattr(
        Model,
        "all_by_fields",
        AsyncMock(side_effect=RuntimeError(db_error)),
    )
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    monkeypatch.setattr(
        ib_route,
        "_redirect_global_edit_to_org_row",
        AsyncMock(return_value=None),
    )
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    with pytest.raises(InternalServerErrorException) as exc_info:
        await ib_route.update_inference_backend(
            MagicMock(), _org_ctx(66), backend.id, backend_in
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == (
        "Failed to verify whether the inference backend is in use"
    )
    assert str(exc_info.value.__cause__) == db_error
    update_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_version_from_yaml_preserves_fail_closed_error(monkeypatch):
    """The YAML update path must not wrap the usage-check 500 or lose its cause."""
    backend = _custom_backend_with_version(backend_id=43)
    db_error = "usage lookup unavailable"

    monkeypatch.setattr(
        Model,
        "all_by_fields",
        AsyncMock(side_effect=RuntimeError(db_error)),
    )
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    update_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "update", update_mock)

    with pytest.raises(InternalServerErrorException) as exc_info:
        await ib_route.update_inference_backend_from_yaml(
            MagicMock(),
            _org_ctx(66),
            backend.id,
            {"content": "backend_name: custom-backend\nversion_configs: {}\n"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == (
        "Failed to verify whether the inference backend is in use"
    )
    assert str(exc_info.value.__cause__) == db_error
    update_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_enabled', [None, False, True])
async def test_backend_options_expose_the_same_command_defaults_as_worker_resolution(
    monkeypatch,
    owner_enabled,
):
    platform = _platform_community_backend()
    platform.enabled = True
    platform.default_run_command = 'default-router --port {{port}}'
    platform.version_configs.root['v1'].run_command = (
        '--workers {{decode_urls | join(",")}}'
    )
    owner = InferenceBackend(
        id=11,
        backend_name=platform.backend_name,
        owner_principal_id=42,
        enabled=owner_enabled,
        default_run_command=None,
        version_configs=VersionConfigDict(
            root={'v2': VersionConfig(image_name='router:v2')}
        ),
    )
    monkeypatch.setattr(
        ib_route,
        '_fetch_visible_backend_rows',
        AsyncMock(return_value=[platform, owner]),
    )
    monkeypatch.setattr(Worker, 'all', AsyncMock(return_value=[]))
    monkeypatch.setattr(RunnerOverrideEntry, 'all', AsyncMock(return_value=[]))
    response = await ib_route.list_backend_configs(None, _org_ctx(42))
    item = next(
        item for item in response.items if item.backend_name == platform.backend_name
    )
    assert item.enabled is True
    assert item.default_run_command == platform.default_run_command
    assert item.default_command_version == 'v1'
    assert (
        next(
            version for version in item.versions if version.version == 'v1'
        ).run_command
        == platform.version_configs.root['v1'].run_command
    )
    assert set(owner.version_configs.root) == {'v2'}
