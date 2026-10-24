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


def _scoped_backend(
    *,
    backend_id: int,
    owner_principal_id: int,
    backend_name: str = "vLLM",
    version_configs=None,
) -> InferenceBackend:
    return InferenceBackend(
        id=backend_id,
        backend_name=backend_name,
        owner_principal_id=owner_principal_id,
        backend_source=BackendSourceEnum.BUILT_IN,
        version_configs=version_configs,
    )


def _model(
    *,
    name: str,
    owner_principal_id: int,
    backend: str = "vLLM",
    backend_version=None,
    replicas: int = 1,
) -> Model:
    return Model(
        name=name,
        owner_principal_id=owner_principal_id,
        backend=backend,
        backend_version=backend_version,
        replicas=replicas,
    )


def _mock_model_usage(monkeypatch, models):
    def matching_models(fields):
        return [
            model
            for model in models
            if all(getattr(model, key) == value for key, value in fields.items())
        ]

    async def all_by_fields(session, fields, **kwargs):
        return matching_models(fields)

    async def all_by_field(session, field, value, **kwargs):
        return matching_models({field: value})

    monkeypatch.setattr(Model, "all_by_fields", all_by_fields)
    monkeypatch.setattr(Model, "all_by_field", all_by_field)


async def _delete_backend(monkeypatch, backend: InferenceBackend):
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    delete_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "delete", delete_mock)

    await ib_route.delete_inference_backend(
        MagicMock(), _org_ctx(backend.owner_principal_id or 1), backend.id
    )
    return delete_mock


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


@pytest.mark.asyncio
async def test_delete_scoped_backend_ignores_running_models_of_other_owners(
    monkeypatch,
):
    """Deleting owner 56's scoped row must not be blocked by owner 66's model."""
    backend = _scoped_backend(backend_id=39, owner_principal_id=56)
    _mock_model_usage(
        monkeypatch,
        [_model(name="owner-66-running", owner_principal_id=66)],
    )

    delete_mock = await _delete_backend(monkeypatch, backend)

    delete_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_scoped_backend_blocks_on_running_model_of_same_owner(
    monkeypatch,
):
    """A running model owned by the scoped backend's owner still blocks delete."""
    backend = _scoped_backend(backend_id=40, owner_principal_id=66)
    _mock_model_usage(
        monkeypatch,
        [_model(name="owner-66-running", owner_principal_id=66)],
    )
    delete_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "delete", delete_mock)
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())

    with pytest.raises(BadRequestException) as exc_info:
        await ib_route.delete_inference_backend(MagicMock(), _org_ctx(66), backend.id)

    assert exc_info.value.status_code == 400
    assert "owner-66-running" in exc_info.value.message
    delete_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_platform_custom_backend_checks_models_for_all_owners(
    monkeypatch,
):
    """Platform custom backends retain the global usage check."""
    backend = InferenceBackend(
        id=1,
        backend_name="platform-custom",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.CUSTOM,
    )
    _mock_model_usage(
        monkeypatch,
        [
            _model(
                name="owner-66-running",
                owner_principal_id=66,
                backend="platform-custom",
            )
        ],
    )
    monkeypatch.setattr(InferenceBackend, "one_by_id", AsyncMock(return_value=backend))
    monkeypatch.setattr(ib_route, "assert_org_owned_writable", MagicMock())
    delete_mock = AsyncMock()
    monkeypatch.setattr(InferenceBackend, "delete", delete_mock)

    with pytest.raises(BadRequestException) as exc_info:
        await ib_route.delete_inference_backend(MagicMock(), _org_ctx(1), backend.id)

    assert exc_info.value.status_code == 400
    assert "owner-66-running" in exc_info.value.message
    delete_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_platform_version_checks_models_for_all_owners(monkeypatch):
    """Platform version removal keeps the global usage check."""
    backend = InferenceBackend(
        id=1,
        backend_name="platform-custom",
        owner_principal_id=None,
        backend_source=BackendSourceEnum.CUSTOM,
        version_configs=VersionConfigDict(
            root={
                "v1-custom": VersionConfig(
                    image_name="custom:v1", built_in_frameworks=None
                )
            }
        ),
    )
    _mock_model_usage(
        monkeypatch,
        [
            _model(
                name="owner-66-running",
                owner_principal_id=66,
                backend="platform-custom",
                backend_version="v1-custom",
            )
        ],
    )

    with pytest.raises(BadRequestException) as exc_info:
        await ib_route._validate_version_removal(
            MagicMock(), backend, VersionConfigDict(root={})
        )

    assert exc_info.value.status_code == 400
    assert "owner-66-running" in exc_info.value.message


@pytest.mark.asyncio
async def test_delete_scoped_backend_without_models_succeeds(monkeypatch):
    """A scoped row with no association in its own owner scope is deletable."""
    backend = _scoped_backend(backend_id=39, owner_principal_id=56)
    _mock_model_usage(monkeypatch, [])

    delete_mock = await _delete_backend(monkeypatch, backend)

    delete_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_scoped_version_blocks_on_same_owner_model(monkeypatch):
    """Removing a scoped custom version checks a running model of that owner."""
    backend = _scoped_backend(
        backend_id=40,
        owner_principal_id=66,
        version_configs=VersionConfigDict(
            root={
                "v1-custom": VersionConfig(
                    image_name="custom:v1", built_in_frameworks=None
                )
            }
        ),
    )
    _mock_model_usage(
        monkeypatch,
        [
            _model(
                name="owner-66-running",
                owner_principal_id=66,
                backend_version="v1-custom",
            )
        ],
    )

    with pytest.raises(BadRequestException) as exc_info:
        await ib_route._validate_version_removal(
            MagicMock(), backend, VersionConfigDict(root={})
        )

    assert exc_info.value.status_code == 400
    assert "owner-66-running" in exc_info.value.message
    assert "v1-custom" in exc_info.value.message


@pytest.mark.asyncio
async def test_remove_scoped_version_ignores_model_of_another_owner(monkeypatch):
    """Another owner's use of the same version must not block removal."""
    backend = _scoped_backend(
        backend_id=40,
        owner_principal_id=66,
        version_configs=VersionConfigDict(
            root={
                "v1-custom": VersionConfig(
                    image_name="custom:v1", built_in_frameworks=None
                )
            }
        ),
    )
    _mock_model_usage(
        monkeypatch,
        [
            _model(
                name="owner-56-running",
                owner_principal_id=56,
                backend_version="v1-custom",
            )
        ],
    )

    await ib_route._validate_version_removal(
        MagicMock(), backend, VersionConfigDict(root={})
    )
