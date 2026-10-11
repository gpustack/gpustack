from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_files as routes
from gpustack.schemas.model_files import ModelFileCreate, ModelFileUpdate
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.principals import OrgRole, PrincipalType


def _context(
    principal_id=10, admin=False, shared=False, system=False, scoped_cluster_id=None
):
    return TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM if system else PrincipalType.USER
        ),
        is_platform_admin=admin,
        current_principal_id=principal_id,
        org_role=OrgRole.OWNER,
        accessible_cluster_ids={200} if shared else set(),
        scoped_cluster_id=scoped_cluster_id,
    )


def _create_payload(**kwargs):
    return ModelFileCreate(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        **{"worker_id": 20, "local_dir": "/models/custom", **kwargs},
    )


@pytest.fixture
def persistence(monkeypatch):
    worker = SimpleNamespace(
        id=20, cluster_id=200, owner_principal_id=10, deleted_at=None
    )
    cluster = SimpleNamespace(id=200, owner_principal_id=10, deleted_at=None)
    mocks = SimpleNamespace(
        worker=worker,
        cluster=cluster,
        get_worker=AsyncMock(return_value=worker),
        get_cluster=AsyncMock(return_value=cluster),
        find_file=AsyncMock(return_value=None),
        list_files=AsyncMock(return_value=[]),
        create=AsyncMock(side_effect=lambda session, row: row),
    )
    monkeypatch.setattr(routes.Worker, "one_by_id", mocks.get_worker)
    monkeypatch.setattr(routes.Cluster, "one_by_id", mocks.get_cluster)
    monkeypatch.setattr(routes.ModelFile, "one_by_fields", mocks.find_file)
    monkeypatch.setattr(routes.ModelFile, "all_by_field", mocks.list_files)
    monkeypatch.setattr(routes.ModelFile, "create", mocks.create)
    return mocks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context,owner,error",
    [
        pytest.param(_context(), 99, NotFoundException, id="other-org"),
        pytest.param(
            _context(shared=True), 99, ForbiddenException, id="shared-other-org"
        ),
        pytest.param(
            _context(shared=True), None, ForbiddenException, id="shared-global"
        ),
        pytest.param(_context(), None, NotFoundException, id="hidden-global"),
        pytest.param(
            _context(admin=True), 99, NotFoundException, id="admin-act-as-other-org"
        ),
        pytest.param(
            _context(admin=True, shared=True),
            99,
            ForbiddenException,
            id="admin-act-as-shared",
        ),
        pytest.param(
            _context(system=True, scoped_cluster_id=100),
            10,
            NotFoundException,
            id="system-other-cluster-same-org",
        ),
        pytest.param(
            _context(system=True, scoped_cluster_id=100, shared=True),
            99,
            NotFoundException,
            id="system-other-cluster-shared",
        ),
    ],
)
async def test_create_denies_unauthorized_worker_before_file_queries(
    persistence, context, owner, error
):
    persistence.cluster.owner_principal_id = owner
    persistence.worker.owner_principal_id = owner
    # Existing files must not reveal model sources or occupied directories.
    persistence.find_file.return_value = object()
    with pytest.raises(error):
        await routes.create_model_file(None, context, _create_payload())

    persistence.find_file.assert_not_awaited()
    persistence.list_files.assert_not_awaited()
    persistence.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context,owner",
    [
        pytest.param(_context(), 10, id="org-owner"),
        pytest.param(_context(admin=True), 10, id="admin-act-as-own-org"),
        pytest.param(_context(principal_id=None, admin=True), 99, id="admin-all"),
        pytest.param(_context(principal_id=None, admin=True), None, id="admin-global"),
        pytest.param(
            _context(principal_id=None, system=True, scoped_cluster_id=200),
            99,
            id="system-own-cluster",
        ),
        pytest.param(
            _context(principal_id=None, system=True), 99, id="platform-system"
        ),
    ],
)
async def test_create_uses_authorized_cluster_scope(persistence, context, owner):
    persistence.cluster.owner_principal_id = owner
    # The cluster is authoritative even if the worker's copied owner is stale.
    persistence.worker.owner_principal_id = 1234
    payload = _create_payload()

    result = await routes.create_model_file(None, context, payload)

    assert result.worker_id == 20
    assert result.cluster_id == 200
    assert result.owner_principal_id == owner
    assert result.local_dir == "/models/custom"
    assert result.source_index == payload.model_source_index
    persistence.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_target",
    [
        "missing-worker",
        "deleted-worker",
        "no-cluster",
        "missing-cluster",
        "deleted-cluster",
    ],
)
async def test_create_rejects_invalid_worker_scope(persistence, invalid_target):
    deleted_at = datetime.now(timezone.utc)
    if invalid_target == "missing-worker":
        persistence.get_worker.return_value = None
    elif invalid_target == "deleted-worker":
        persistence.worker.deleted_at = deleted_at
    elif invalid_target == "no-cluster":
        persistence.worker.cluster_id = None
    elif invalid_target == "missing-cluster":
        persistence.get_cluster.return_value = None
    else:
        persistence.cluster.deleted_at = deleted_at

    with pytest.raises(NotFoundException):
        await routes.create_model_file(None, _context(), _create_payload())

    persistence.find_file.assert_not_awaited()
    persistence.list_files.assert_not_awaited()
    persistence.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_requires_target_worker(persistence):
    with pytest.raises(InvalidException):
        await routes.create_model_file(
            None, _context(), _create_payload(worker_id=None)
        )

    persistence.get_worker.assert_not_awaited()
    persistence.find_file.assert_not_awaited()
    persistence.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory_conflict", [False, True])
async def test_create_checks_duplicates_for_authorized_worker(
    persistence, directory_conflict
):
    if directory_conflict:
        persistence.list_files.return_value = [_create_payload()]
    else:
        persistence.find_file.return_value = _create_payload()

    with pytest.raises(AlreadyExistsException):
        await routes.create_model_file(None, _context(), _create_payload())

    persistence.create.assert_not_awaited()


@pytest.fixture
def stored_file(monkeypatch):
    row = SimpleNamespace(
        id=1, worker_id=20, cluster_id=200, owner_principal_id=10, update=AsyncMock()
    )
    monkeypatch.setattr(routes.ModelFile, "one_by_id", AsyncMock(return_value=row))
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_id", [None, 21])
@pytest.mark.parametrize("system", [False, True])
async def test_update_cannot_reassign_worker(stored_file, worker_id, system):
    payload = ModelFileUpdate(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        worker_id=worker_id,
    )
    with pytest.raises(InvalidException):
        await routes.update_model_file(
            None, _context(system=system, scoped_cluster_id=200), 1, payload
        )

    stored_file.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_worker", [False, True])
async def test_worker_can_report_download_progress(stored_file, include_worker):
    payload = ModelFileUpdate(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="test/model",
        download_progress=50,
        resolved_paths=["/models/custom/model.bin"],
        **({"worker_id": 20} if include_worker else {}),
    )
    context = _context(principal_id=None, system=True, scoped_cluster_id=200)

    result = await routes.update_model_file(None, context, 1, payload)

    assert result is stored_file
    stored_file.update.assert_awaited_once_with(None, payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    [_context(principal_id=99), _context(system=True, scoped_cluster_id=100)],
)
async def test_update_denies_invisible_file(stored_file, context):
    payload = ModelFileUpdate(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="test/model", worker_id=20
    )
    with pytest.raises(NotFoundException):
        await routes.update_model_file(None, context, 1, payload)

    stored_file.update.assert_not_awaited()
