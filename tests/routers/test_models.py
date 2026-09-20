from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import delete, func, update
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import models as models_route
from gpustack.routes.models import (
    create_model,
    export_models,
    import_models,
    update_model,
)
from gpustack.routes.model_common import ModelStateFilterEnum
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.deployment_document import (
    ENTRY_FIELDS,
    OVERWRITABLE_FIELDS,
    DeploymentExportRequest,
    DeploymentImportRequest,
    dump_deployments,
    load_deployments,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteTarget,
)
from gpustack.schemas.models import (
    GPUSelector,
    LoraListEntry,
    Model,
    ModelCreate,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelUpdate,
    SourceEnum,
)
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.utils.export_limits import attachment_headers

DEFAULT_ORG_ID = platform_principal_id()
CUSTOM_ORG_ID = 5
OTHER_ORG_ID = 7
CLUSTER_ID = 101


def _ctx(current_principal_id, is_admin=False, accessible_cluster_ids=None):
    user = MagicMock()
    user.id = 99
    user.is_admin = is_admin
    # Tenant helpers compare user.kind against PrincipalType.SYSTEM; pin it
    # to a non-SYSTEM kind so a bare mock can't drift into the SYSTEM bypass.
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=current_principal_id,
        org_role=None,
        accessible_cluster_ids=set(accessible_cluster_ids or []),
    )


def _model_create(cluster_id=None):
    return ModelCreate(
        name="m1",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        cluster_id=cluster_id,
    )


def _cluster(owner_principal_id, cluster_id=CLUSTER_ID, deleted=False):
    cluster = MagicMock()
    cluster.id = cluster_id
    cluster.owner_principal_id = owner_principal_id
    cluster.deleted_at = object() if deleted else None
    return cluster


@pytest.mark.asyncio
async def test_create_model_rejects_default_org_cluster_for_custom_org(monkeypatch):
    """A custom org cannot deploy onto a visible cluster owned by another
    org (e.g. the Default org's shared cluster) — 403, not 404."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(DEFAULT_ORG_ID)),
    )

    with pytest.raises(ForbiddenException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A cluster the caller can't see is reported as missing (404), not
    forbidden (403), so cross-tenant cluster ids can't be probed."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_missing_cluster(monkeypatch):
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_deleted_cluster(monkeypatch):
    """A soft-deleted cluster is treated as missing (404)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID, deleted=True)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_allows_own_org_cluster(monkeypatch):
    """An own-org cluster passes the org-alignment check and proceeds to
    the name-uniqueness check (signalled here by AlreadyExists)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID)),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        AsyncMock(return_value=MagicMock()),
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_derives_owner_from_cluster(monkeypatch):
    """Admin in "All" mode (no principal context) derives the owning org
    from the chosen cluster; the ownership check then passes even for a
    non-default org's cluster, and the model is stamped with that owner."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )
    one_by_fields = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        one_by_fields,
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=CLUSTER_ID),
        )

    # The uniqueness pre-check runs against the org derived from the
    # cluster, not the platform default.
    assert one_by_fields.await_args.args[1]["owner_principal_id"] == OTHER_ORG_ID


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_rejects_missing_cluster(monkeypatch):
    """Admin "All" mode still rejects a non-existent cluster rather than
    stamping the model with the platform default."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=999),
        )


async def _run_update(monkeypatch, ctx, cluster_return):
    """Drive update_model for an owned model pointed at ``cluster_return``."""
    model = MagicMock()
    model.owner_principal_id = CUSTOM_ORG_ID
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_resource_visible",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=cluster_return),
    )
    await update_model(
        MagicMock(),
        ctx,
        1,
        ModelUpdate(
            name="m1",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            cluster_id=CLUSTER_ID,
        ),
    )


@pytest.mark.asyncio
async def test_update_model_rejects_cross_org_cluster(monkeypatch):
    """A visible cluster owned by another org is a 403 on update."""
    with pytest.raises(ForbiddenException):
        await _run_update(
            monkeypatch,
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _cluster(DEFAULT_ORG_ID),
        )


@pytest.mark.asyncio
async def test_update_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A non-visible cluster is a 404 on update, not a 403 — no probing."""
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), _cluster(OTHER_ORG_ID))


@pytest.mark.asyncio
async def test_update_model_rejects_missing_cluster(monkeypatch):
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), None)


@pytest.mark.parametrize(
    "ready, replicas, state, expected",
    [
        (2, 3, ModelStateFilterEnum.READY, True),
        (0, 3, ModelStateFilterEnum.READY, False),
        (0, 3, ModelStateFilterEnum.NOT_READY, True),
        (2, 3, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.STOPPED, True),
        (0, 3, ModelStateFilterEnum.STOPPED, False),
        (0, 3, None, True),
    ],
)
def test_model_watch_filter_applies_state(
    monkeypatch, ready, replicas, state, expected
):
    """The /models watch stream honors ``state`` via replica counts."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=state
    )
    data = SimpleNamespace(ready_replicas=ready, replicas=replicas)
    assert visible(data) is expected


def test_model_watch_filter_passes_id_only_delete_events(monkeypatch):
    """ID-only DELETED payloads lack replica counts and must not be dropped
    by the state filter, else watch clients hold stale rows."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=ModelStateFilterEnum.READY
    )
    assert visible({"id": 7}) is True


@pytest.mark.asyncio
async def test_update_model_rejects_gpu_selector_on_vgpu_model(monkeypatch):
    """A sparse PUT setting gpu_selector on a model that already carries
    gpu_type_selector must fail mutual-exclusion validation against the
    merged (stored + request) state, not just the request payload."""
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.models import GPUSelector, GPUTypeSelector

    stored = MagicMock()
    stored.owner_principal_id = CUSTOM_ORG_ID
    stored.cluster_id = CLUSTER_ID
    stored.gpu_type_selector = GPUTypeSelector(
        type="pool-a100",
        accelerator_sliced_memory_percentage=50,
        accelerator_sliced_cores_percentage=50,
    )
    stored.gpu_selector = None
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id",
        AsyncMock(return_value=stored),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_resource_visible",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_cluster_belongs_to_org",
        AsyncMock(),
    )

    with pytest.raises(BadRequestException):
        await update_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            1,
            ModelUpdate(
                name="m1",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/repo",
                gpu_selector=GPUSelector(
                    gpu_ids=["worker-1:nvidia:0"],
                    gpus_per_replica=1,
                ),
            ),
        )


# ==================== Deployment YAML export / import ====================
#
# The document schema (what an exported entry carries, how strictly it reads
# back) and the two routes, the latter against an in-memory SQLite database
# with the handlers driven directly through a ``TenantContext``.

EXPORTED_AT = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)

TABLES = (
    Principal.__table__,
    Cluster.__table__,
    Model.__table__,
    ModelInstance.__table__,
    ModelRoute.__table__,
    ModelRouteTarget.__table__,
    ModelRoutePrincipalLink.__table__,
)

# What a user would hand-write: a routed base model with a LoRA, explicit GPU
# placement and a credential, plus a plain embedding model.
DOCUMENT = """
- name: qwen3-8b
  source: huggingface
  huggingface_repo_id: Qwen/Qwen3-8B
  backend: vLLM
  backend_parameters:
  - --max-model-len=32768
  env:
    HF_TOKEN: hf_xxx
  gpu_selector:
    gpu_ids:
    - worker-1:cuda:0
  lora_list:
  - lora_name: sql
    lora_repo_name: org/sql-lora
  enable_model_route: true
- name: bge-m3
  source: huggingface
  huggingface_repo_id: BAAI/bge-m3
  replicas: 2
"""

# A live scaling schedule drives `replicas` itself, so it is kept out of
# DOCUMENT (which the overwrite tests stop entry by entry) and appended only
# where a round trip has to carry one.
SCHEDULED_ENTRY = """- name: scheduled
  source: huggingface
  huggingface_repo_id: org/scheduled
  scaling_schedule:
    enabled: true
    baseline_replicas: 1
    rules:
    - start_cron: 0 8 * * *
      duration_seconds: 3600
      replicas: 4
      name: daytime
"""


def _model_row(
    name, cluster_id=1, owner_principal_id=DEFAULT_ORG_ID, **fields
) -> Model:
    fields.setdefault("source", SourceEnum.HUGGING_FACE)
    fields.setdefault("huggingface_repo_id", f"org/{name}")
    return Model(
        name=name,
        cluster_id=cluster_id,
        owner_principal_id=owner_principal_id,
        **fields,
    )


def _exported_row(**overrides) -> Model:
    """A fully configured row with every server-derived field set, so the dump
    tests can check that none of those leak into the document."""
    fields = dict(
        id=7,
        name="qwen3-8b",
        huggingface_repo_id="Qwen/Qwen3-8B",
        backend="vLLM",
        backend_version="0.11.0",
        backend_parameters=["--max-model-len=32768"],
        env={"HF_TOKEN": "hf_xxx"},
        gpu_selector=GPUSelector(gpu_ids=["worker-1:cuda:0"]),
        lora_list=[
            LoraListEntry(
                lora_name="qwen3-8b:sql",
                lora_repo_name="org/sql-lora",
                path="/var/lib/gpustack/cache/sql",
                model_file_id=9,
            )
        ],
        meta={"n_params": 8_000_000_000},
        ready_replicas=1,
        cluster_id=3,
        owner_principal_id=CUSTOM_ORG_ID,
        access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
    )
    fields.update(overrides)
    return _model_row(fields.pop("name"), **fields)


def test_dump_keeps_user_input_and_drops_server_state():
    text = dump_deployments(
        [_exported_row()],
        route_backed_ids={7},
        cluster_names={3: "c1"},
        exported_at=EXPORTED_AT,
    )

    assert text.startswith("# Exported from GPUStack v")
    assert "at 2026-09-07T10:00:00Z\n" in text
    # Each deployment starts at the left margin, never indented into a list.
    assert "\nname: qwen3-8b\n" in text
    (entry,) = yaml.safe_load_all(text)

    # The cluster travels by name; its id never leaves the server.
    assert entry["cluster_name"] == "c1"
    # ``name`` and ``cluster`` lead, everything else keeps the schema's
    # declaration order.
    assert list(entry)[:4] == ["name", "cluster_name", "source", "huggingface_repo_id"]
    assert list(entry)[-1] == "enable_model_route"
    assert entry["enable_model_route"] is True
    assert entry["backend_version"] == "0.11.0"
    assert entry["backend_parameters"] == ["--max-model-len=32768"]
    assert entry["env"] == {"HF_TOKEN": "hf_xxx"}
    assert entry["gpu_selector"] == {"gpu_ids": ["worker-1:cuda:0"]}
    assert entry["lora_list"] == [
        {"lora_name": "sql", "lora_repo_name": "org/sql-lora", "source": "huggingface"}
    ]

    for field in (
        "id",
        "created_at",
        "updated_at",
        "ready_replicas",
        "meta",
        "cluster_id",
        "owner_principal_id",
        "access_policy",
    ):
        assert field not in entry, field
    # ``None`` never round-trips into an explicit null.
    assert "description" not in entry
    assert "worker_selector" in entry  # an empty dict is user input, kept


def test_dump_is_byte_stable_and_infers_the_route_flag_per_model():
    models = [_exported_row(), _exported_row(id=8, name="second", lora_list=None)]
    dump = dict(route_backed_ids={7}, cluster_names={3: "c1"}, exported_at=EXPORTED_AT)

    first = dump_deployments(models, **dump)
    second = dump_deployments(models, **dump)
    assert first == second

    # One `---` between the two deployments, and none before the first.
    assert first.count("\n---\n") == 1
    flags = [entry["enable_model_route"] for entry in yaml.safe_load_all(first)]
    assert flags == [True, False]


def test_entry_fields_are_pinned():
    """The document's field set is derived, so a new column on the deployment
    schema would silently join both the export and what an overwrite writes.
    Pinning it makes that a deliberate edit, here and in the docs."""
    assert ENTRY_FIELDS == [
        "name",
        "cluster_name",
        "source",
        "huggingface_repo_id",
        "huggingface_filename",
        "model_scope_model_id",
        "model_scope_file_path",
        "local_path",
        "description",
        "replicas",
        "categories",
        "placement_strategy",
        "cpu_offloading",
        "distributed_inference_across_workers",
        "worker_selector",
        "gpu_selector",
        "gpu_type_selector",
        "backend",
        "backend_version",
        "backend_parameters",
        "image_name",
        "run_command",
        "native_anthropic_api",
        "env",
        "restart_on_error",
        "distributable",
        "extended_kv_cache",
        "speculative_config",
        "scaling_schedule",
        "generic_proxy",
        "lora_list",
        "enable_model_route",
    ]
    # An overwrite settles routes rather than writing a column, and never
    # moves a deployment between clusters.
    assert OVERWRITABLE_FIELDS == [
        field
        for field in ENTRY_FIELDS
        if field not in ("enable_model_route", "cluster_name")
    ]
    # Tenancy and identity are the fields that must never be written by an
    # import, whatever else the schema grows. The cluster travels by name
    # instead, which the route resolves within the caller's own Org.
    for field in ("id", "cluster_id", "owner_principal_id", "access_policy", "meta"):
        assert field not in ENTRY_FIELDS, field


@pytest.mark.parametrize(
    "text, message",
    [
        # A mapping is one deployment, so what is left to reject outright is a
        # file holding no deployment at all.
        ("just a string\n", "one deployment per '---' section"),
        ("\n", "one deployment per '---' section"),
        ("- [\n", "not valid YAML"),
    ],
)
def test_load_rejects_a_malformed_document_outright(text, message):
    with pytest.raises(ValueError) as raised:
        load_deployments(text)
    assert message in str(raised.value)


def test_load_reads_both_spellings_of_the_same_deployments():
    """One deployment per YAML document is what export writes; a single list
    is how it used to, and an already exported file has to stay importable."""
    sections = """name: a
source: huggingface
huggingface_repo_id: org/a
---
name: b
source: huggingface
huggingface_repo_id: org/b
"""
    listed = """- name: a
  source: huggingface
  huggingface_repo_id: org/a
- name: b
  source: huggingface
  huggingface_repo_id: org/b
"""
    for text in (sections, listed):
        loaded = load_deployments(text)
        assert [(item.index, item.name, item.errors) for item in loaded] == [
            (0, "a", []),
            (1, "b", []),
        ]

    # A separator around the outside marks a document boundary, not an empty
    # deployment; one deployment needs no separator at all.
    assert len(load_deployments(f"---\n{sections}---\n")) == 2
    assert len(load_deployments(sections.split("---")[0])) == 1


def test_load_reports_every_entry_problem_and_keeps_the_good_entries():
    text = """
- name: good
  source: huggingface
  huggingface_repo_id: org/good
  .anchor: &shared {}
- name: broken
  source: huggingface
  huggingface_repo_id: org/broken
  replicas: -1
  colour: red
  id: 12
  cluster_id: 3
  gpu_selector:
    gpu_ids: [worker-1:cuda:0]
    typo: 1
  lora_list:
  - lora_name: sql
    lora_repo_name: org/sql-lora
    colour: red
    path: /var/lib/gpustack/cache/sql
- name: good
  source: huggingface
  huggingface_repo_id: org/again
- not a mapping
- name: [oops]
  source: huggingface
  huggingface_repo_id: org/list
- name: 123
  source: huggingface
  huggingface_repo_id: org/number
"""
    loaded = load_deployments(text)

    # One entry per document item, in order, whether or not it parsed. A name
    # that is not a string is no name at all.
    assert [(item.index, item.name) for item in loaded] == [
        (0, "good"),
        (1, "broken"),
        (2, "good"),
        (3, None),
        (4, None),
        (5, None),
    ]
    assert [item.entry is not None for item in loaded] == [True] + [False] * 5
    assert loaded[0].label == "deployment[0] (good)"
    assert loaded[3].label == "deployment[3]"

    # The schema error keeps pydantic's own text; the entry carries it
    # unlabelled, since the entry it belongs to is right there.
    broken = loaded[1].errors
    assert broken[2].startswith("replicas: ")
    assert broken[:2] + broken[3:] == [
        "server-managed field(s) are not allowed: cluster_id, id, lora_list[0].path",
        "unknown field(s): colour, gpu_selector.typo, lora_list[0].colour",
    ]
    assert loaded[2].errors == [
        "duplicate name, already used by deployment[0]",
    ]
    assert loaded[3].errors == ["must be a mapping"]
    assert loaded[4].errors[0].startswith("name: ")
    assert loaded[5].errors[0].startswith("name: ")

    # An entry that validated needs no raw copy — its projection is what a
    # client shows. One that did not has no projection, so the file's own
    # text is all there is to put in front of the user to fix.
    assert [bool(item.raw) for item in loaded] == [False, True, True, False, True, True]
    assert loaded[1].raw["colour"] == "red"
    assert loaded[4].raw["name"] == ["oops"]


def test_attachment_header_fallback_stays_a_well_formed_quoted_string():
    header = attachment_headers('模型"x.yaml')["Content-Disposition"]
    assert header == (
        'attachment; filename="___x.yaml"; '
        "filename*=UTF-8''%E6%A8%A1%E5%9E%8B%22x.yaml"
    )


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        for table in TABLES:
            await conn.run_sync(table.create)
    yield e
    await e.dispose()


@pytest.fixture
def no_gpu_lookup(monkeypatch):
    """``gpu_selector`` validation needs live workers; placement is not under test."""
    monkeypatch.setattr("gpustack.routes.models.validate_gpu_ids", AsyncMock())


async def _seed(session: AsyncSession, *rows):
    session.add_all(rows)
    await session.commit()
    return rows


async def _count(session: AsyncSession, table) -> int:
    return (await session.exec(select(func.count()).select_from(table))).one()


async def _route_names(session: AsyncSession):
    return sorted(route.name for route in await ModelRoute.all_by_fields(session))


def _entries(response):
    assert response.media_type == "application/x-yaml"
    return list(yaml.safe_load_all(response.body))


def _body_without_header(response) -> str:
    return response.body.decode().split("\n", 1)[1]


async def _import(session, ctx, content, cluster_id=1, dry_run=False, **overrides):
    return await import_models(
        session,
        ctx,
        DeploymentImportRequest(
            content=content, cluster_id=cluster_id, dry_run=dry_run, **overrides
        ),
    )


async def _stop_all(session):
    """Only a stopped deployment can be overwritten."""
    await session.exec(update(Model).values(replicas=0))
    await session.commit()


def _stopped(document: str) -> str:
    """The document with every entry stopped, to match rows _stop_all left."""
    return document.replace("  replicas: 2\n", "").replace(
        "  source: huggingface\n", "  source: huggingface\n  replicas: 0\n"
    )


def _plan(result) -> list:
    """The plan as (name, action, changed field names) per entry."""
    return [
        (
            entry.name,
            entry.action and entry.action.value,
            [change.field for change in entry.changes],
        )
        for entry in result.entries
    ]


@pytest.mark.asyncio
async def test_export_scopes_to_the_caller_and_names_the_file(engine):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        first, second, foreign = await _seed(
            session,
            _model_row("first", env={"HF_TOKEN": "hf_xxx"}),
            _model_row("second", cluster_id=2),
            _model_row("foreign", owner_principal_id=CUSTOM_ORG_ID),
        )
        await _seed(
            session,
            ModelRoute(name="first", created_model_id=first.id),
            # A LoRA child route also points at its base model; without the
            # primary route the flag must stay off.
            ModelRoute(name="second:sql", created_model_id=second.id),
        )

        # Everything the Org can see, in id order; the route flag is inferred.
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest()
        )
        entries = _entries(response)
        assert [entry["name"] for entry in entries] == ["first", "second"]
        assert [entry["enable_model_route"] for entry in entries] == [True, False]
        assert entries[0]["env"] == {"HF_TOKEN": "hf_xxx"}
        assert "cluster_id" not in entries[0]
        assert response.headers["content-disposition"].startswith(
            'attachment; filename="gpustack-deployments-'
        )

        # ``cluster_id`` narrows; a single model is named after itself.
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(cluster_id=2)
        )
        assert [entry["name"] for entry in _entries(response)] == ["second"]
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="second.yaml"'
        )

        # A cross-tenant or unknown id is 404 with no partial result.
        with pytest.raises(NotFoundException):
            await export_models(
                session,
                _ctx(DEFAULT_ORG_ID),
                DeploymentExportRequest(ids=[first.id, foreign.id]),
            )
        with pytest.raises(NotFoundException):
            await export_models(
                session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(ids=[9999])
            )

        # The admin in "All" mode sees every Org, but a document covers one:
        # names are unique per Org, so two Orgs' rows could collide in a file.
        admin = _ctx(None, is_admin=True)
        with pytest.raises(BadRequestException) as raised:
            await export_models(
                session, admin, DeploymentExportRequest(ids=[foreign.id, second.id])
            )
        assert "more than one organization" in raised.value.message
        response = await export_models(
            session, admin, DeploymentExportRequest(ids=[foreign.id])
        )
        assert [entry["name"] for entry in _entries(response)] == ["foreign"]

        # A name with shell/quote characters still yields a well-formed header.
        (odd,) = await _seed(session, _model_row('odd"name/v1'))
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(ids=[odd.id])
        )
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="odd_name_v1.yaml"'
        )


@pytest.mark.asyncio
async def test_export_omits_scheduler_placement(engine, no_gpu_lookup):
    """Where the scheduler put a deployment must never reach the document:
    those GPUs need not exist in the cluster it is imported into. A sentinel
    for a property nothing enforces -- placement lives on ModelInstance, and
    only a user writes Model.gpu_selector."""
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID),
            _model_row("auto", worker_selector={"zone": "a"}),
            _model_row(
                "manual",
                gpu_selector=GPUSelector(
                    gpu_ids=["worker-1:cuda:0", "worker-1:cuda:1"],
                    gpus_per_replica=2,
                ),
            ),
        )
        auto = await Model.one_by_field(session, "name", "auto")
        await _seed(
            session,
            ModelInstance(
                name="auto-0",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/auto",
                model_id=auto.id,
                model_name="auto",
                worker_id=4,
                gpu_indexes=[0, 1],
                state=ModelInstanceStateEnum.RUNNING,
            ),
        )

        entries = _entries(await export_models(session, ctx, DeploymentExportRequest()))
        by_name = {entry["name"]: entry for entry in entries}
        assert "gpu_selector" not in by_name["auto"]
        assert "gpu_type_selector" not in by_name["auto"]
        assert by_name["auto"]["worker_selector"] == {"zone": "a"}
        # A manual pick is the user's own intent, and survives verbatim.
        assert by_name["manual"]["gpu_selector"] == {
            "gpu_ids": ["worker-1:cuda:0", "worker-1:cuda:1"],
            "gpus_per_replica": 2,
        }


@pytest.mark.asyncio
async def test_import_round_trips_an_export(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )

        document = DOCUMENT + SCHEDULED_ENTRY
        result = await _import(session, ctx, document)
        assert result.dry_run is False
        assert _plan(result) == [
            ("qwen3-8b", "create", []),
            ("bge-m3", "create", []),
            ("scheduled", "create", []),
        ]
        assert [item.name for item in result.items] == [
            "qwen3-8b",
            "bge-m3",
            "scheduled",
        ]
        assert [item.cluster_id for item in result.items] == [1, 1, 1]
        # Public form: the LoRA name is bare, as everywhere else in the API.
        assert result.items[0].model_dump()["lora_list"][0]["lora_name"] == "sql"
        # The route flag created the base route and the LoRA child route.
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]
        assert await _count(session, ModelRouteTarget.__table__) == 2

        exported = await export_models(session, ctx, DeploymentExportRequest())

        # Re-importing an export of rows that still exist changes nothing --
        # including while they run, since nothing would be written.
        result = await _import(session, ctx, exported.body.decode(), dry_run=True)
        assert [entry.action.value for entry in result.entries] == ["unchanged"] * 3
        assert result.valid is True

        for table in (ModelRouteTarget, ModelRoute, Model):
            await session.exec(delete(table))
        await session.commit()
        assert await _count(session, Model.__table__) == 0

        # No cluster forced: the document carries its own, which is the whole
        # of what restoring a backup has to go on.
        await _import(session, ctx, exported.body.decode(), cluster_id=None)
        # The scheduled entry survives the trip with its rules intact.
        scheduled = await Model.one_by_field(session, "name", "scheduled")
        assert scheduled.scaling_schedule.rules[0].start_cron == "0 8 * * *"
        exported_again = await export_models(session, ctx, DeploymentExportRequest())
        assert _body_without_header(exported_again) == _body_without_header(exported)
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]


@pytest.mark.asyncio
async def test_each_deployment_returns_to_its_own_cluster(engine, no_gpu_lookup):
    """A selection spanning clusters exports as one document and imports back
    whole, every deployment to the cluster it came from."""
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID),
            Cluster(id=2, name="c2", owner_principal_id=DEFAULT_ORG_ID),
            _model_row("here", cluster_id=1, replicas=0),
            _model_row("there", cluster_id=2, replicas=0),
        )
        exported = (
            await export_models(session, ctx, DeploymentExportRequest())
        ).body.decode()
        assert [entry["cluster_name"] for entry in yaml.safe_load_all(exported)] == [
            "c1",
            "c2",
        ]

        for table in (ModelRouteTarget, ModelRoute, Model):
            await session.exec(delete(table))
        await session.commit()

        result = await _import(session, ctx, exported, cluster_id=None)
        assert [(item.name, item.cluster_id) for item in result.items] == [
            ("here", 1),
            ("there", 2),
        ]

        # Re-importing the same document changes nothing: the cluster is a
        # document field, so a deployment that is where the file says it is
        # reads as unchanged rather than as a move.
        plan = await _import(session, ctx, exported, cluster_id=None, dry_run=True)
        assert [entry.action.value for entry in plan.entries] == ["unchanged"] * 2

        # A document with no cluster of its own leaves an existing deployment
        # where it is, and has nothing to go on for a new one.
        bare = """
- name: there
  source: huggingface
  huggingface_repo_id: org/there
  replicas: 3
- name: fresh
  source: huggingface
  huggingface_repo_id: org/fresh
"""
        plan = await _import(session, ctx, bare, cluster_id=None, dry_run=True)
        assert plan.entries[0].errors == []
        assert plan.entries[0].desired["cluster_name"] == "c2"
        assert plan.entries[1].errors == [
            'no cluster; choose a target cluster for the import, or add a "cluster_name" '
            "field to this deployment"
        ]

        # A cluster this installation does not have is named, not guessed at,
        # and said once: a deployment that already exists somewhere else must
        # not also be told it targets no cluster, which is the same fault
        # heard twice.
        plan = await _import(
            session,
            ctx,
            bare.replace(
                "- name: fresh\n", "- name: fresh\n  cluster_name: elsewhere\n"
            ).replace("- name: there\n", "- name: there\n  cluster_name: elsewhere\n"),
            cluster_id=None,
            dry_run=True,
        )
        not_found = [
            "cluster 'elsewhere' not found; rename it to a cluster in this "
            "installation, or choose a target cluster for the import"
        ]
        assert plan.entries[0].errors == not_found
        assert plan.entries[1].errors == not_found


@pytest.mark.asyncio
async def test_an_admin_following_the_file_reads_the_org_off_it(engine, no_gpu_lookup):
    """A platform admin in "All" mode has no Org context to import under, so
    the Org is read off the clusters the document names -- and refused unless
    they agree on exactly one."""
    admin = _ctx(None, is_admin=True)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="platform", owner_principal_id=DEFAULT_ORG_ID),
            Cluster(id=2, name="org", owner_principal_id=CUSTOM_ORG_ID),
        )
        named = DOCUMENT.replace(
            "  source: huggingface\n", "  source: huggingface\n  cluster_name: org\n"
        )

        result = await _import(session, admin, named, cluster_id=None)
        assert {item.owner_principal_id for item in result.items} == {CUSTOM_ORG_ID}
        assert {item.cluster_id for item in result.items} == {2}

        # Two Orgs in one file is refused rather than split or collapsed.
        spans = named.replace("cluster_name: org", "cluster_name: platform", 1)
        with pytest.raises(BadRequestException) as raised:
            await _import(session, admin, spans, cluster_id=None)
        assert "more than one organization" in raised.value.message

        # Nothing to read it off: the admin has to say where the file goes.
        with pytest.raises(BadRequestException) as raised:
            await _import(session, admin, DOCUMENT, cluster_id=None)
        assert "no organization to import into" in raised.value.message

        # A cluster this installation does not have is the same dead end.
        with pytest.raises(BadRequestException) as raised:
            await _import(
                session,
                admin,
                named.replace("cluster_name: org", "cluster_name: elsewhere"),
                cluster_id=None,
            )
        assert "no cluster named elsewhere" in raised.value.message


@pytest.mark.asyncio
async def test_two_clusters_of_one_name_are_never_guessed_between(
    engine, no_gpu_lookup
):
    """An Org left with two clusters of one name: resolving by name has to
    refuse rather than pick, and a move is settled on the resolved id."""
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )
        document = """
- name: bge-m3
  source: huggingface
  huggingface_repo_id: BAAI/bge-m3
  replicas: 0
  cluster_name: c1
"""
        await _import(session, ctx, document, cluster_id=None)
        assert (await Model.one_by_field(session, "name", "bge-m3")).cluster_id == 1

        await _seed(
            session, Cluster(id=2, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )
        plan = await _import(session, ctx, document, cluster_id=None, dry_run=True)
        assert plan.entries[0].errors == [
            "more than one cluster is named 'c1'; rename them apart, or choose "
            "a target cluster for the import"
        ]

        # Forcing one is unambiguous, and moves the deployment -- even though
        # both sides of the diff spell the cluster the same way.
        plan = await _import(session, ctx, document, cluster_id=2, dry_run=True)
        assert (plan.entries[0].errors, plan.entries[0].changes) == ([], [])
        assert plan.entries[0].action.value == "update"
        await _import(session, ctx, document, cluster_id=2, overwrite=["bge-m3"])
        session.expire_all()
        assert (await Model.one_by_field(session, "name", "bge-m3")).cluster_id == 2


@pytest.mark.asyncio
async def test_import_reports_every_problem_on_its_own_entry(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID),
            _model_row("running", replicas=1),
            _model_row("broken", replicas=0),
        )
        document = """
- name: running
  source: huggingface
  huggingface_repo_id: org/running
- name: fine
  source: huggingface
  huggingface_repo_id: org/fine
- name: broken
  source: huggingface
  huggingface_repo_id: org/broken
  replicas: -1
  colour: red
- name: fine
  source: huggingface
  huggingface_repo_id: org/fine-again
- name: bad-params
  source: huggingface
  huggingface_repo_id: org/bad
  backend_parameters:
  - --port=8000
- name: [oops]
  source: huggingface
  huggingface_repo_id: org/oops
"""
        result = await _import(session, ctx, document, dry_run=True)
        assert result.valid is False
        errors = [entry.errors for entry in result.entries]
        assert errors[0] == [
            "already exists and is running (replicas=1); stop it before overwriting"
        ]
        assert errors[1] == []
        assert errors[2][0] == "unknown field(s): colour"
        assert errors[2][1].startswith("replicas: ")
        assert errors[3] == ["duplicate name, already used by deployment[1]"]
        assert errors[4] == [
            "Setting the port using --port is not supported. Ports are "
            "automatically allocated by GPUStack."
        ]
        # A name that is not a string is an invalid entry like any other,
        # rather than reaching the lookup and the plan model as itself.
        assert errors[5][0].startswith("name: ")
        assert result.entries[5].name is None
        # An entry that did not validate has no plan, but the deployment it
        # names is still projected: the diff keeps the side the user reads to
        # work out what they broke.
        assert result.entries[2].action is None
        assert result.entries[2].desired == {}
        assert result.entries[2].current["name"] == "broken"
        assert result.entries[2].current["huggingface_repo_id"] == "org/broken"
        # An entry naming nothing that exists has nothing to show there.
        assert result.entries[4].current == {}
        assert await _count(session, Model.__table__) == 2

        # Writing the same document is still a 400 — only the preview is
        # error-free — and it names each entry there.
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, document)
        assert "deployment[2] (broken): unknown field(s): colour" in (
            raised.value.message
        )
        assert await _count(session, Model.__table__) == 2

        # A cluster the caller cannot use is one 404, not one error per entry.
        with pytest.raises(NotFoundException):
            await _import(session, ctx, DOCUMENT, cluster_id=42)
        # A document holding no deployment at all has no plan to show.
        with pytest.raises(BadRequestException):
            await _import(session, ctx, "just a string\n", dry_run=True)
        assert await _count(session, Model.__table__) == 2

        # A LoRA route name owned by another model only conflicts while the
        # routes are created; the error still names the entry, nothing stays.
        await _seed(session, ModelRoute(name="qwen3-8b:sql", created_model_id=999))
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, DOCUMENT)
        assert raised.value.message.startswith("deployment[0] (qwen3-8b): LoRA route")
        assert await _count(session, Model.__table__) == 2


@pytest.mark.asyncio
async def test_overwrite_replaces_only_what_was_confirmed(engine, no_gpu_lookup):
    ctx = _ctx(CUSTOM_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=2, name="org", owner_principal_id=CUSTOM_ORG_ID)
        )
        await _import(session, ctx, DOCUMENT, cluster_id=2)
        before = await Model.one_by_field(session, "name", "qwen3-8b")
        identity = (
            before.id,
            before.created_at,
            before.owner_principal_id,
            before.access_policy,
            before.cluster_id,
        )

        # A running deployment is never overwritten, so stop them first, and
        # drop the GPU pin from the document.
        await _stop_all(session)
        edited = _stopped(DOCUMENT).replace(
            "  gpu_selector:\n    gpu_ids:\n    - worker-1:cuda:0\n", ""
        )

        # Without the caller's confirmation the overwrite is refused outright.
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, edited, cluster_id=2)
        assert "confirm the overwrite before importing" in raised.value.message
        after = await Model.one_by_field(session, "name", "qwen3-8b")
        assert after.gpu_selector is not None

        result = await _import(
            session, ctx, edited, cluster_id=2, overwrite=["qwen3-8b"]
        )
        assert [item.name for item in result.items] == ["qwen3-8b", "bge-m3"]
        after = await Model.one_by_field(session, "name", "qwen3-8b")
        # A whole replacement: a field dropped from the document goes back to
        # its default, returning the deployment to automatic placement.
        assert after.gpu_selector is None
        # Tenancy is never written by an import.
        assert (
            after.id,
            after.created_at,
            after.owner_principal_id,
            after.access_policy,
            after.cluster_id,
        ) == identity

        # Re-applying it now changes nothing, so it needs no confirmation.
        result = await _import(session, ctx, edited, cluster_id=2)
        assert {entry.action.value for entry in result.entries} == {"unchanged"}

        # Setting a nested column takes a different path from clearing one,
        # and mutating one is exactly one change.
        repinned = _stopped(DOCUMENT).replace("worker-1:cuda:0", "worker-2:cuda:3")
        result = await _import(session, ctx, repinned, cluster_id=2, dry_run=True)
        assert [change.field for change in result.entries[0].changes] == [
            "gpu_selector"
        ]
        await _import(session, ctx, repinned, cluster_id=2, overwrite=["qwen3-8b"])
        session.expire_all()
        after = await Model.one_by_field(session, "name", "qwen3-8b")
        assert after.gpu_selector.gpu_ids == ["worker-2:cuda:3"]

        # Every unconfirmed overwrite is named at once, not just the first.
        both = _stopped(DOCUMENT).replace("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
        both = both.replace("BAAI/bge-m3", "BAAI/bge-m3-v2")
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, both, cluster_id=2)
        assert raised.value.message.split("\n") == [
            "deployment[0] (qwen3-8b): already exists; confirm the overwrite "
            "before importing",
            "deployment[1] (bge-m3): already exists; confirm the overwrite "
            "before importing",
        ]


@pytest.mark.asyncio
async def test_overwrite_settles_the_model_route(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )
        await _import(session, ctx, DOCUMENT)
        await _stop_all(session)
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]

        # A second deployment attached to the same route is not covered by the
        # "it is stopped" argument, so the route stays put.
        primary = await ModelRoute.one_by_field(session, "name", "qwen3-8b")
        (other,) = await _seed(session, _model_row("other"))
        await _seed(
            session,
            ModelRouteTarget(
                name="other-deployment",
                route_name=primary.name,
                route_id=primary.id,
                model_id=other.id,
                weight=100,
            ),
        )
        unrouted = _stopped(DOCUMENT).replace("  enable_model_route: true\n", "")
        result = await _import(session, ctx, unrouted, dry_run=True)
        assert result.entries[0].errors == [
            "model route(s) qwen3-8b also target other deployments; "
            "detach them before disabling the route"
        ]
        assert "enable_model_route" in [
            change.field for change in result.entries[0].changes
        ]

        # A LoRA child route is deleted by the same step, so it is held to the
        # same test -- sharing one blocks the entry just as the primary does.
        await session.exec(
            delete(ModelRouteTarget).where(ModelRouteTarget.model_id == other.id)
        )
        child = await ModelRoute.one_by_field(session, "name", "qwen3-8b:sql")
        await _seed(
            session,
            ModelRouteTarget(
                name="other-lora",
                route_name=child.name,
                route_id=child.id,
                model_id=other.id,
                weight=100,
            ),
        )
        result = await _import(session, ctx, unrouted, dry_run=True)
        assert result.entries[0].errors == [
            "model route(s) qwen3-8b:sql also target other deployments; "
            "detach them before disabling the route"
        ]

        # Once it serves only this deployment, disabling it takes the primary
        # route and its LoRA children with it.
        await session.exec(
            delete(ModelRouteTarget).where(ModelRouteTarget.model_id == other.id)
        )
        await session.commit()
        await _import(session, ctx, unrouted, overwrite=["qwen3-8b"])
        assert await _route_names(session) == []

        # And asking for it back builds it again.
        await _import(session, ctx, _stopped(DOCUMENT), overwrite=["qwen3-8b"])
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]

        # Keeping the route enabled still reaps a LoRA child the document stops
        # listing, so a shared one blocks the entry just the same.
        child = await ModelRoute.one_by_field(session, "name", "qwen3-8b:sql")
        await _seed(
            session,
            ModelRouteTarget(
                name="other-lora",
                route_name=child.name,
                route_id=child.id,
                model_id=other.id,
                weight=100,
            ),
        )
        no_lora = _stopped(DOCUMENT).replace(
            "  lora_list:\n  - lora_name: sql\n    lora_repo_name: org/sql-lora\n", ""
        )
        result = await _import(session, ctx, no_lora, dry_run=True)
        assert result.entries[0].errors == [
            "model route(s) qwen3-8b:sql also target other deployments; "
            "detach them before removing the adapter(s)"
        ]

        # Detached, dropping the adapter takes only its own route with it.
        await session.exec(
            delete(ModelRouteTarget).where(ModelRouteTarget.model_id == other.id)
        )
        await session.commit()
        await _import(session, ctx, no_lora, overwrite=["qwen3-8b"])
        assert await _route_names(session) == ["qwen3-8b"]


@pytest.mark.asyncio
async def test_overwrite_guards_a_shared_lora_route_with_no_primary(
    engine, no_gpu_lookup
):
    # Disabling the route deletes every route the deployment owns. With the
    # primary route already gone out of band, a LoRA child it left behind is
    # still one of those, and another deployment may target it -- so the
    # blocker has to fire on the child even with no primary route beside it.
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )
        await _import(session, ctx, DOCUMENT)
        await _stop_all(session)

        # The primary route disappears; its LoRA child stays and another
        # deployment picks it up.
        await session.exec(delete(ModelRoute).where(ModelRoute.name == "qwen3-8b"))
        child = await ModelRoute.one_by_field(session, "name", "qwen3-8b:sql")
        (other,) = await _seed(session, _model_row("other"))
        await _seed(
            session,
            ModelRouteTarget(
                name="other-lora",
                route_name=child.name,
                route_id=child.id,
                model_id=other.id,
                weight=100,
            ),
        )
        await session.commit()

        # Removing enable_model_route disables the route; dropping the GPU pin
        # is the change that makes this an overwrite rather than a no-op.
        disabled = _stopped(DOCUMENT).replace("  enable_model_route: true\n", "")
        disabled = disabled.replace(
            "  gpu_selector:\n    gpu_ids:\n    - worker-1:cuda:0\n", ""
        )
        result = await _import(session, ctx, disabled, dry_run=True)
        assert result.entries[0].errors == [
            "model route(s) qwen3-8b:sql also target other deployments; "
            "detach them before disabling the route"
        ]


@pytest.mark.asyncio
async def test_overwrite_waits_for_a_running_deployment_to_stop(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID),
            Cluster(id=3, name="c3", owner_principal_id=DEFAULT_ORG_ID),
            _model_row("bge-m3", replicas=3),
        )
        running = """
- name: bge-m3
  source: huggingface
  huggingface_repo_id: BAAI/bge-m3
  replicas: 1
"""
        # Whether the document -- or an override -- says zero is irrelevant:
        # what counts is the row in the database.
        for kwargs in ({}, {"replica_overrides": {"bge-m3": 0}}):
            result = await _import(session, ctx, running, dry_run=True, **kwargs)
            assert result.entries[0].errors == [
                "already exists and is running (replicas=3); "
                "stop it before overwriting"
            ]
        with pytest.raises(BadRequestException):
            await _import(session, ctx, running, overwrite=["bge-m3"])

        # Scaled to zero but not drained yet: replicas is the operator's
        # intent, the instance rows are the cluster's state, and the overwrite
        # path needs both.
        stopped = await Model.one_by_field(session, "name", "bge-m3")
        await stopped.update(session, {"replicas": 0})
        await _seed(
            session,
            ModelInstance(
                name="bge-m3-0",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="BAAI/bge-m3",
                model_id=stopped.id,
                model_name="bge-m3",
            ),
        )
        result = await _import(session, ctx, running, dry_run=True)
        assert result.entries[0].errors == [
            "already exists and still has 1 instance(s) shutting down; "
            "wait for them to stop before overwriting"
        ]
        await session.exec(delete(ModelInstance))
        await session.commit()

        # Stopped and drained, a cluster change is an ordinary field change:
        # the row carries no instances to leave behind, and the next ones are
        # placed in whatever cluster it names by then. Writing the cluster
        # down and forcing one are the same claim, so both move it.
        names_c3 = running.replace(
            "- name: bge-m3\n", "- name: bge-m3\n  cluster_name: c3\n"
        )
        for document, cluster_id in ((running, 3), (names_c3, None)):
            result = await _import(
                session, ctx, document, cluster_id=cluster_id, dry_run=True
            )
            assert result.entries[0].errors == []
            assert result.entries[0].desired["cluster_name"] == "c3"

        await _import(session, ctx, names_c3, cluster_id=None, overwrite=["bge-m3"])
        moved = await Model.one_by_field(session, "name", "bge-m3")
        assert moved.cluster_id == 3


@pytest.mark.asyncio
async def test_replica_overrides_apply_to_every_kind_of_entry(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )

        # The preview shows the count that will actually be written, for a
        # manually placed entry (qwen3-8b) as much as an auto-scheduled one.
        result = await _import(
            session,
            ctx,
            DOCUMENT,
            dry_run=True,
            replica_overrides={"qwen3-8b": 3, "bge-m3": 5},
        )
        assert result.valid is True
        assert [entry.desired["replicas"] for entry in result.entries] == [3, 5]
        await _import(
            session, ctx, DOCUMENT, replica_overrides={"qwen3-8b": 3, "bge-m3": 5}
        )
        stored = await Model.all_by_fields(session, {})
        assert {model.name: model.replicas for model in stored} == {
            "qwen3-8b": 3,
            "bge-m3": 5,
        }

        # An enabled schedule owns `replicas`, so the count is what the
        # schedule falls back to rather than a value it would overwrite.
        result = await _import(
            session,
            ctx,
            SCHEDULED_ENTRY,
            dry_run=True,
            replica_overrides={"scheduled": 3},
        )
        assert result.entries[0].desired["scaling_schedule"]["baseline_replicas"] == 3

        # A name that matches nothing is a mistake, not a silent no-op.
        with pytest.raises(BadRequestException) as raised:
            await _import(
                session, ctx, DOCUMENT, dry_run=True, replica_overrides={"typo": 1}
            )
        assert "names no deployment in the document: typo" in raised.value.message


@pytest.mark.asyncio
async def test_dry_run_plans_without_writing(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )

        result = await _import(session, ctx, DOCUMENT, dry_run=True)
        assert result.dry_run is True
        assert result.valid is True
        assert result.items == []
        assert _plan(result) == [("qwen3-8b", "create", []), ("bge-m3", "create", [])]
        # Nothing of either name exists, so there is nothing to diff against.
        assert [entry.current for entry in result.entries] == [{}, {}]
        assert await _count(session, Model.__table__) == 0
        assert await _count(session, ModelRoute.__table__) == 0

        await _import(session, ctx, DOCUMENT)
        assert await _count(session, Model.__table__) == 2

        # Edit the document and the diff names exactly what would change:
        # a raised replica count, and a GPU selector deleted outright.
        edited = DOCUMENT.replace("  replicas: 2", "  replicas: 5").replace(
            "  gpu_selector:\n    gpu_ids:\n    - worker-1:cuda:0\n", ""
        )
        result = await _import(session, ctx, edited, dry_run=True)
        assert _plan(result) == [
            ("qwen3-8b", "update", ["gpu_selector"]),
            ("bge-m3", "update", ["replicas"]),
        ]
        (change,) = result.entries[0].changes
        assert change.current == {"gpu_ids": ["worker-1:cuda:0"]}
        assert change.desired is None
        assert result.entries[1].changes[0].current == 2
        assert result.entries[1].changes[0].desired == 5

        # The row being replaced comes back whole, projected onto the same
        # fields in the same order as the entry that would replace it, so the
        # two read as one document against another rather than as two shapes.
        current, desired = result.entries[0].current, result.entries[0].desired
        assert current["gpu_selector"] == {"gpu_ids": ["worker-1:cuda:0"]}
        assert "gpu_selector" not in desired
        assert [field for field in current if field in desired] == [
            field for field in desired if field in current
        ]
        for field in ("id", "cluster_id", "owner_principal_id"):
            assert field not in current, field

        # Picking GPUs by hand submits a null worker_selector, which drops out
        # of the export and comes back as the schema's empty default. Unset
        # either way, so re-importing reports no change.
        await session.exec(update(Model).values(worker_selector=None, categories=None))
        await session.commit()
        exported = await export_models(session, ctx, DeploymentExportRequest())
        result = await _import(session, ctx, exported.body.decode(), dry_run=True)
        assert [entry.action.value for entry in result.entries] == ["unchanged"] * 2


@pytest.mark.asyncio
async def test_import_stamps_the_callers_org_and_refuses_foreign_clusters(
    engine, no_gpu_lookup
):
    ctx = _ctx(CUSTOM_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="platform", owner_principal_id=DEFAULT_ORG_ID),
            Cluster(id=2, name="org", owner_principal_id=CUSTOM_ORG_ID),
        )

        # Another Org's cluster is reported as missing, and nothing is written.
        with pytest.raises(NotFoundException):
            await _import(session, ctx, DOCUMENT, cluster_id=1)
        assert await _count(session, Model.__table__) == 0

        # Own cluster: rows are stamped with the Org, scoped to it, and the
        # Org is granted on the route it asked for.
        result = await _import(session, ctx, DOCUMENT, cluster_id=2)
        assert {item.owner_principal_id for item in result.items} == {CUSTOM_ORG_ID}
        assert {item.access_policy for item in result.items} == {
            AccessPolicyEnum.ALLOWED_PRINCIPALS
        }
        route = await ModelRoute.one_by_field(session, "name", "qwen3-8b")
        granted = await session.exec(
            select(ModelRoutePrincipalLink.principal_id).where(
                ModelRoutePrincipalLink.route_id == route.id
            )
        )
        assert list(granted.all()) == [CUSTOM_ORG_ID]
