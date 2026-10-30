"""Cluster readers receive masks; authorized writes preserve or replace secrets."""

import base64
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from sqlalchemy.dialects import mysql, postgresql

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.routes import clusters
from gpustack.schemas.clusters import (
    Cluster,
    ClusterProvider,
    ImageCredential,
    K8sOptions,
    build_data_dir_mount,
)
from gpustack.schemas.config import PredefinedConfigNoDefaults
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.bus import Event, EventType
from gpustack.server.db import get_session


def _context(scope):
    return TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=scope == "admin",
        current_principal_id=202 if scope == "shared" else 101,
        org_role=OrgRole.OWNER if scope == "owner" else OrgRole.MEMBER,
        accessible_cluster_ids={1} if scope == "shared" else set(),
    )


@pytest.fixture
def cluster():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Cluster(
        id=1,
        name="shared-cluster",
        provider=ClusterProvider.Kubernetes,
        owner_principal_id=101,
        created_at=now,
        updated_at=now,
        registration_token="private-registration-token",
        worker_config=PredefinedConfigNoDefaults(
            huggingface_token="private-hf-token", debug=False
        ),
        k8s_options=K8sOptions(
            volume_mounts=[build_data_dir_mount()],
            image_credentials=[
                ImageCredential(
                    registry=f"registry-{i}.example",
                    username="user",
                    password=f"private-password-{i}",
                )
                for i in (1, 2)
            ],
        ),
    )


@pytest.fixture
def client(monkeypatch, cluster):
    session = AsyncMock()

    @asynccontextmanager
    async def session_context():
        yield session

    async def update(self, session, source):
        for field in source.model_fields_set:
            setattr(self, field, getattr(source, field))

    monkeypatch.setattr(clusters, "async_session", session_context)
    monkeypatch.setattr(Cluster, "one_by_id", AsyncMock(return_value=cluster))
    monkeypatch.setattr(Cluster, "all_by_fields", AsyncMock(return_value=[cluster]))
    monkeypatch.setattr(Cluster, "update", update)
    app = FastAPI()
    register_handlers(app)
    app.include_router(clusters.router, prefix="/v2/clusters")
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: _context("admin")
    with TestClient(app) as client:
        yield client


def _assert_masked(data):
    assert data["worker_config"]["huggingface_token"] == "**********"
    assert data["worker_config"]["debug"] is False
    credentials = data["k8s_options"]["imageCredentials"]
    assert [item["password"] for item in credentials] == ["**********"] * 2
    assert [item["registry"] for item in credentials] == [
        "registry-1.example",
        "registry-2.example",
    ]
    assert "private-" not in json.dumps(data)


@pytest.mark.parametrize("scope", ["member", "shared", "owner", "admin"])
@pytest.mark.parametrize("path", ["", "?page=0", "/1"])
def test_cluster_reads_mask_credentials(client, cluster, scope, path):
    client.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    response = client.get(f"/v2/clusters{path}")
    assert response.status_code == 200
    data = response.json()
    _assert_masked(data if path == "/1" else data["items"][0])
    assert cluster.worker_config.huggingface_token == "private-hf-token"
    assert cluster.k8s_options.image_credentials[0].password == "private-password-1"


@pytest.mark.parametrize("scope", ["member", "shared", "owner", "admin"])
@pytest.mark.parametrize("local_history", [False, True])
def test_cluster_watch_masks_data_and_omits_secret_history(
    monkeypatch, client, cluster, scope, local_history
):
    client.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    old_config = {"huggingface_token": "private-retired-token"}
    new_config = cluster.worker_config
    if local_history:
        old_config, new_config = [old_config], [new_config]
    changes = {
        "name": ("old-name", cluster.name),
        "worker_config": (old_config, new_config),
        "k8s_options": (None, cluster.k8s_options),
        "registration_token": (
            "private-retired-registration",
            cluster.registration_token,
        ),
        "hashed_suffix": ("private-suffix", "private-new-suffix"),
    }
    original = jsonable_encoder(changes)

    async def subscribe(**kwargs):
        for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED):
            yield Event(type=event_type, data=cluster, changed_fields=changes)

    monkeypatch.setattr(Cluster, "subscribe", subscribe)
    response = client.get("/v2/clusters?watch=true")
    assert response.status_code == 200
    assert "private-" not in response.text
    frames = [json.loads(line) for line in response.text.splitlines() if line]
    assert len(frames) == 3
    for frame in frames:
        _assert_masked(frame["data"])
        assert frame["changed_fields"] == {"name": ["old-name", cluster.name]}
    assert jsonable_encoder(changes) == original


def _update_payload(cluster):
    return {
        "name": cluster.name,
        "worker_config": {"debug": True},
        "k8s_options": {
            "volumeMounts": [build_data_dir_mount().model_dump(by_alias=True)],
            "imageCredentials": [
                {"registry": "registry-1.example", "username": "user"}
            ],
        },
    }


@pytest.mark.parametrize("scope", ["owner", "admin"])
@pytest.mark.parametrize("value", ["omitted", "**********", "replacement", None, ""])
def test_update_preserves_replaces_and_clears_credentials(
    client, cluster, scope, value
):
    client.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    body = _update_payload(cluster)
    if value != "omitted":
        body["worker_config"]["huggingface_token"] = value
        body["k8s_options"]["imageCredentials"][0]["password"] = value
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 200, response.text
    keep = value in ("omitted", "**********")
    assert cluster.worker_config.huggingface_token == (
        "private-hf-token" if keep else value
    )
    assert cluster.k8s_options.image_credentials[0].password == (
        "private-password-1" if keep else value
    )
    assert len(cluster.k8s_options.image_credentials) == 1
    assert cluster.worker_config.debug is True
    assert "private-" not in response.text
    assert "replacement" not in response.text


def test_read_edit_write_round_trip_matches_credentials_by_identity(client, cluster):
    body = client.get("/v2/clusters/1").json()
    body["worker_config"]["debug"] = True
    body["k8s_options"]["imageCredentials"].reverse()
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 200, response.text
    assert cluster.worker_config.huggingface_token == "private-hf-token"
    assert [item.password for item in cluster.k8s_options.image_credentials] == [
        "private-password-2",
        "private-password-1",
    ]


@pytest.mark.parametrize("field", ["registry", "username"])
def test_mask_cannot_move_a_password_to_a_different_identity(client, cluster, field):
    body = _update_payload(cluster)
    credential = body["k8s_options"]["imageCredentials"][0]
    credential["password"] = "**********"
    credential[field] = "different"
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 400
    assert "private-" not in response.text
    assert cluster.k8s_options.image_credentials[0].password == "private-password-1"


def test_ambiguous_identity_requires_an_explicit_password(client, cluster):
    cluster.k8s_options.image_credentials.append(
        ImageCredential(
            registry="registry-1.example", username="user", password="private-other"
        )
    )
    response = client.put("/v2/clusters/1", json=_update_payload(cluster))
    assert response.status_code == 400
    assert "private-" not in response.text


@pytest.mark.parametrize("value", ["omitted", None, []])
def test_image_credential_list_can_be_preserved_or_removed(client, cluster, value):
    body = _update_payload(cluster)
    if value == "omitted":
        del body["k8s_options"]["imageCredentials"]
    else:
        body["k8s_options"]["imageCredentials"] = value
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 200, response.text
    if value == "omitted":
        assert len(cluster.k8s_options.image_credentials) == 2
        assert cluster.k8s_options.image_credentials[0].password == "private-password-1"
    else:
        assert cluster.k8s_options.image_credentials == value


def test_omitted_configuration_keeps_stored_credentials(client, cluster):
    response = client.put("/v2/clusters/1", json={"name": cluster.name})
    assert response.status_code == 200, response.text
    assert cluster.worker_config.huggingface_token == "private-hf-token"
    assert cluster.k8s_options.image_credentials[0].password == "private-password-1"


def test_explicit_null_worker_configuration_clears_its_token(client, cluster):
    response = client.put(
        "/v2/clusters/1", json={"name": cluster.name, "worker_config": None}
    )
    assert response.status_code == 200, response.text
    assert cluster.worker_config is None


def test_new_registry_without_password_does_not_inherit_one(client, cluster):
    body = _update_payload(cluster)
    body["k8s_options"]["imageCredentials"][0]["registry"] = "new.example"
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 200, response.text
    assert cluster.k8s_options.image_credentials[0].password is None


@pytest.mark.parametrize("field", ["huggingface_token", "password"])
def test_mask_without_stored_secret_is_rejected(client, cluster, field):
    body = _update_payload(cluster)
    if field == "huggingface_token":
        cluster.worker_config.huggingface_token = None
        body["worker_config"][field] = "**********"
    else:
        cluster.k8s_options.image_credentials[0].password = None
        body["k8s_options"]["imageCredentials"][0][field] = "**********"
    response = client.put("/v2/clusters/1", json=body)
    assert response.status_code == 400


@pytest.mark.parametrize("field", [None, "huggingface_token", "password"])
def test_create_masks_response_and_refuses_unresolved_masks(
    monkeypatch, client, cluster, field
):
    async def create(session, source, **kwargs):
        source.id = cluster.id
        source.created_at = cluster.created_at
        source.updated_at = cluster.updated_at
        return source

    create_record = AsyncMock(side_effect=create)
    monkeypatch.setattr(Cluster, "one_by_fields", AsyncMock(return_value=None))
    monkeypatch.setattr(Cluster, "first_by_field", AsyncMock(return_value=None))
    monkeypatch.setattr(Cluster, "create", create_record)
    monkeypatch.setattr(Cluster, "save", AsyncMock())
    monkeypatch.setattr(
        clusters.Principal, "create", AsyncMock(return_value=SimpleNamespace(id=501))
    )
    monkeypatch.setattr(clusters.ApiKey, "create", AsyncMock())
    monkeypatch.setattr(clusters, "platform_principal_id", lambda: 0)
    body = {
        "name": cluster.name,
        "provider": "Kubernetes",
        "owner_principal_id": 101,
        "worker_config": cluster.worker_config.model_dump(exclude_none=True),
        "k8s_options": cluster.k8s_options.model_dump(by_alias=True, exclude_none=True),
    }
    if field == "huggingface_token":
        body["worker_config"][field] = "**********"
    elif field == "password":
        body["k8s_options"]["imageCredentials"][0][field] = "**********"
    response = client.post("/v2/clusters", json=body)
    if field is not None:
        assert response.status_code == 400
        create_record.assert_not_awaited()
        return
    assert response.status_code == 200, response.text
    _assert_masked(response.json())
    stored = create_record.call_args.args[1]
    assert stored.worker_config.huggingface_token == "private-hf-token"
    assert stored.k8s_options.image_credentials[0].password == "private-password-1"


@pytest.mark.parametrize("scope", ["member", "shared"])
def test_readers_cannot_update_or_retrieve_registration_secrets(client, cluster, scope):
    client.app.dependency_overrides[get_tenant_context] = lambda: _context(scope)
    response = client.put("/v2/clusters/1", json=_update_payload(cluster))
    assert response.status_code == 403
    for endpoint in ("registration-token", "manifests"):
        response = client.get(f"/v2/clusters/1/{endpoint}")
        assert response.status_code == 403
        assert "private-" not in response.text
    assert cluster.worker_config.huggingface_token == "private-hf-token"


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
def test_storage_and_registration_keep_real_credentials(monkeypatch, cluster, dialect):
    for name in ("worker_config", "k8s_options"):
        column_type = Cluster.__table__.c[name].type
        encoded = column_type.bind_processor(dialect)(getattr(cluster, name))
        restored = column_type.result_processor(dialect, None)(encoded)
        assert restored == getattr(cluster, name)
        assert "private-" in encoded
        assert "**********" not in encoded

    monkeypatch.setattr(
        clusters, "get_cluster_image_name", lambda *_: "gpustack/test:dev"
    )
    registration = clusters.get_registration_from_cluster(
        SimpleNamespace(url=SimpleNamespace(scheme="http", netloc="server")), cluster
    )
    assert registration.env["HF_TOKEN"] == "private-hf-token"
    config = TemplateConfig(registration=registration, k8s_options=cluster.k8s_options)
    secret = json.loads(
        base64.b64decode(config.image_pull_secrets[0].dockerconfigjson_b64)
    )
    assert secret["auths"]["registry-1.example"]["password"] == "private-password-1"
