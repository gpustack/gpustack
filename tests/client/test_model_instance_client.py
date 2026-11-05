from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.client import ClientSet
from gpustack.client.model_instance_client import ModelInstanceClient
from gpustack.schemas.models import (
    ModelInstancePublic,
    ModelInstanceWorkloadUpdate,
    ModelInstanceStopReport,
    ModelInstanceUpdate,
    SourceEnum,
)
from tests.utils.model import new_model_instance


@pytest.fixture
def instance():
    return ModelInstancePublic.model_validate(
        new_model_instance(7, 'test', 1),
        update={
            'source': SourceEnum.HUGGING_FACE,
            'huggingface_repo_id': 'org/repo',
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        },
    )


def test_clientset_uses_the_instance_client():
    clientset = ClientSet('http://server.example')
    assert isinstance(clientset.model_instances, ModelInstanceClient)
    assert clientset.model_instances._client is clientset.http_client


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('watching', [False, True])
@pytest.mark.parametrize('present', [False, True])
def test_cached_lookup_never_falls_back_to_http(instance, enabled, watching, present):
    transport = MagicMock()
    client = ModelInstanceClient(transport, enable_cache=enabled)
    client._watch_started = watching
    client._cache = {instance.id: instance} if present else {}

    if enabled and watching:
        assert client.get_cached(instance.id) is (instance if present else None)
    else:
        with pytest.raises(ServiceUnavailableException):
            client.get_cached(instance.id)
    transport.get_httpx_client.assert_not_called()


@pytest.mark.parametrize(
    'operation,payload,path,method',
    [
        (
            'update',
            ModelInstanceUpdate(
                name='updated',
                model_name='test',
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id='org/repo',
            ),
            '/model-instances/7',
            'put',
        ),
        (
            'report_workloads',
            ModelInstanceWorkloadUpdate(restart_count=3, workloads=[]),
            '/model-instances/7/workload-report',
            'put',
        ),
        (
            'confirm_stop',
            ModelInstanceStopReport(token=uuid4(), restart_count=3),
            '/model-instances/7/stop-confirmation',
            'post',
        ),
    ],
)
def test_requests_use_their_dedicated_endpoint(
    instance, operation, payload, path, method
):
    transport = MagicMock()
    send = getattr(transport.get_httpx_client.return_value, method)
    send.return_value = httpx.Response(200, json=instance.model_dump(mode='json'))
    client = ModelInstanceClient(transport)
    assert getattr(client, operation)(instance.id, payload) == instance
    send.assert_called_once_with(
        path,
        content=payload.model_dump_json(),
        headers={'Content-Type': 'application/json'},
    )


@pytest.mark.parametrize('resource', ['models', 'model_instances'])
@pytest.mark.parametrize('completed', [False, True])
def test_sdk_delete_accepts_pending_or_completed_null_body(resource, completed):
    clientset = ClientSet('http://server.example')
    client = getattr(clientset, resource)
    transport = MagicMock()
    client._client = transport
    transport.get_httpx_client.return_value.delete.return_value = httpx.Response(
        200 if completed else 202,
        json=None,
    )
    assert client.delete(7) is None
    transport.get_httpx_client.return_value.delete.assert_called_once_with(
        f'{client._url}/7'
    )


def test_sdk_restart_exposes_pending_replacement_startup():
    clientset = ClientSet('http://server.example')
    transport = MagicMock()
    clientset.models._client = transport
    transport.get_httpx_client.return_value.post.return_value = httpx.Response(
        202,
        json=dict(
            accepted=True,
            completed=False,
            phase='restarting',
            spec_digest='sha1:target',
            restarted=True,
            deleted_instances=['old'],
            observe_url='/v2/models/7',
        ),
    )
    result = clientset.models.restart(7)
    assert result.completed is False
    assert result.phase == 'restarting'
    assert result.spec_digest == 'sha1:target'
    assert result.deleted_instances == ['old']
    transport.get_httpx_client.return_value.post.assert_called_once_with(
        '/models/7/restart'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('resource', ['models', 'model_instances'])
@pytest.mark.parametrize('present', [False, True])
async def test_id_only_delete_enriches_from_cache_without_extra_status(
    instance, resource, present
):
    from gpustack.schemas.models import ModelPublic
    from gpustack.server.bus import Event, EventType

    clientset = ClientSet('http://server.example')
    api = getattr(clientset, resource)
    cached = (
        instance
        if resource == 'model_instances'
        else ModelPublic(
            id=7,
            name='test',
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id='org/repo',
            created_at=instance.created_at,
            updated_at=instance.updated_at,
        )
    )
    before = cached.model_dump()
    api._cache = {7: cached} if present else {}
    event = Event(type=EventType.DELETED, id=7, data={'id': 7})
    await api._update_cache_from_event(event)
    if present:
        assert event.data is cached
        assert event.data.model_dump() == before
        assert 'lifecycle' not in event.data.model_dump()
    else:
        assert event.data == {'id': 7}
    assert cached.model_dump() == before
    assert api._cache == {}


@pytest.mark.parametrize('status', [200, 409])
def test_configuration_checked_get_bypasses_watch_cache(instance, status):
    from gpustack.api.exceptions import HTTPException

    transport = MagicMock()
    send = transport.get_httpx_client.return_value.get
    send.return_value = httpx.Response(
        status,
        json=(
            instance.model_dump(mode='json')
            if status == 200
            else {'code': 409, 'reason': 'Conflict', 'message': 'Configuration changed'}
        ),
    )
    client = ModelInstanceClient(transport)
    client._watch_started = True
    client._cache = {instance.id: instance}
    if status == 409:
        with pytest.raises(HTTPException) as exc:
            client.get(instance.id, check_configuration=True)
        assert exc.value.status_code == 409
    else:
        assert client.get(instance.id, check_configuration=True) == instance
    send.assert_called_once_with(
        '/model-instances/7', params={'check_configuration': True}
    )
