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
    ModelInstanceRankUpdate,
    ModelInstanceStopUpdate,
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
    'update',
    [
        ModelInstanceUpdate(
            name='updated',
            model_name='test',
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id='org/repo',
        ),
        ModelInstanceRankUpdate(rank_report={'restart_count': 3, 'workloads': []}),
        ModelInstanceStopUpdate(stop_report={'token': uuid4(), 'restart_count': 3}),
    ],
    ids=['instance', 'rank', 'stop'],
)
def test_update_preserves_each_request_envelope(instance, update):
    transport = MagicMock()
    transport.get_httpx_client.return_value.put.return_value = httpx.Response(
        200, json=instance.model_dump(mode='json')
    )
    client = ModelInstanceClient(transport)

    assert client.update(instance.id, update) == instance
    transport.get_httpx_client.return_value.put.assert_called_once_with(
        '/model-instances/7',
        content=update.model_dump_json(),
        headers={'Content-Type': 'application/json'},
    )
