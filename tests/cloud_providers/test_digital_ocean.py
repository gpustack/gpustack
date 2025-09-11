import pytest
from gpustack.cloud_providers.digital_ocean import DigitalOceanClient
from gpustack.cloud_providers.abstract import CloudInstanceCreate, InstanceState
from gpustack.cloud_providers.abstract import Volume


@pytest.mark.asyncio
async def test_generate_user_data(do_client):
    image_name = 'gpustack/worker:latest'
    registration_token = 'test-token'
    server_url = 'http://localhost:8080'
    user_data = do_client.generate_user_data(
        image_name=image_name,
        registration_token=registration_token,
        server_url=server_url,
    )
    assert user_data.startswith('#cloud-config')
    assert 'docker run' in user_data
    assert registration_token in user_data
    assert server_url in user_data
    import yaml

    yaml.safe_load(user_data)  # Ensure it's valid YAML


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    class droplets:
        @staticmethod
        def create(body):
            return {'droplet': {'id': '12345'}}

        @staticmethod
        def destroy(external_id):
            return None

        @staticmethod
        def destroy_with_associated_resources_dangerous(
            external_id: str, x_dangerous: bool
        ):
            return None

        @staticmethod
        def get(external_id):
            return {
                'droplet': {
                    'id': external_id,
                    'name': 'test-droplet',
                    'image': {'slug': 'ubuntu-20-04-x64'},
                    'size_slug': 's-1vcpu-1gb',
                    'region': {'slug': 'nyc3'},
                    'networks': {'v4': [{'type': 'public', 'ip_address': '1.2.3.4'}]},
                    'status': 'active',
                    'volume_ids': ['vol-1'],
                }
            }

    class ssh_keys:
        @staticmethod
        def create(body):
            return {'ssh_key': {'id': 'ssh-1'}}

        @staticmethod
        def delete(id):
            return None

    class volumes:
        @staticmethod
        def create(body):
            return {'volume': {'id': 'vol-1'}}

    class volume_actions:
        @staticmethod
        def post_by_id(volume_id, body):
            return {"action": {"status": "completed"}}


@pytest.fixture
def do_client(monkeypatch):
    monkeypatch.setattr('gpustack.cloud_providers.digital_ocean.Client', DummyClient)
    return DigitalOceanClient(token='dummy-token')


@pytest.mark.asyncio
async def test_create_instance(do_client):
    instance = CloudInstanceCreate(
        name='test-droplet',
        image='ubuntu-20-04-x64',
        type='s-1vcpu-1gb',
        region='nyc3',
        ssh_key_id='ssh-1',
        user_data=None,
        labels={'env': 'test'},
    )
    droplet_id = await do_client.create_instance(instance)
    assert droplet_id == '12345'


@pytest.mark.asyncio
async def test_delete_instance(do_client):
    await do_client.delete_instance('12345')


@pytest.mark.asyncio
async def test_get_instance(do_client):
    instance = await do_client.get_instance('12345')
    assert instance is not None
    assert instance.external_id == '12345'
    assert instance.status == InstanceState.RUNNING
    assert instance.ip_address == '1.2.3.4'


@pytest.mark.asyncio
async def test_wait_for_started(do_client):
    instance = await do_client.get_instance('12345')
    started = await do_client.wait_for_started(instance, backoff=0, limit=2)
    assert started.status == InstanceState.RUNNING


@pytest.mark.asyncio
async def test_create_ssh_key(do_client):
    ssh_id = await do_client.create_ssh_key(
        'worker1', 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC'
    )
    assert ssh_id == 'ssh-1'


@pytest.mark.asyncio
async def test_delete_ssh_key(do_client):
    await do_client.delete_ssh_key('ssh-1')


@pytest.mark.asyncio
async def test_create_volumes_and_attach(do_client):
    volume_ids = await do_client.create_volumes_and_attach(
        '12345', 'nyc3', Volume(size_gb=10, format='ext4')
    )
    assert volume_ids == ['vol-1']
