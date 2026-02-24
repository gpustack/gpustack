import logging
import random
import string
import asyncio
from typing import List, Optional, Dict, Any
from .abstract import (
    ProviderClientBase,
    CloudInstance,
    CloudInstanceCreate,
    InstanceState,
)
from pydo.aio import Client
from gpustack.schemas.clusters import Volume
from gpustack.cloud_providers.user_data import UserDataTemplate
from gpustack_runtime.detector import ManufacturerEnum

logger = logging.getLogger(__name__)

status_mapping = {
    "new": InstanceState.CREATED,
    "active": InstanceState.RUNNING,
}


class DigitalOceanClient(ProviderClientBase):
    client: Client

    def __init__(self, token: str):
        self.client = Client(token=token, timeout=30)

    async def create_instance(self, instance: CloudInstanceCreate) -> str:
        tags: List[str] = [f"{k}:{v}" for k, v in instance.labels.items()]
        req = {
            "name": instance.name,
            "image": instance.image,
            "size": instance.type,
            "region": instance.region,
            "ssh_keys": [instance.ssh_key_id],
            "user_data": instance.user_data,
            "tags": tags,
        }
        try:
            logger.info(f"Creating digital ocean droplet with name {instance.name}")
            logger.debug(f"Request body: {req}")
            droplet_resp = await self.client.droplets.create(body=req)
            id = droplet_resp['droplet']['id']
            return str(id)
        except Exception as e:
            logger.error(f"Failed to create digital ocean instance: {e}")
            raise e

    async def delete_instance(self, external_id: str):
        logger.info(f"Deleting digital ocean instance with id {external_id}")
        delete_response = (
            await self.client.droplets.destroy_with_associated_resources_dangerous(
                external_id,
                x_dangerous=True,
            )
        )
        if delete_response is None:
            return
        logger.warning(
            f"Failed to delete droplet {external_id}, Delete response: {delete_response}"
        )
        raise RuntimeError(
            f"Failed to delete droplet {external_id}, {delete_response.message}"
        )

    async def get_instance(self, external_id: str) -> Optional[CloudInstance]:
        response = await self.client.droplets.get(external_id)
        instance: Dict[str, Any] = response.get('droplet', None)
        if instance is None:
            return None
        ip_address = None
        v4_list = instance.get('networks', {}).get('v4', [])
        for net in v4_list:
            if net.get('type') == 'public':
                ip_address = net.get('ip_address')
                break
        status: InstanceState = status_mapping.get(
            instance.get('status'), InstanceState.UNKNOWN
        )
        return CloudInstance(
            external_id=str(instance.get('id')),
            name=instance.get('name'),
            image=instance.get('image', {}).get('slug', ''),
            type=instance.get('size_slug'),
            region=instance.get('region', {}).get('slug', ''),
            ssh_key_id=None,
            volume_ids=instance.get('volume_ids', []),
            user_data=None,
            status=status,
            ip_address=ip_address,
        )

    async def wait_for_started(
        self, external_id: str, backoff: int = 15, limit: int = 20
    ) -> CloudInstance:
        for _ in range(limit):
            instance = await self.get_instance(external_id)
            if instance and instance.status == InstanceState.RUNNING:
                return instance
            await asyncio.sleep(backoff)
        raise TimeoutError(
            f"DigitalOcean droplet {external_id} did not start within {limit} retries"
        )

    async def wait_for_public_ip(
        self, external_id: str, backoff: int = 15, limit: int = 20
    ) -> CloudInstance:
        for _ in range(limit):
            instance = await self.get_instance(external_id)
            if (
                instance
                and instance.ip_address is not None
                and instance.ip_address != ""
            ):
                return instance
            await asyncio.sleep(backoff)
        raise TimeoutError(
            f"DigitalOcean droplet {external_id} did not acquire a public IP within {limit} retries"
        )

    async def create_ssh_key(self, worker_name: str, public_key: str) -> str:
        ssh_key_resp = await self.client.ssh_keys.create(
            body={"name": f"sshkey-{worker_name}", "public_key": public_key},
        )
        id = ssh_key_resp['ssh_key']['id']
        return str(id)

    async def delete_ssh_key(self, id: str):
        await self.client.ssh_keys.delete(id)

    async def create_volumes_and_attach(
        self, worker_id: int, external_id: str, region: str, *volumes: Volume
    ) -> List[str]:
        # validate volumes
        volume_ids = []
        if len(volumes) == 0:
            return volume_ids
        for idx, volume in enumerate(volumes):
            size_gb = volume.size_gb
            if size_gb is None or size_gb <= 0:
                raise ValueError(
                    f"Volume #{idx} missing or invalid 'size_gb': {volume}"
                )
            format = volume.format
            if format is None or format not in ['ext4', 'xfs']:
                raise ValueError(f"Volume #{idx} missing or invalid 'format': {volume}")
            if len(format) > (16 - 2 - 2 - 2):
                # 16 is max label length, 2 for underscores, 2 for index digits and 2 for hashed prefix
                raise ValueError(f"Volume #{idx} 'format' too long: {volume}")
        random_prefix = ''.join(random.choices(string.ascii_lowercase, k=6))
        for volume in volumes:
            index = volumes.index(volume)
            label = f"{random_prefix}_{volume.format}_{index}"
            if len(label) > 16:
                label = label[-16:]
            name = (
                f'{volume.name}-{worker_id}' if volume.name else label.replace('_', '-')
            )
            logger.info(
                f"Creating volume {name} of size {volume.size_gb}GB in region {region}"
            )
            vol_resp = await self.client.volumes.create(
                body={
                    "size_gigabytes": volume.size_gb,
                    "name": name,
                    "region": region,
                    "filesystem_type": volume.format,
                    "filesystem_label": label,
                },
            )
            vol_id = vol_resp['volume']['id']
            volume_ids.append(str(vol_id))
            logger.info(f"Attaching volume {vol_id} to droplet {external_id}")
            resp = await self.client.volume_actions.post_by_id(
                volume_id=vol_id,
                body={"type": "attach", "droplet_id": external_id, "region": region},
            )
            id: str = resp.get('id', None)
            message: str = resp.get('message', None)
            if id is not None and message is not None:
                logger.error(
                    f"Failed to attach volume {vol_id} to droplet {external_id}, response: {message}"
                )
                raise RuntimeError(
                    f"Failed to attach volume {vol_id} to droplet {external_id}, response: {message}"
                )
        return volume_ids

    async def construct_user_data(
        self,
        server_url,
        token,
        image_name,
        os_image,
        worker_name,
        secret_configs: Dict[str, Any] = {},
    ) -> UserDataTemplate:
        image_info = await self.client.images.get(os_image)
        distribution = image_info.get('image', {}).get('distribution', '').lower()
        image_slug = image_info.get('image', {}).get('slug', '').lower()
        setup_driver = None
        install_driver = None
        # This is a trick to find out AI/ML ready images of DigitalOcean
        if image_slug.startswith("gpu"):
            # AMD will not take affect for now
            if os_image.lower().find("amd") != -1:
                setup_driver = ManufacturerEnum.AMD
            else:
                setup_driver = ManufacturerEnum.NVIDIA
        elif distribution in ['ubuntu', 'debian']:
            install_driver = ManufacturerEnum.NVIDIA
            setup_driver = ManufacturerEnum.NVIDIA
        user_data = await super().construct_user_data(
            server_url, token, image_name, os_image, worker_name, secret_configs
        )
        user_data.distribution = distribution
        user_data.setup_driver = setup_driver
        user_data.install_driver = install_driver
        user_data.insert_runcmd(
            "mkdir -p /var/lib/gpustack",
            "curl -s http://169.254.169.254/metadata/v1/id > /var/lib/gpustack/external_id",
            'ip=$(curl -s http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address); ip_lc=$(echo "$ip" | tr "A-Z" "a-z"); if [ "$ip_lc" != "not found" ]; then echo "$ip" > /var/lib/gpustack/advertise_address; fi',
        )
        return user_data

    @classmethod
    def get_api_endpoint(cls) -> str:
        return "https://api.digitalocean.com"

    @classmethod
    def process_header(cls, ak: str, sk: str, options: dict, headers: dict) -> dict:
        headers["Authorization"] = f"Bearer {sk}"
        return headers
