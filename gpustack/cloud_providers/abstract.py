import jinja2
from dataclasses import dataclass
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from enum import Enum
from gpustack.schemas.clusters import Volume


class InstanceState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass
class CloudInstanceCreate:
    name: str
    image: str
    type: str
    region: str
    ssh_key_id: str
    user_data: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


@dataclass
class CloudInstance(CloudInstanceCreate):
    external_id: Optional[str] = None
    status: InstanceState = InstanceState.CREATED
    ip_address: Optional[str] = None
    ssh_key_id: Optional[str] = None
    volume_ids: Optional[List[str]] = None


default_cloudinit_template = """#cloud-config
package_update: true
packages:
- docker.io  # For Debian/Ubuntu
- docker-engine # For CentOS/RHEL (adjust as needed for specific distributions)
runcmd:
- systemctl enable docker
- systemctl start docker
- |
    docker run -d --name gpustack-worker \
    --restart=unless-stopped --net=host --ipc=host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /var/lib/gpustack:/var/lib/gpustack \
    {{ image_name }} \
    --server-url {{ server_url }} \
    --registration-token {{ registration_token }}
"""


class ProviderClientBase(ABC):
    """
    The lifecycle is like:
    1. create_ssh_key
    2. create_instance with created ssh_key
    3. wait_for_started
    4. wait_for_public_ip
    5. [optional] create_volumes_and_attach
    6. delete_instance
    7. [optional] delete_ssh_key
    """

    @abstractmethod
    async def create_instance(self, instance: CloudInstanceCreate) -> Optional[str]:
        pass

    @abstractmethod
    async def delete_instance(self, external_id: str):
        pass

    @abstractmethod
    async def get_instance(self, external_id: str) -> Optional[CloudInstance]:
        pass

    @abstractmethod
    async def wait_for_started(
        self, external_id: str, backoff: int = 5, limit: int = 60
    ) -> CloudInstance:
        pass

    @abstractmethod
    async def wait_for_public_ip(
        self, external_id: str, backoff: int = 5, limit: int = 60
    ) -> CloudInstance:
        pass

    @abstractmethod
    async def create_ssh_key(self, worker_name: str, public_key: str) -> str:
        pass

    @abstractmethod
    async def delete_ssh_key(self, id: str):
        pass

    @abstractmethod
    async def create_volumes_and_attach(
        self, external_id: str, region: str, *volumes: Volume
    ) -> List[str]:
        """
        Create volumes and attach them to the instance.
        Volumes should be tuple of {"size_gb": 10, "format": "ext4", "name": "my-volume"}, the name is optional.
        """
        pass

    @classmethod
    def generate_user_data(
        cls, image_name: str, registration_token: str, server_url: str
    ) -> str:
        return jinja2.Template(default_cloudinit_template).render(
            image_name=image_name,
            registration_token=registration_token,
            server_url=server_url,
        )

    @classmethod
    def get_api_endpoint(cls) -> str:
        return ""

    @classmethod
    def process_header(cls, ak: str, sk: str, options: dict, headers: dict) -> dict:
        return headers
