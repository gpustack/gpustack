from dataclasses import dataclass
from typing import Optional, List, Dict, Any
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

    @abstractmethod
    async def determine_linux_distribution(self, image_id: str) -> Optional[str]:
        """
        Determine the linux distribution of the instance.
        Return values can be: "ubuntu", "debian", "centos", "rocky", "almalinux", "unknown"
        """
        pass

    @classmethod
    def modify_cloud_init(cls, user_data: Dict[str, Any]):
        pass

    @classmethod
    def get_api_endpoint(cls) -> str:
        return ""

    @classmethod
    def process_header(cls, ak: str, sk: str, options: dict, headers: dict) -> dict:
        return headers
