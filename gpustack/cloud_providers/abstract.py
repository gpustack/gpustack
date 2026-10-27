from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from gpustack.schemas.clusters import Volume
from gpustack.cloud_providers.user_data import UserDataTemplate


class InstanceState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    # Provisioning failed on the provider side. Terminal: unlike CREATED it
    # will never become RUNNING, so waiters must give up instead of polling to
    # their timeout.
    FAILED = "failed"
    UNKNOWN = "unknown"


class InstanceProvisioningFailed(Exception):
    """
    The provider gave up creating the instance.

    Distinct from a timeout: waiting longer or polling again cannot help, so
    callers should surface it instead of retrying. The provisioning controller
    treats it as terminal and moves the worker to ERROR.
    """


@dataclass
class CloudInstanceCreate:
    name: str
    image: str
    type: str
    region: str
    # None when the provider has no SSH key API, i.e. create_ssh_key
    # registered nothing to attach by id.
    ssh_key_id: Optional[str] = None
    user_data: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    # Token identifying one attempt at creating this instance, for providers
    # that offer replay protection. Stable across retries of the same attempt,
    # different for a later one. See construct_cloud_instance.
    idempotency_key: Optional[str] = None


@dataclass
class CloudInstance(CloudInstanceCreate):
    external_id: Optional[str] = None
    status: InstanceState = InstanceState.CREATED
    ip_address: Optional[str] = None
    ssh_key_id: Optional[str] = None
    volume_ids: Optional[List[str]] = None
    # ``(host, port)`` SSH actually answers on, when that is not
    # ``ip_address:22``. Providers that publish instances behind a shared
    # address with mapped ports need it, and it is the only way a caller can
    # build a working connection hint. None means use ip_address:22.
    ssh_endpoint: Optional[Tuple[str, int]] = None
    ssh_user: Optional[str] = None


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
    async def create_ssh_key(self, worker_name: str, public_key: str) -> Optional[str]:
        """
        Register the public key with the provider and return its id, which the
        caller records on ``Credential.external_id``.

        Return None when there is nothing to register, i.e. the provider has no
        SSH key API and the key travels inside the user data instead (see
        :meth:`construct_user_data`). The caller then leaves ``external_id``
        NULL and skips :meth:`delete_ssh_key` on teardown.
        """
        pass

    @abstractmethod
    async def delete_ssh_key(self, id: str):
        pass

    @abstractmethod
    async def create_volumes_and_attach(
        self, worker_id: int, external_id: str, region: str, *volumes: Volume
    ) -> List[str]:
        """
        Create volumes and attach them to the instance.
        Volumes should be tuple of {"size_gb": 10, "format": "ext4", "name": "my-volume"}, the name is optional.
        """
        pass

    async def construct_user_data(
        self,
        server_url: str,
        token: str,
        image_name: str,
        os_image: str,
        worker_name: str,
        secret_configs: Dict[str, Any] = {},
        ssh_public_key: Optional[str] = None,
    ) -> UserDataTemplate:
        """
        Build the cloud-init document for a worker.

        ``ssh_public_key`` is the key created by :meth:`create_ssh_key`, handed
        over so each provider can decide how the instance ends up trusting it.
        The base implementation ignores it: providers with an SSH key API
        (DigitalOcean) pass the key's id on ``CloudInstanceCreate.ssh_key_id``
        and let the API inject it. A provider without such an API overrides
        this and calls ``user_data.add_ssh_authorized_keys(ssh_public_key)``,
        so the key travels inside the user data instead.
        """
        user_data = UserDataTemplate(
            server_url=server_url,
            token=token,
            image_name=image_name,
            secret_configs=secret_configs,
            worker_name=worker_name,
        )
        return user_data

    @classmethod
    @abstractmethod
    def get_api_endpoint(cls) -> str:
        """
        Base URL of the provider's API.

        Abstract rather than defaulting to ``""``: a provider with no endpoint
        cannot be talked to, and an empty one is not inert. It reaches
        ``urljoin("", path)`` in the credential proxy, which yields a relative
        URL that the route's same-host check waves through — the request then
        dies inside the HTTP client as a generic 500. Requiring an
        implementation moves that from a puzzling failure at request time to an
        obvious one when the provider is written.
        """
        ...

    @classmethod
    def process_header(cls, ak: str, sk: str, options: dict, headers: dict) -> dict:
        return headers
