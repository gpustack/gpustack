"""
Async SDK for the Shuihua open API (基于 API key 认证的开放接口, v1.4.0).

Generated from ``open-api.openapi_v5.json``. The API wraps every successful
response in ``{"data": ...}`` (list endpoints add ``{"meta": ...}``) and every
failure in ``{"error": {"code": ..., "message": ...}}``; this client unwraps
both and raises :class:`ShuihuaAPIError` on failure.

Creation is asynchronous: ``create_vm`` returns as soon as the lease is
accepted (HTTP 202, ``status=creating``) and the caller polls
:meth:`ShuihuaClient.get_vm` until the status reaches ``active`` or ``failed``.

Usage::

    async with ShuihuaClient(api_key="amp_live_xxx") as client:
        specs = await client.list_gpu_instances()
        images = await client.list_images()
        vm = await client.create_vm(
            template_id=specs[0].template_id,
            image_id=images[0].image_uuid,
            idempotency_key="my-stable-token",
            user_data="#cloud-config\\npackages:\\n  - htop",
        )
        while vm.status in PENDING_STATUSES:
            vm = await client.get_vm(vm.id)
        await client.terminate_vm(vm.id)
"""

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, field_validator
from gpustack_runtime.detector import ManufacturerEnum

from gpustack.config.config import get_global_config
from gpustack.schemas.clusters import Volume
from gpustack.cloud_providers.user_data import UserDataTemplate
from .abstract import (
    CloudInstance,
    CloudInstanceCreate,
    InstanceProvisioningFailed,
    InstanceState,
    ProviderClientBase,
)

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1/open"

# Default read timeout for normal calls. Creation only has to be accepted, not
# completed, so it needs no timeout of its own.
DEFAULT_TIMEOUT = 30
# cloud-init user data is capped at 64KB by the server.
MAX_USER_DATA_SIZE = 65536
# Internal port the API defaults to when an instance has no port mapping.
DEFAULT_SSH_PORT = 22
# The API caps the idempotency key and the instance name at 64 characters.
MAX_IDEMPOTENCY_KEY_SIZE = 64
MAX_INSTANCE_NAME_SIZE = 64


class VMStatus(str, Enum):
    CREATING = "creating"
    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    EXPIRED = "expired"
    TERMINATED = "terminated"


# Statuses a VM passes through before it is either usable or lost.
PENDING_STATUSES = frozenset({VMStatus.CREATING, VMStatus.PROCESSING})

# A lease that expired is powered off but still exists, unlike a terminated one.
status_mapping = {
    # creating = queued for a backend worker to claim, processing = the flint
    # instance is being built. Both are on their way to active.
    VMStatus.CREATING: InstanceState.CREATED,
    VMStatus.PROCESSING: InstanceState.CREATED,
    VMStatus.ACTIVE: InstanceState.RUNNING,
    VMStatus.FAILED: InstanceState.FAILED,
    VMStatus.EXPIRED: InstanceState.STOPPED,
    VMStatus.TERMINATED: InstanceState.TERMINATED,
}


class GpuInstance(BaseModel):
    """An orderable spec template: GPU model, hourly price and remaining stock."""

    template_id: Optional[int] = None
    name: Optional[str] = None
    gpu_model: Optional[str] = None
    price_per_hour: Optional[float] = Field(
        default=None, description="Hourly price in CNY"
    )
    remaining: Optional[int] = Field(default=None, description="Remaining available")


class Image(BaseModel):
    id: Optional[int] = None
    image_uuid: Optional[str] = Field(
        default=None, description="Pass as image_id when creating a VM"
    )
    name: Optional[str] = None
    os_distro: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PortMapping(BaseModel):
    """One public endpoint mapped onto an internal port of the instance."""

    public_ip: Optional[str] = None
    public_port: Optional[int] = None
    internal_port: Optional[int] = Field(
        default=None, description="Internal port, e.g. 22 for SSH or 80 for HTTP"
    )


class VMSummary(BaseModel):
    id: Optional[int] = None
    template_id: Optional[int] = None
    template_name: Optional[str] = None
    gpu_model: Optional[str] = None
    instance_id: Optional[str] = None
    instance_name: Optional[str] = None
    ip_address: Optional[str] = Field(default=None, description="Instance IP")
    port_mappings: Optional[List[PortMapping]] = Field(
        default=None,
        description=(
            "Port mappings; one internal IP may expose several public ports. "
            "Absent when the instance has no mapping."
        ),
    )
    internal_port: Optional[int] = Field(
        default=None,
        description=(
            "Default internal port (22), only returned when there is no port "
            "mapping. See port_mappings otherwise."
        ),
    )
    status: Optional[VMStatus] = None
    start_time: Optional[datetime] = None

    @field_validator("status", mode="before")
    @classmethod
    def _tolerate_unknown_status(cls, value):
        """Degrade an unrecognised status to None instead of failing the parse.

        The status set grew by three values between API 1.3.0 and 1.4.0, and a
        strict enum would turn the next addition into a hard error — not just
        for that VM but for the whole ``list_vms`` page it appears on, taking
        every worker on the credential down with it. None maps to
        ``InstanceState.UNKNOWN``, so callers keep polling and the warning says
        why.
        """
        if value is None or isinstance(value, VMStatus):
            return value
        try:
            return VMStatus(value)
        except ValueError:
            logger.warning(f"Ignoring unknown Shuihua VM status {value!r}")
            return None

    def public_endpoint(
        self, internal_port: int = DEFAULT_SSH_PORT
    ) -> Optional[Tuple[str, int]]:
        """Resolve ``(public_ip, public_port)`` for an internal port.

        Returns ``None`` when the instance has no mapping for that port, in which
        case reach it directly on ``ip_address``.
        """
        for mapping in self.port_mappings or []:
            if mapping.internal_port == internal_port:
                if mapping.public_ip and mapping.public_port:
                    return mapping.public_ip, mapping.public_port
                return None
        return None


class VMDetail(VMSummary):
    ssh_user: Optional[str] = None
    ssh_private_key: Optional[str] = Field(
        default=None,
        description=(
            "PEM private key, only returned for VMs created without user_data. "
            "Empty in self-contained (user_data) mode."
        ),
    )
    end_time: Optional[datetime] = Field(default=None, description="Lease end time")


class PaginationMeta(BaseModel):
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None


class ShuihuaAPIError(Exception):
    """Raised when the API returns a non-2xx response or an error envelope."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        payload: Optional[Any] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.payload = payload
        super().__init__(
            f"Shuihua API error (status={status_code}, code={code}): {message}"
        )


class ShuihuaClient:
    """Async client for the Shuihua open API.

    Authentication is a bearer API key (``amp_live_`` prefixed).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """``client`` lets the caller inject a shared/mocked ``AsyncClient``; it is
        not closed by this class and every request carries an absolute URL, so its
        own ``base_url`` and headers are irrelevant."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def __aenter__(self) -> "ShuihuaClient":
        return self

    async def __aexit__(self, *_exc_info):
        await self.aclose()

    async def aclose(self):
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[Any, Optional[PaginationMeta]]:
        """Send a request and unwrap the ``data`` / ``meta`` envelope."""
        url = f"{self._base_url}{API_PREFIX}{path}"
        try:
            response = await self._client.request(
                method,
                url,
                params=params,
                json=json,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as e:
            raise ShuihuaAPIError(f"Request to {method} {url} failed: {e}") from e

        try:
            body = response.json()
        except ValueError:
            body = None

        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            error = body["error"]
            raise ShuihuaAPIError(
                error.get("message", "unknown error"),
                status_code=response.status_code,
                code=error.get("code"),
                payload=body,
            )

        if response.is_error:
            raise ShuihuaAPIError(
                body if isinstance(body, str) else response.text,
                status_code=response.status_code,
                payload=body,
            )

        if not isinstance(body, dict):
            raise ShuihuaAPIError(
                f"Unexpected response body from {method} {url}: {response.text}",
                status_code=response.status_code,
                payload=body,
            )

        meta = body.get("meta")
        return body.get("data"), PaginationMeta(**meta) if meta else None

    async def list_gpu_instances(self) -> List[GpuInstance]:
        """List spec templates with GPU model, hourly price and remaining stock."""
        data, _ = await self._request("GET", "/gpu-instances")
        return [GpuInstance(**item) for item in data or []]

    async def list_images(self) -> List[Image]:
        """List available OS images. Use ``image_uuid`` as ``image_id`` on create."""
        data, _ = await self._request("GET", "/images")
        return [Image(**item) for item in data or []]

    async def create_vm(
        self,
        template_id: int,
        image_id: str,
        idempotency_key: str,
        user_data: Optional[str] = None,
        instance_name: Optional[str] = None,
    ) -> VMDetail:
        """Accept a VM for creation and start billing (first hour is pre-charged).

        Asynchronous: returns as soon as the lease is accepted (HTTP 202) with
        ``status=creating``. Poll :meth:`get_vm` until the status leaves
        :data:`PENDING_STATUSES` — ``active`` means ready, ``failed`` means the
        creation gave up.

        ``idempotency_key`` is required and guards against replays: the same
        key from the same user returns the lease from the first request instead
        of creating (and charging for) a second one. It must therefore be
        derived from something stable across retries. A creation that ended in
        ``failed`` can only be retried under a *new* key.

        Passing ``user_data`` switches the VM to self-contained mode: no SSH key
        is injected or generated, so ``ssh_private_key`` comes back empty and the
        public key must be embedded in the cloud-init script.
        """
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        if len(idempotency_key) > MAX_IDEMPOTENCY_KEY_SIZE:
            raise ValueError(
                f"idempotency_key exceeds the {MAX_IDEMPOTENCY_KEY_SIZE} "
                "characters limit"
            )
        body: Dict[str, Any] = {
            "template_id": template_id,
            # The image list has been observed to hand back uuids with
            # surrounding whitespace, which travels through the worker pool's
            # os_image untouched; strip it rather than posting an unknown image.
            "image_id": (image_id or "").strip(),
            "idempotency_key": idempotency_key,
        }
        if user_data is not None:
            if not user_data.startswith("#cloud-config"):
                raise ValueError("user_data must start with '#cloud-config'")
            if len(user_data.encode()) > MAX_USER_DATA_SIZE:
                raise ValueError(
                    f"user_data exceeds the {MAX_USER_DATA_SIZE} bytes limit"
                )
            body["user_data"] = user_data
        if instance_name:
            if len(instance_name) > MAX_INSTANCE_NAME_SIZE:
                raise ValueError(
                    f"instance_name exceeds the {MAX_INSTANCE_NAME_SIZE} "
                    "characters limit"
                )
            body["instance_name"] = instance_name

        logger.info(
            f"Creating Shuihua VM with template {template_id} and image {image_id}"
        )
        data, _ = await self._request("POST", "/vms", json=body)
        return VMDetail(**(data or {}))

    async def list_vms(
        self, page: int = 1, page_size: int = 20
    ) -> Tuple[List[VMSummary], PaginationMeta]:
        """List the VMs created by the current API key, one page at a time."""
        data, meta = await self._request(
            "GET", "/vms", params={"page": page, "page_size": page_size}
        )
        return [VMSummary(**item) for item in data or []], meta or PaginationMeta()

    async def iter_vms(self, page_size: int = 20) -> AsyncIterator[VMSummary]:
        """Iterate over every VM of the current API key, following pagination."""
        page = 1
        while True:
            items, meta = await self.list_vms(page=page, page_size=page_size)
            for item in items:
                yield item
            # total_pages is optional, so it cannot be the only stop condition:
            # coercing a missing one to 0 would end the walk after page 1. A
            # short page means the last one either way.
            total_pages = meta.total_pages
            if (
                not items
                or len(items) < page_size
                or (total_pages is not None and page >= total_pages)
            ):
                return
            page += 1

    async def get_vm(self, id: int) -> VMDetail:
        """Get a VM owned by the current API key.

        VMs created without ``user_data`` also return their SSH private key.
        """
        data, _ = await self._request("GET", f"/vms/{id}")
        return VMDetail(**(data or {}))

    async def start_vm(self, id: int) -> str:
        """Power on a VM. Returns the server message."""
        logger.info(f"Starting Shuihua VM {id}")
        return await self._vm_action(id, "start")

    async def stop_vm(self, id: int) -> str:
        """Power off a VM. Returns the server message."""
        logger.info(f"Stopping Shuihua VM {id}")
        return await self._vm_action(id, "stop")

    async def terminate_vm(self, id: int) -> str:
        """Destroy a VM, settle billing by the minute and end the lease."""
        logger.info(f"Terminating Shuihua VM {id}")
        return await self._vm_action(id, "terminate")

    async def _vm_action(self, id: int, action: str) -> str:
        data, _ = await self._request("POST", f"/vms/{id}/{action}")
        return (data or {}).get("message", "")


def get_endpoint() -> str:
    """Base URL of the Shuihua API, from the server's ``shuihua_api_base_url``.

    That setting carries a default, so this always resolves to a host.
    """
    return get_global_config().shuihua_api_base_url.rstrip("/")


class ShuihuaProviderClient(ProviderClientBase):
    """Adapts :class:`ShuihuaClient` to the cloud provider lifecycle.

    Two API-level differences from a full IaaS provider shape the mapping:

    * There is no SSH key API, so there is nothing to register and
      ``create_ssh_key`` returns None. The key reaches the instance through
      ``construct_user_data``, which writes it into the cloud-config as an
      authorized key — every instance runs in Shuihua's self-contained mode.
    * There is no block storage API, so volumes are rejected rather than
      silently dropped.

    ``CloudInstanceCreate.type`` carries the ``template_id`` (a numeric spec
    template) and ``image`` the ``image_uuid``; ``name`` becomes the instance
    name and ``labels`` anchor the idempotency key, while ``region`` has no
    counterpart in the API and is ignored.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self._api_key = api_key
        self._base_url = base_url or get_endpoint()

    @asynccontextmanager
    async def _api(self) -> AsyncGenerator[ShuihuaClient, None]:
        """One short-lived HTTP client per call.

        Provider clients are rebuilt on every reconcile, so holding an open
        connection pool on the instance would leak sockets.
        """
        client = ShuihuaClient(api_key=self._api_key, base_url=self._base_url)
        try:
            yield client
        finally:
            await client.aclose()

    async def create_instance(self, instance: CloudInstanceCreate) -> str:
        try:
            template_id = int(instance.type)
        except (TypeError, ValueError):
            raise ValueError(
                f"Shuihua instance type must be a numeric template_id, got {instance.type!r}"
            )
        if not instance.user_data:
            raise ValueError("Shuihua requires user_data to bootstrap the worker")
        async with self._api() as client:
            vm = await client.create_vm(
                template_id=template_id,
                image_id=instance.image,
                idempotency_key=_idempotency_key(instance),
                user_data=instance.user_data,
                instance_name=(instance.name or "")[:MAX_INSTANCE_NAME_SIZE] or None,
            )
        if vm.id is None:
            raise RuntimeError(
                f"Shuihua returned no VM id when creating instance {instance.name}"
            )
        if status_mapping.get(vm.status) in TERMINAL_STATES:
            # Not a fresh lease: replaying the key of one that already ended
            # hands that same dead lease back, and only a new key can start
            # over, so there is nothing here to wait for.
            raise InstanceProvisioningFailed(
                f"Shuihua answered the create for {instance.name} with VM "
                f"{vm.id} already {vm.status.value if vm.status else None}; "
                "recreate the worker to retry under a new idempotency key"
            )
        # Creation is asynchronous, so the lease usually comes back as
        # 'creating' here; wait_for_started polls it to completion. An
        # unrecognised status also falls through to that poll rather than
        # failing outright.
        return str(vm.id)

    async def delete_instance(self, external_id: str):
        vm_id = _parse_vm_id(external_id)
        if vm_id is None:
            # Nothing to terminate, and raising would block the worker's
            # deletion on an id that can never resolve to a VM.
            return
        async with self._api() as client:
            try:
                await client.terminate_vm(vm_id)
            except ShuihuaAPIError as e:
                # Only 404. Terminating an already-terminated lease answers 200
                # ("terminated"), so 400 never means "already gone" — it is any
                # bad request, plausibly "cannot terminate while creating", and
                # swallowing it would delete the worker row while a live,
                # billing VM stays up with nothing pointing at it.
                if e.status_code == 404:
                    logger.info(
                        f"Shuihua VM {external_id} no longer exists, skipping delete: {e}"
                    )
                    return
                raise

    async def get_instance(self, external_id: str) -> Optional[CloudInstance]:
        vm_id = _parse_vm_id(external_id)
        if vm_id is None:
            return None
        async with self._api() as client:
            try:
                vm = await client.get_vm(vm_id)
            except ShuihuaAPIError as e:
                if e.status_code == 404:
                    return None
                raise
        return _to_cloud_instance(vm)

    async def wait_for_started(
        self, external_id: str, backoff: int = 15, limit: int = 20
    ) -> CloudInstance:
        for _ in range(limit):
            instance = await self.get_instance(external_id)
            if instance and instance.status == InstanceState.RUNNING:
                return instance
            _raise_if_terminal(external_id, instance)
            await asyncio.sleep(backoff)
        raise TimeoutError(
            f"Shuihua VM {external_id} did not start within {limit} retries"
        )

    async def wait_for_public_ip(
        self, external_id: str, backoff: int = 15, limit: int = 20
    ) -> CloudInstance:
        for _ in range(limit):
            instance = await self.get_instance(external_id)
            if instance and instance.ip_address:
                return instance
            _raise_if_terminal(external_id, instance)
            await asyncio.sleep(backoff)
        raise TimeoutError(
            f"Shuihua VM {external_id} did not acquire an IP within {limit} retries"
        )

    async def create_ssh_key(self, worker_name: str, public_key: str) -> Optional[str]:
        """No key API upstream, so there is nothing to register.

        The key reaches the instance through :meth:`construct_user_data`
        instead, so no external id exists and the caller leaves
        ``Credential.external_id`` NULL.
        """
        return None

    async def delete_ssh_key(self, id: str):
        """Unreachable: with no external id recorded, teardown skips this."""
        return

    async def create_volumes_and_attach(
        self, worker_id: int, external_id: str, region: str, *volumes: Volume
    ) -> List[str]:
        if volumes:
            raise NotImplementedError(
                "Shuihua has no block storage API; remove the worker pool's volumes"
            )
        return []

    async def construct_user_data(
        self,
        server_url,
        token,
        image_name,
        os_image,
        worker_name,
        secret_configs: Dict[str, Any] = {},
        ssh_public_key: Optional[str] = None,
    ) -> UserDataTemplate:
        user_data = await super().construct_user_data(
            server_url,
            token,
            image_name,
            os_image,
            worker_name,
            secret_configs,
            ssh_public_key,
        )
        # Shuihua only offers GPU images that already carry both the NVIDIA
        # driver and the container toolkit (checked on a live instance: nvidia-ctk
        # is on PATH), so setup only has to point Docker at the runtime -- there
        # is nothing to install. Were the toolkit ever missing, this would fail
        # quietly: cloud-init keeps going past a failed runcmd, so the worker
        # would register normally and simply report no GPUs.
        user_data.distribution = "ubuntu"
        user_data.setup_driver = ManufacturerEnum.NVIDIA
        user_data.install_driver = None
        user_data.insert_runcmd("mkdir -p /var/lib/gpustack")
        # Passing user_data puts the VM in self-contained mode, where Shuihua
        # injects no key of its own, so the worker's key has to ride along here.
        if ssh_public_key:
            user_data.add_ssh_authorized_keys(ssh_public_key)
        return user_data

    @classmethod
    def get_api_endpoint(cls) -> str:
        return get_endpoint()

    @classmethod
    def process_header(cls, ak: str, sk: str, options: dict, headers: dict) -> dict:
        headers["Authorization"] = f"Bearer {sk}"
        return headers


def _parse_vm_id(external_id: str) -> Optional[int]:
    """The VM id an external_id names, or None when it names none.

    Every id this client hands out is a stringified integer, so a value that
    isn't one cannot match a VM. Callers treat None as "no such instance"
    rather than letting a ValueError escape into the provisioning controller,
    where it would strand the worker over an id that can never resolve.
    """
    try:
        return int(external_id)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring malformed Shuihua VM id {external_id!r}")
        return None


def _idempotency_key(instance: CloudInstanceCreate) -> str:
    """The replay-guard token to send, which Shuihua requires on every create.

    Normally supplied by ``construct_cloud_instance``, which anchors it to the
    worker row so that it stays put across retries of one attempt and turns
    over for the next. The fallback only covers a hand-built
    ``CloudInstanceCreate``: it is stable, but has no notion of attempts, so
    the same worker can never get a second VM under it.
    """
    if instance.idempotency_key:
        return instance.idempotency_key
    labels = instance.labels or {}
    identity = "/".join(str(labels.get(key, "")) for key in ("cluster_id", "worker_id"))
    digest = hashlib.sha256(f"{identity}/{instance.name}".encode()).hexdigest()
    return f"gpustack-{digest[:32]}"


# Statuses a lease can no longer leave for 'active'. EXPIRED is left out: it is
# only powered off, and the API does offer a start endpoint, so it is not
# provably dead.
TERMINAL_STATES = (InstanceState.FAILED, InstanceState.TERMINATED)


def _raise_if_terminal(external_id: str, instance: Optional[CloudInstance]) -> None:
    """Give up on a lease that can no longer start, instead of polling it.

    Replaying the key of a lease that already ended hands back that same dead
    lease rather than creating a VM, so without this the wait spins to its
    timeout and the next reconcile starts the same wait over again.
    """
    if instance and instance.status in TERMINAL_STATES:
        raise InstanceProvisioningFailed(
            f"Shuihua VM {external_id} is {instance.status.value} and will not "
            "start; recreate the worker to retry under a new idempotency key"
        )


def _to_cloud_instance(vm: VMDetail) -> CloudInstance:
    # The instance's own address, deliberately not the public IP of its port
    # mapping: one public IP fronts every instance in the account, differing
    # only by port, so it identifies nothing and cannot serve as an advertise
    # address. Only ports 22 and 80 are mapped anyway — never the worker's own
    # port — so these workers are reached through the tunnel rather than
    # inbound, and the private address is what identifies them. Callers that
    # do need to connect (SSH) get the mapped endpoint below instead.
    ip_address = vm.ip_address
    return CloudInstance(
        external_id=str(vm.id),
        name=vm.instance_name or "",
        image="",
        type=str(vm.template_id) if vm.template_id is not None else "",
        region="",
        # SSH is only reachable through the mapping, on the shared public IP at
        # an instance-specific port — port 22 of ip_address answers nothing.
        ssh_endpoint=vm.public_endpoint(DEFAULT_SSH_PORT),
        ssh_user=vm.ssh_user or None,
        ssh_key_id=None,
        volume_ids=[],
        user_data=None,
        status=status_mapping.get(vm.status, InstanceState.UNKNOWN),
        ip_address=ip_address,
    )
