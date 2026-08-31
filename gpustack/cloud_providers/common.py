import base64
import hashlib
from datetime import timezone
from typing import Dict, Tuple, Type, Callable
from .abstract import ProviderClientBase, CloudInstanceCreate
from .digital_ocean import DigitalOceanClient
from .shuihua import ShuihuaProviderClient
from gpustack.schemas.clusters import ClusterProvider, CloudCredential, Credential
from gpustack.schemas.workers import Worker
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519


factory: Dict[
    ClusterProvider,
    Tuple[Type[ProviderClientBase], Callable[[CloudCredential], ProviderClientBase]],
] = {
    ClusterProvider.DigitalOcean: (
        DigitalOceanClient,
        lambda credential: DigitalOceanClient(token=credential.secret),
    ),
    ClusterProvider.Shuihua: (
        ShuihuaProviderClient,
        lambda credential: ShuihuaProviderClient(api_key=credential.secret),
    ),
}


def get_client_from_provider(
    provider: ClusterProvider,
    credential: CloudCredential,
) -> ProviderClientBase:
    type_factory = factory.get(provider, None)
    if type_factory is None:
        raise ValueError(f"Unsupported provider: {provider}")
    f = type_factory[1]
    return f(credential)


def creation_idempotency_key(worker: Worker) -> str:
    """
    A token identifying one attempt at creating this worker's instance.

    Providers offering replay protection (Shuihua requires a key) send this, so
    that retrying a create whose outcome was never learned returns the first
    instance instead of provisioning — and charging for — a second one. The
    live case is the server being killed mid-provision: nothing records the
    failure, so the worker is still PROVISIONING and gets picked up again on
    restart.

    ``updated_at`` supplies the stability. The provisioning state machine
    commits once per step, so the value read here was persisted by the
    *previous* step and survives this one rolling back, while any later write
    that re-drives the worker (resetting it out of ERROR) necessarily bumps it
    and so mints a new token. This holds only because the controller re-reads
    the worker from the database on every reconcile — caching it across steps
    would break it.

    Whole seconds: MySQL stores TIMESTAMP without fractional digits while
    PostgreSQL keeps microseconds, so the key stays the same either way.

    The limit of anchoring on ``updated_at`` is that only failures leaving no
    trace are covered. Any failure the reconcile loop records -- it writes
    ``state=ERROR`` on every exception, including a timeout on a create the
    provider had already accepted and charged for -- bumps the column, so a
    fresh attempt would mint a new key and pay for a second instance. That is
    survivable today because nothing re-drives an ERROR worker: recovery means
    deleting it, and the replacement is a different row that *should* get its
    own key. Adding an in-place retry would make it real, and would mean
    persisting the key instead (in ``worker.provider_config``, committed in the
    state-machine step *before* the create, or the crash window reopens).
    """
    updated_at = worker.updated_at
    if updated_at is not None and updated_at.tzinfo is None:
        # The column holds naive UTC; reading it as local time would shift it.
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    attempt = int(updated_at.timestamp()) if updated_at is not None else 0
    identity = f"{worker.cluster_id}/{worker.id}/{worker.name}/{attempt}"
    return f"gpustack-{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


def construct_cloud_instance(
    worker: Worker, ssh_key: Credential, user_data: str
) -> CloudInstanceCreate:
    """
    Assuming the cloud instance is not created
    """
    cluster = worker.cluster
    pool = worker.worker_pool
    labels = dict(worker.labels or {})
    labels.pop("provider", None)
    labels.pop("instance_type", None)
    return CloudInstanceCreate(
        name=worker.name,
        image=pool.os_image,
        type=pool.instance_type,
        region=cluster.region,
        ssh_key_id=ssh_key.external_id,
        user_data=user_data,
        labels={
            "cluster_id": cluster.id,
            "worker_id": worker.id,
            **labels,
        },
        idempotency_key=creation_idempotency_key(worker),
    )


def generate_ssh_key_pair(
    algorithm: str = "ED25519", key_size: int = 2048
) -> Tuple[str, str]:
    """
    algorithm: RSA or ED25519
    returns private_key in base64 encoded, public_key in pem format
    """
    if algorithm.upper() == "RSA":
        key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        key_bytes = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key = (
            key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
            .decode()
        )
    elif algorithm.upper() == "ED25519":
        key = ed25519.Ed25519PrivateKey.generate()
        key_bytes = key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key = (
            key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
            .decode()
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    private_key_b64 = base64.b64encode(key_bytes).decode()
    return private_key_b64, public_key


def key_bytes_to_openssh_pem(key_bytes: bytes, algorithm: str):
    if algorithm.upper() == "RSA":
        return key_bytes
    elif algorithm.upper() == "ED25519":
        key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
    else:
        raise ValueError("Unsupported algorithm")
    return pem
