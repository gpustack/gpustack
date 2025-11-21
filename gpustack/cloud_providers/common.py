import base64
from typing import Dict, Tuple, Type, Callable
from .abstract import ProviderClientBase, CloudInstanceCreate
from .digital_ocean import DigitalOceanClient
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
