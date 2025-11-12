import base64
import yaml
from typing import Dict, Tuple, Type, Callable, Optional
from .abstract import ProviderClientBase, CloudInstanceCreate
from .digital_ocean import DigitalOceanClient
from gpustack.schemas.clusters import ClusterProvider, CloudCredential, Credential
from gpustack.schemas.workers import Worker
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519
from gpustack.config.config import Config
from gpustack.cloud_providers.user_data import user_data_distribution


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
    labels = worker.labels or {}
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


def construct_docker_run_script(image_name: str, token: str, server_url: str) -> str:
    return f"""#!/bin/bash
set -e
echo "$(date): trying to bring up gpustack worker container..." >> /var/log/post-reboot.log

docker run -d --name gpustack-worker \
--restart=unless-stopped \
--privileged --net=host \
-v /var/lib/gpustack:/var/lib/gpustack \
-v /var/run/docker.sock:/var/run/docker.sock \
{image_name} --server-url {server_url} --token {token}

echo "$(date): gpustack worker container started" >> /var/log/post-reboot.log
"""


def construct_user_data(
    config: Config, worker: Worker, distribution: Optional[str], public: bool
) -> str:
    """
    Construct the cloud_init data for the worker.
    """
    provider = worker.cluster.provider
    type_factory = factory.get(provider, None)
    if type_factory is None:
        raise ValueError(f"Unsupported provider: {provider}")
    t = type_factory[0]
    image_name = config.get_image_name()
    server_url = config.server_external_url
    if server_url is None:
        raise ValueError("server_external_url is not set in the config")
    script = construct_docker_run_script(
        image_name=image_name,
        token=worker.cluster.registration_token,
        server_url=server_url,
    )
    user_data = user_data_distribution(public=public, distribution=distribution)
    to_write_files: list[dict] = user_data.setdefault('write_files', [])
    to_write_files.insert(
        0,
        {
            "content": script,
            "path": "/opt/gpustack-run-worker.sh",
            "permissions": "0755",
        },
    )
    user_data['write_files'] = to_write_files
    t.modify_cloud_init(user_data)
    data = yaml.safe_dump(user_data)
    data = "#cloud-config\n" + data
    return data


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
