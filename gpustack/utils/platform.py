import platform
import os
import logging
from typing import Optional
from kubernetes import client
from kubernetes.client import Configuration
from kubernetes.config.kube_config import KubeConfigLoader, KubeConfigMerger
from kubernetes.client.exceptions import ApiException
from kubernetes.config.incluster_config import (
    InClusterConfigLoader,
    SERVICE_TOKEN_FILENAME,
    SERVICE_CERT_FILENAME,
)

logger = logging.getLogger(__name__)


def system() -> str:
    """
    Get the current operating system name in lowercase.
    """
    return platform.uname().system.lower()


def native_arch() -> str:
    """
    Get the native architecture of the machine in lowercase.
    """
    return platform.machine().lower()


_ARCH_ALIAS_MAPPING = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "i386": "386",
    "i686": "386",
    "arm64": "arm64",
    "aarch64": "arm64",
    "armv7l": "arm",
}


def arch() -> str:
    """
    Get the architecture of the machine in a standardized format.
    If the architecture is not recognized, return the native architecture.
    """
    na = native_arch()
    return _ARCH_ALIAS_MAPPING.get(native_arch(), na)


def system_arch() -> str:
    """
    Get the system and architecture in the format "system/arch".
    """
    return f"{system()}/{arch()}"


def is_inside_kubernetes() -> bool:
    """
    Check if the code is running inside a Kubernetes cluster.
    This is determined by the presence of specific environment variables and files.
    """
    return os.getenv("KUBERNETES_SERVICE_HOST") is not None and os.path.exists(
        "/var/run/secrets/kubernetes.io/serviceaccount/token"
    )


def is_supported_higress(kubeconfig: Optional[str] = None) -> bool:
    configuration = Configuration()
    if kubeconfig and os.path.exists(kubeconfig):
        cfg_loader = KubeConfigLoader(config_dict=KubeConfigMerger(kubeconfig).config)
        if not cfg_loader._load_user_token():
            cfg_loader._load_user_pass_token()
        cfg_loader._load_cluster_info()
        cfg_loader._set_config(configuration)
    elif is_inside_kubernetes():
        cfg_loader = InClusterConfigLoader(
            token_filename=SERVICE_TOKEN_FILENAME,
            cert_filename=SERVICE_CERT_FILENAME,
        )
        cfg_loader.load_and_set(configuration)
    else:
        return False

    try:
        api_client = client.ApiClient(configuration=configuration)
        networking_client = client.NetworkingV1Api(api_client=api_client)
        networking_client.read_ingress_class(name="higress")
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        logger.debug(f"Error checking for Higress IngressClass: {e}")
        return False
