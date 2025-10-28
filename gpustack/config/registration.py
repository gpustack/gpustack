import os
import time
from typing import Optional
from gpustack.client import ClientSet
from gpustack.client.worker_manager_clients import (
    WorkerRegistrationClient,
)
from gpustack.security import API_KEY_PREFIX
from gpustack.utils.uuid import get_legacy_uuid, get_system_uuid

registration_token_filename = "token"
worker_token_filename = "worker_token"


def read_token(data_dir: str, filename) -> Optional[str]:
    token_path = os.path.join(data_dir, filename)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    return None


def write_token(data_dir: str, filename: str, token: str):
    token_path = os.path.join(data_dir, filename)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            existing_token = f.read().strip()
        if existing_token == token:
            return  # Token is already written
    with open(token_path, "w") as f:
        f.write(token + "\n")


def read_worker_token(data_dir: str) -> Optional[str]:
    return read_token(data_dir, worker_token_filename)


def write_worker_token(data_dir: str, token: str):
    write_token(data_dir, worker_token_filename, token)


def read_registration_token(data_dir: str) -> Optional[str]:
    return read_token(data_dir, registration_token_filename)


def write_registration_token(data_dir: str, token: str):
    write_token(data_dir, registration_token_filename, token)


def registration_client(
    data_dir: str,
    server_url: str,
    registration_token: Optional[str] = None,
    wait_token_file: bool = False,
) -> Optional[WorkerRegistrationClient]:
    # if token exists, skip registration
    worker_token = read_worker_token(data_dir)
    if worker_token is not None:
        return None
    if registration_token is None and wait_token_file:
        timeout = 10
        start_time = time.time()
        while True:
            registration_token = read_registration_token(data_dir)
            if registration_token is not None:
                break
            if time.time() - start_time > timeout:
                raise FileNotFoundError("Registration token file not found")
            time.sleep(0.5)
    if registration_token:
        if not registration_token.startswith(API_KEY_PREFIX):
            legacy_uuid = get_legacy_uuid(data_dir) or get_system_uuid(data_dir, False)
            if not legacy_uuid:
                raise ValueError(
                    "Legacy UUID not found, please re-register the worker."
                )
            registration_token = f"{API_KEY_PREFIX}_{legacy_uuid}_{registration_token}"
        clientset = ClientSet(
            base_url=server_url,
            api_key=registration_token,
        )
        return WorkerRegistrationClient(clientset.http_client)
    return None
