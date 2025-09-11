import os
import time
from typing import Optional
from gpustack.config.config import Config
from gpustack.client import ClientSet
from gpustack.client.worker_manager_clients import (
    WorkerRegistrationClient,
)

registration_token_filename = "registration_token"


def write_registration_token(data_dir: str, token: str):
    token_path = os.path.join(data_dir, registration_token_filename)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            existing_token = f.read().strip()
        if existing_token == token:
            return  # Token is already written
    with open(token_path, "w") as f:
        f.write(token)


def registration_client(
    cfg: Config, wait_token_file: bool = False
) -> Optional[WorkerRegistrationClient]:
    # if token exists, skip registration
    if cfg.token:
        return
    if not cfg.server_url:
        raise ValueError("Server URL is not set in the configuration.")
    token_path = os.path.join(cfg.data_dir, registration_token_filename)
    if wait_token_file:
        timeout = 10
        start_time = time.time()
        while not os.path.exists(token_path):
            if time.time() - start_time > timeout:
                raise FileNotFoundError(
                    f"Registration token file not found: {token_path}"
                )
            time.sleep(0.5)
    registration_token = cfg.registration_token
    if not registration_token and os.path.exists(token_path):
        with open(token_path, "r") as f:
            registration_token = f.read().strip()
    if registration_token:
        clientset = ClientSet(
            base_url=cfg.server_url,
            api_key=registration_token,
        )
        return WorkerRegistrationClient(clientset.http_client)
    return None
