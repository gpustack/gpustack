import os
import shutil
import signal
import socket
import subprocess
import time


def move_generated_client():
    generated_root_path = "gpustack-client"
    generated_module_path = f"{generated_root_path}/generated_client"
    target_path = "gpustack/generated_client"

    if not os.path.exists(generated_module_path):
        return

    if os.path.exists(target_path):
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)

    shutil.move(generated_module_path, target_path)
    shutil.rmtree(generated_root_path)


def generated_client_code():
    subprocess.run(
        [
            "openapi-python-client",
            "generate",
            "--url",
            "http://localhost/openapi.json",
            "--config",
            "gpustack/codegen/config.yaml",
        ],
        check=True,
    )


def start_server() -> subprocess.Popen | None:
    host = "localhost"
    port = 80
    port_in_use = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        port_in_use = result == 0

    if port_in_use:
        return None

    server_process = subprocess.Popen(["gpustack", "server", "--disable-agent"])

    time.sleep(3)

    return server_process


def stop_server(server_process: subprocess.Popen | None):
    if server_process:
        server_process.send_signal(signal.SIGTERM)
        server_process.wait()


def main():
    server_process = start_server()

    try:
        generated_client_code()
        move_generated_client()
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
