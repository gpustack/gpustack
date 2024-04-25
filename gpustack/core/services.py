import os
import subprocess
import time

import redis
from ..logging import logger
from .config import configs

redis_process = None


def at_exit():
    global redis_process
    if redis_process:
        logger.debug(f"Stopping Redis server with PID {redis_process.pid}")

        redis_process.terminate()
        redis_process.wait()

        logger.debug("Redis server stopped.")


def start_redis_server(
    executable: str,
    port: int = 6379,
    password: str | None = None,
    stdout_file: str | None = None,
    stderr_file: str | None = None,
):
    """Start a single Redis server.

    Args:
        executable (str): Full path of the redis-server executable.
        port (int): Try to start a Redis server at this port.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        password (str): Prevents external clients without the password
            from connecting to Redis if provided.

    Raises:
        Exception: An exception is raised if Redis could not be started.
    """
    if check_redis_running(port):
        logger.info("Redis is already running on port {}".format(port))
        return

    command = [executable]
    if port:
        command.extend(["--port", str(port)])
    if password:
        command.extend(["--requirepass", password])

    # Enable AOF persistence
    command.extend(["--appendonly", "yes"])
    # Set dir
    redis_dir = f"{configs.data_dir}/redis"
    if not os.path.exists(redis_dir):
        os.makedirs(redis_dir)
    command.extend(["--dir", redis_dir])

    # Prepare the subprocess.Popen arguments for stdout and stderr
    stdout_arg = subprocess.PIPE if stdout_file is None else open(stdout_file, "w")
    stderr_arg = subprocess.PIPE if stderr_file is None else open(stderr_file, "w")

    try:
        # Use subprocess.Popen to start the Redis server
        global redis_process
        redis_process = subprocess.Popen(
            command,
            stdout=stdout_arg,  # Redirect stdout
            stderr=stderr_arg,  # Redirect stderr
        )
    except Exception as e:
        raise Exception(f"Failed to start Redis server: {e}")

    try:
        wait_for_redis_to_start(port, password)
    except (redis.exceptions.ResponseError, RuntimeError):
        raise RuntimeError(
            "Couldn't start Redis. "
            "Check log files: {} {}".format(
                stdout_file.name if stdout_file is not None else "<stdout>",
                stderr_file.name if stdout_file is not None else "<stderr>",
            )
        )


def check_redis_running(port: int):
    """Check if Redis is already running on the given port.

    Args:
        port (int): The port of the redis server.

    Returns:
        bool: True if a Redis server is running on the given port, False otherwise.
    """
    try:
        redis_client = redis.StrictRedis(host="127.0.0.1", port=port)
        redis_client.ping()
        return True
    except (redis.ConnectionError, redis.ResponseError):
        return False


def wait_for_redis_to_start(port: int, password: str | None):
    """Wait for a Redis server to be available.

    This is accomplished by creating a Redis client and sending a random
    command to the server until the command gets through.

    Args:
        port (int): The port of the redis server.
        password (str): The password of the redis server.

    Raises:
        Exception: An exception is raised if we could not connect with Redis.
    """
    redis_client = redis.StrictRedis(host="127.0.0.1", port=port, password=password)
    # Wait for the Redis server to start.
    num_retries = 6
    delay = 0.1
    for i in range(num_retries):
        try:
            redis_client.client_list()
        except redis.AuthenticationError as authEx:
            raise RuntimeError(
                "Unable to connect to Redis at :{}.".format(port)
            ) from authEx
        except redis.ConnectionError as connEx:
            if i >= num_retries - 1:
                raise RuntimeError(
                    f"Unable to connect to Redis at :{port} after {num_retries} retries."
                ) from connEx
            # Wait a little bit.
            time.sleep(delay)
            delay *= 2
        else:
            logger.info("Started Redis server.")
            break
    else:
        raise RuntimeError(
            f"Unable to connect to Redis at :{port} after {num_retries} retries."
        )


if __name__ == "__main__":
    start_redis_server(
        executable="redis-server",
    )
