# Docker Installation

You can use the official Docker image to run GPUStack in a container. Installation using docker is supported on:

- Linux with Nvidia GPUs

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Run GPUStack with Docker

Run the following command to start the GPUStack server:

```shell
docker run -d --gpus all -p 80:80 gpustack/gpustack
```

You can set additional flags for the `gpustack start` command by appending them to the docker run command. For example, to start a GPUStack worker:

```shell
docker run -d --gpus all -p 80:80 gpustack/gpustack start --server-url http://myserver --token mytoken
```

For more configurations, please refer to the [CLI Reference](../cli-reference/start.md).

## Run GPUStack with Docker Compose

Get the docker-compose file from GPUStack repository, run the following command to start the GPUStack server:

```shell
docker-compose up -d
```

You can update the `docker-compose.yml` file to customize the command while starting a GPUStack worker.
