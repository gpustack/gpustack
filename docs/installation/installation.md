# Installation

## Prerequisites

**GPUStack server:**

- [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (Windows and macOS) is also supported.

**GPUStack workers:**

- [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop is **not** supported.
- Only Linux is supported for GPUStack worker nodes. If you use Windows, consider using WSL2 and avoid using Docker Desktop. macOS is not supported for GPUStack worker nodes.
- Ensure the appropriate GPU drivers and container toolkits are installed for your hardware. See the [Installation Requirements](./requirements.md) for details.

## Install GPUStack Server

Run the following command to install and start the GPUStack server using Docker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

## Startup

Check the GPUStack container logs:

```bash
sudo docker logs -f gpustack
```

If everything is normal, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## Add GPU Clusters and Worker Nodes

Please follow the UI instructions on the `Clusters` and `Workers` pages to add GPU clusters and worker nodes.

## Custom Configuration

The following sections describe examples of custom configuration options when starting the GPUStack server container. For a full list of available options, refer to the [CLI Reference](../cli-reference/start.md).

### Enable HTTPS with Custom Certificate


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

### Using an External Database

By default, GPUStack uses an embedded PostgreSQL database. To use an external database such as PostgreSQL or MySQL, set the `GPUSTACK_DATABASE_URL` environment variable or use the `--database-url` argument when starting the GPUStack container:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### Configure External Server URL

If you use a cloud provider to provision workers, set the external server URL for worker registration to ensure that workers can connect to the server correctly.

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

## Installation via Docker Compose

This guide explains how to deploy GPUStack and observability components (Prometheus, Grafana) using Docker Compose.


### Overview of Services

**Services:**

- **gpustack-server**: Central server for scheduling, management, and built-in inference.
- **prometheus**: Metrics collection.
- **grafana**: Metrics visualization.

### Prerequisites

- [Docker Compose](https://docs.docker.com/compose/install/) must be installed.
- [Required ports](./requirements.md#port-requirements) must be available.

### Deployment

Get the latest Docker Compose file from the [GPUStack GitHub repository](https://github.com/gpustack/gpustack/blob/v2.0.1/docker-compose/docker-compose.server.yaml).

Run the following command to start the GPUStack server and observability components:

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

If everything is normal, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```
