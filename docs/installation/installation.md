# Installation via Docker

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

!!! note

    GPUStack v2 uses a single unified container image for all GPU device types.

## Startup

Check the GPUStack container logs:

```bash
sudo docker logs -f gpustack
```

Once the server is up, open `http://your_host_ip` in a browser to access the GPUStack UI.

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

`GPUSTACK_SSL_CERTFILE` should contain the server certificate followed by any
intermediate certificates. When the server certificate is signed by a private
CA, also add its PEM CA bundle so worker and benchmark containers can verify
the server automatically:

```diff
 sudo docker run -d --name gpustack \
     ...
+    -e GPUSTACK_SSL_CA_CERTFILE=/path/to/cert_files/ca-bundle.crt \
     gpustack/gpustack
     ...
```

For a self-signed server certificate, the certificate file itself is used when
no separate CA bundle is configured. Certificates issued by public CAs require
no additional CA configuration.

The registration command includes a SHA-256 checksum of the bootstrap CA bundle.
The worker checks this checksum before installing a bundle downloaded without TLS
verification. This is not certificate pinning: if the server already passes normal
TLS verification, the worker uses the existing trust store without downloading or
checking the bootstrap bundle.

The worker first tries to install the verified CA into the system trust store.
If that store cannot be updated or the worker's TLS client uses a different
bundle, the worker uses a temporary merged bundle via `SSL_CERT_FILE` for its
process and child processes. Existing operator-provided certificates are retained;
replacing an injected bootstrap CA does not retain earlier injected bundles.

As a last resort, when the server certificate cannot be verified on a worker at
all, set `GPUSTACK_INSECURE_TLS=true` on that worker to skip certificate
verification on its connection to the server. Traffic stays encrypted but is no
longer protected against interception, so use it only on trusted networks. See
[Environment Variables](../environment-variables.md).

### Using an External Database

By default, GPUStack uses an embedded PostgreSQL database. To use an external database such as PostgreSQL or MySQL, set the `GPUSTACK_DATABASE_URL` environment variable or use the `--database-url` argument when starting the GPUStack container. See [Database Requirements](requirements.md#database-requirements) for the list of compatible databases and verified versions.

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

### Additional Trusted CAs

If GPUStack needs to communicate with services that use certificates issued by a private or corporate CA (e.g., a self-hosted Identity Provider, a Hugging Face mirror, or an internal API endpoint), mount the CA certificate into the container under `/usr/local/share/ca-certificates/`. GPUStack will automatically import the mounted CA certificates during startup and add them to the system trust store.

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
     gpustack/gpustack
     ...
```

!!! note

    The CA certificate must be PEM-encoded with a `.crt` extension. You can mount multiple CA certificates by adding additional `--volume` flags.

## Installation via Docker Compose

### Prerequisites

- [Docker Compose](https://docs.docker.com/compose/install/) must be installed.
- [Required ports](./requirements.md#port-requirements) must be available.

### Deployment

The Docker Compose files and configuration files are maintained in the [GPUStack repository](https://github.com/gpustack/gpustack/tree/main/docker-compose).

Run the following commands to clone the latest stable release:

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

Start the GPUStack server:

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

Once the server is up, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

For built-in and external observability options, see [Observability](../user-guide/observability.md).
