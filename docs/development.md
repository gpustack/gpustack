# Development Guide

## Prerequisites

1. Install Python (version 3.10 to 3.12).
2. Run a database (PostgreSQL or MySQL).

For example, to run a PostgreSQL database using Docker:
```bash
docker run --name my-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d --restart=always postgres
```

## Set Up Environment

```bash
make install
```

## Run

Use `disabled` gateway mode and connect to your database for development:

```bash
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

## Build

```bash
make build
```

And check artifacts in `dist`.

## Test

```bash
make test
```

## Package

Building the container image requires Docker with the [Buildx](https://docs.docker.com/build/install-buildx/) plugin:

```bash
make package
```

If `docker buildx version` fails, install the plugin with the package manager (`apt-get install docker-buildx-plugin` or `yum install docker-buildx-plugin`), or drop the release binary in place:

```bash
mkdir -p /usr/local/lib/docker/cli-plugins
VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/download/${VER}/buildx-${VER}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
docker buildx version
```

The unauthenticated GitHub API allows 60 requests per hour per IP. If it rate limits you, `VER` ends up empty and the download 404s — pick a version from the [releases page](https://github.com/docker/buildx/releases) and set it by hand.

Install the plugin under `/usr/local/lib/docker/cli-plugins` rather than `~/.docker/cli-plugins` when packaging with `sudo`, otherwise the changed `HOME` hides it from the Docker CLI.

## Update Dependencies

```bash
uv add <something>
```

Or

```bash
uv add --dev <something>
```

For dev/testing dependencies.
