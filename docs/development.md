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

## Update Dependencies

```bash
uv add <something>
```

Or

```bash
uv add --dev <something>
```

For dev/testing dependencies.
