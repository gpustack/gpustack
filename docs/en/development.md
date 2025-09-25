# Development Guide

## Prerequisites

Install Python (version 3.10 to 3.12).

## Set Up Environment

```bash
make install
```

## Run

```bash
poetry run gpustack
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
poetry add <something>
```

Or

```bash
poetry add --group dev <something>
```

For dev/testing dependencies.
