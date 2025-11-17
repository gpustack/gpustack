# Development Guide

## Prerequisites

Install Python (version 3.10 to 3.12).

## Set Up Environment

```bash
make install
```

## Run

```bash
uv run gpustack
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
