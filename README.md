# GPUStack

Turn your set of GPUs into a full-fledged LLM stack with a snap.

## Development

### Prerequisites

Install `python@3.11` and `poetry`.

### Set Up Environment

```
make install
```

### Run

```
poetry run gpustack
```

### Build

```
make build
```

And check artifacts in `dist`.

### Test

```
make test
```

### Update Dependencies

```
poetry add <something>
```

Or

```
poetry add --dev <something>
```

For dev/testing dependencies.
