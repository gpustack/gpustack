---
hide:
  - toc
---

# gpustack list-images

List images.

```bash
gpustack list-images [OPTIONS]

```

## Configurations

| <div style="width:220px">Flag</div>                        | <div style="width:120px">Default</div> | Description                                                                        |
|------------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------|
| `--backend` `{cann,corex,cuda,dtk,maca,musa,neuware,rocm}` | (empty)                                | Filter gpustack/runner images by backend name                                      |
| `--backend-version` `BACKEND_VERSION`                      | (empty)                                | Filter gpustack/runner images by exact backend version                             |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`        | (empty)                                | Filter gpustack/runner images by backend version prefix                            |
| `--backend-variant` `BACKEND_VARIANT`                      | (empty)                                | Filter gpustack/runner images by backend variant                                   |
| `--service` `{voxbox,vllm,mindie,sglang}`                  | (empty)                                | Filter gpustack/runner images by service name                                      |
| `--service-version` `SERVICE_VERSION`                      | (empty)                                | Filter gpustack/runner images by exact service version                             |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`        | (empty)                                | Filter gpustack/runner images by service version prefix                            |
| `--repository` `REPOSITORY`                                | (empty)                                | Filter images by repository name                                                   |
| `--platform` `{linux/amd64,linux/arm64}`                   | (empty)                                | Filter images by platform                                                          |
| `  --format {text,json}`                                   | `text`                                 | Output format. `text` for human-readable text format, `json` for JSON array format |
