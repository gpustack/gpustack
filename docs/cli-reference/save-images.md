---
hide:
  - toc
---

# gpustack save-images

Save images as OCI/Docker Archive to local path, powered by https://github.com/containers/skopeo.

```bash
gpustack save-images [OPTIONS] [output]

```

## Configurations

| <div style="width:220px">Flag</div>                             | <div style="width:120px">Default</div> | Description                                                             |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (empty)                                | Filter gpustack/runner images by backend name                           |
| `--backend-version` `BACKEND_VERSION`                           | (empty)                                | Filter gpustack/runner images by exact backend version                  |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (empty)                                | Filter gpustack/runner images by backend version prefix                 |
| `--backend-variant` `BACKEND_VARIANT`                           | (empty)                                | Filter gpustack/runner images by backend variant                        |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (empty)                                | Filter gpustack/runner images by service name                           |
| `--service-version` `SERVICE_VERSION`                           | (empty)                                | Filter gpustack/runner images by exact service version                  |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (empty)                                | Filter gpustack/runner images by service version prefix                 |
| `--repository` `REPOSITORY`                                     | (empty)                                | Filter images by repository name                                        |
| `--platform` `{linux/amd64,linux/arm64}`                        | (empty)                                | Filter images by platform                                               |
| `--deprecated`                                                  |                                        | Include deprecated images in the listing                                |
| `--max-workers` `MAX_WORKERS`                                   | `1`                                    | Maximum number of worker threads to use for copying images concurrently |
| `--max-retries` `MAX_RETRIES`                                   | `1`                                    | Maximum number of retries for copying an image                          |
| `--source`, `--src` `SOURCE`                                    | `docker.io`                            | Source registry                                                         |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE`      | (empty)                                | Source namespace in the source registry                                 |
| `--source-username`, `--src-user` `SOURCE_USERNAME`             | (env: `SOURCE_USERNAME`)               | Username for source registry authentication                             |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD`           | (env: `SOURCE_PASSWORD`)               | Password/Token for source registry authentication                       |
| `--archive-format` `{oci,docker}`                               | `oci`                                  | Archive format to save                                                  |
| `output`                                                        | (current working directory)            | Output directory to save images                                         |
