---
hide:
  - toc
---

# gpustack copy-images

Copy images to other registry, powered by https://github.com/containers/skopeo.

```bash
gpustack copy-images [OPTIONS]

```

## Configurations

| <div style="width:220px">Flag</div>                                   | <div style="width:120px">Default</div>                          | Description                                                             |
|-----------------------------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------|
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}`       | (empty)                                                         | Filter gpustack/runner images by backend name                           |
| `--backend-version` `BACKEND_VERSION`                                 | (empty)                                                         | Filter gpustack/runner images by exact backend version                  |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`                   | (empty)                                                         | Filter gpustack/runner images by backend version prefix                 |
| `--backend-variant` `BACKEND_VARIANT`                                 | (empty)                                                         | Filter gpustack/runner images by backend variant                        |
| `--service` `{voxbox,vllm,mindie,sglang}`                             | (empty)                                                         | Filter gpustack/runner images by service name                           |
| `--service-version` `SERVICE_VERSION`                                 | (empty)                                                         | Filter gpustack/runner images by exact service version                  |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`                   | (empty)                                                         | Filter gpustack/runner images by service version prefix                 |
| `--repository` `REPOSITORY`                                           | (empty)                                                         | Filter images by repository name                                        |
| `--platform` `{linux/amd64,linux/arm64}`                              | (empty)                                                         | Filter images by platform                                               |
| `--deprecated`                                                        |                                                                 | Include deprecated images in the listing                                |
| `--max-workers` `MAX_WORKERS`                                         | `1`                                                             | Maximum number of worker threads to use for copying images concurrently |
| `--max-retries` `MAX_RETRIES`                                         | `1`                                                             | Maximum number of retries for copying an image                          |
| `--source`, `--src` `SOURCE`                                          | `docker.io`                                                     | Source registry                                                         |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE`            | (empty)                                                         | Source namespace in the source registry                                 |
| `--source-username`, `--src-user` `SOURCE_USERNAME`                   | (env: `SOURCE_USERNAME`)                                        | Username for source registry authentication                             |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD`                 | (env: `SOURCE_PASSWORD`)                                        | Password/Token for source registry authentication                       |
| `--destination`, `--dest` `DESTINATION`                               | `docker.io` (env: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | Destination registry                                                    |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (env: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`)    | Destination namespace in the destination registry                       |
| `--destination-username`, `--dest-user` `DESTINATION_USERNAME`        | (env: `DESTINATION_USERNAME`)                                   | Username for destination registry authentication                        |
| `--destination-password`, `--dest-passwd` `DESTINATION_PASSWORD`      | (env: `DESTINATION_PASSWORD`)                                   | Password/Token for destination registry authentication                  |
