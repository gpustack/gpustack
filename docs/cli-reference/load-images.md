---
hide:
  - toc
---

# gpustack load-images

Load images from OCI/Docker Archive to local container image storage, powered by https://github.com/containers/skopeo.

```bash
gpustack load-images [OPTIONS] [input]

```

## Configurations

| <div style="width:220px">Flag</div>                                   | <div style="width:120px">Default</div>                          | Description                                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--repository` `REPOSITORY`                                           | (empty)                                                         | Filter images by repository name                                        |
| `--platform` `{linux/amd64,linux/arm64}`                              | (empty)                                                         | Filter images by platform                                               |
| `--deprecated`                                                        |                                                                 | Include deprecated images in the listing                                |
| `--max-workers` `MAX_WORKERS`                                         | `1`                                                             | Maximum number of worker threads to use for copying images concurrently |
| `--max-retries` `MAX_RETRIES`                                         | `1`                                                             | Maximum number of retries for copying an image                          |
| `--destination`, `--dest` `DESTINATION`                               | `docker.io` (env: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | Override destination registry                                           |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (env: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`)    | Override namespace in the destination registry                          |
| `--archive-format` `{oci,docker}`                                     | `oci`                                                           | Archive format to save                                                  |
| `--storage` `{docker,podman}`                                         | `docker`                                                        | Container image storage to load images into                             |
| `input`                                                               | (current working directory)                                     | Input directory to load images from                                     |
