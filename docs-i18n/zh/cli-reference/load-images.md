---
translation_status: translated
hide:
  - toc
---

# gpustack load-images { #gpustack-load-images }

使用 https://github.com/containers/skopeo，将 OCI/Docker Archive 中的镜像加载到本地容器镜像存储。

```bash
gpustack load-images [OPTIONS] [input]

```

## 配置 { #configurations }

| <div style="width:220px">参数</div>                                   | <div style="width:120px">默认值</div>                          | 说明                                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--repository` `REPOSITORY`                                           | (empty)                                                         | 按仓库名称筛选镜像                                        |
| `--platform` `{linux/amd64,linux/arm64}`                              | (empty)                                                         | 按平台筛选镜像                                               |
| `--deprecated`                                                        |                                                                 | 在列表中包含已弃用的镜像                                |
| `--max-workers` `MAX_WORKERS`                                         | `1`                                                             | 并发复制镜像所使用的最大线程数 |
| `--max-retries` `MAX_RETRIES`                                         | `1`                                                             | 复制单个镜像的最大重试次数                          |
| `--destination`, `--dest` `DESTINATION`                               | `docker.io` (env: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | 覆盖目标镜像仓库                                           |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (env: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`)    | 覆盖目标镜像仓库中的命名空间                          |
| `--archive-format` `{oci,docker}`                                     | `oci`                                                           | 保存时使用的归档格式                                                  |
| `--storage` `{docker,podman}`                                         | `docker`                                                        | 加载镜像的目标容器镜像存储                             |
| `input`                                                               | (current working directory)                                     | 加载镜像的输入目录                                     |
