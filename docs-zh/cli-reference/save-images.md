---
translation_status: translated
hide:
  - toc
---

# gpustack save-images { #gpustack-save-images }

使用 https://github.com/containers/skopeo，将镜像以 OCI/Docker Archive 格式保存到本地路径。

```bash
gpustack save-images [OPTIONS] [output]

```

## 配置 { #configurations }

| <div style="width:220px">参数</div>                             | <div style="width:120px">默认值</div> | 说明                                                             |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (empty)                                | 按后端名称筛选 gpustack/runner 镜像                           |
| `--backend-version` `BACKEND_VERSION`                           | (empty)                                | 按精确的后端版本筛选 gpustack/runner 镜像                  |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (empty)                                | 按后端版本前缀筛选 gpustack/runner 镜像                 |
| `--backend-variant` `BACKEND_VARIANT`                           | (empty)                                | 按后端变体筛选 gpustack/runner 镜像                        |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (empty)                                | 按服务名称筛选 gpustack/runner 镜像                           |
| `--service-version` `SERVICE_VERSION`                           | (empty)                                | 按精确的服务版本筛选 gpustack/runner 镜像                  |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (empty)                                | 按服务版本前缀筛选 gpustack/runner 镜像                 |
| `--repository` `REPOSITORY`                                     | (empty)                                | 按仓库名称筛选镜像                                        |
| `--platform` `{linux/amd64,linux/arm64}`                        | (empty)                                | 按平台筛选镜像                                               |
| `--deprecated`                                                  |                                        | 在列表中包含已弃用的镜像                                |
| `--max-workers` `MAX_WORKERS`                                   | `1`                                    | 并发复制镜像所使用的最大线程数 |
| `--max-retries` `MAX_RETRIES`                                   | `1`                                    | 复制单个镜像的最大重试次数                          |
| `--source`, `--src` `SOURCE`                                    | `docker.io`                            | 源镜像仓库                                                         |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE`      | (empty)                                | 源镜像仓库中的源命名空间                                 |
| `--source-username`, `--src-user` `SOURCE_USERNAME`             | (env: `SOURCE_USERNAME`)               | 源镜像仓库身份认证使用的用户名                             |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD`           | (env: `SOURCE_PASSWORD`)               | 源镜像仓库身份认证使用的密码或 Token                       |
| `--archive-format` `{oci,docker}`                               | `oci`                                  | 保存时使用的归档格式                                                  |
| `output`                                                        | (current working directory)            | 保存镜像的输出目录                                         |
