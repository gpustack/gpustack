---
translation_status: translated
hide:
  - toc
---

# gpustack list-images { #gpustack-list-images }

列出镜像。

```bash
gpustack list-images [OPTIONS]

```

## 配置 { #configurations }

| <div style="width:220px">参数</div>                             | <div style="width:120px">默认值</div> | 说明                                                                        |
|-----------------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------|
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (empty)                                | 按后端名称筛选 gpustack/runner 镜像                                      |
| `--backend-version` `BACKEND_VERSION`                           | (empty)                                | 按精确的后端版本筛选 gpustack/runner 镜像                             |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (empty)                                | 按后端版本前缀筛选 gpustack/runner 镜像                            |
| `--backend-variant` `BACKEND_VARIANT`                           | (empty)                                | 按后端变体筛选 gpustack/runner 镜像                                   |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (empty)                                | 按服务名称筛选 gpustack/runner 镜像                                      |
| `--service-version` `SERVICE_VERSION`                           | (empty)                                | 按精确的服务版本筛选 gpustack/runner 镜像                             |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (empty)                                | 按服务版本前缀筛选 gpustack/runner 镜像                            |
| `--repository` `REPOSITORY`                                     | (empty)                                | 按仓库名称筛选镜像                                                   |
| `--platform` `{linux/amd64,linux/arm64}`                        | (empty)                                | 按平台筛选镜像                                                          |
| `--format {text,json}`                                          | `text`                                 | 输出格式。`text` 表示供人阅读的文本格式，`json` 表示 JSON 数组格式 |
