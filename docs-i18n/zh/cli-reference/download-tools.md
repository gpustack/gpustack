---
translation_status: translated
hide:
  - toc
---

# gpustack download-tools { #gpustack-download-tools }

下载依赖工具，包括 gguf-parser 和 fastfetch。

```bash
gpustack download-tools [OPTIONS]
```

## 配置 { #configurations }

| <div style="width:180px">参数</div> | <div style="width:100px">默认值</div> | 说明                                                                   |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `--tools-download-base-url` value | (empty)                                | 下载依赖工具的基础 URL。                                        |
| `--save-archive` value              | (empty)                                | 将已下载工具保存为 tar 归档的路径。                               |
| `--load-archive` value              | (empty)                                | 从 tar 归档加载已下载工具的路径，以替代下载。     |
| `--system` value                    | Default is the current OS.             | 下载工具的目标操作系统。可选值：`linux`、`windows`、`macos`。 |
| `--arch` value                      | Default is the current architecture.   | 下载工具的目标架构。可选值：`amd64`、`arm64`。                |
