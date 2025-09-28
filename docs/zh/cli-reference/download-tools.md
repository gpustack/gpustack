---
hide:
  - toc
---

# gpustack download-tools

下载依赖工具，包括 llama-box、gguf-parser 和 fastfetch。

```bash
gpustack download-tools [OPTIONS]
```

## 配置

| <div style="width:180px">标志</div> | <div style="width:100px">默认值</div> | 说明                                                                                 |
| ----------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| `----tools-download-base-url` 值    | （空）                               | 用于下载依赖工具的基础 URL。                                                         |
| `--save-archive` 值                 | （空）                               | 将已下载工具保存为 tar 归档的路径。                                                  |
| `--load-archive` 值                 | （空）                               | 从 tar 归档加载已下载工具的路径（而不是在线下载）。                                  |
| `--system` 值                       | 默认为当前操作系统。                 | 要为其下载工具的操作系统。可选项：`linux`、`windows`、`macos`。                      |
| `--arch` 值                         | 默认为当前架构。                     | 要为其下载工具的体系结构。可选项：`amd64`、`arm64`。                                 |
| `--device` 值                       | 默认为当前设备。                     | 要为其下载工具的设备。可选项：`cuda`、`mps`、`npu`、`dcu`、`musa`、`cpu`。            |