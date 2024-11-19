---
hide:
  - toc
---

# gpustack download-tools

Download dependency tools, including llama-box, gguf-parser, and fastfetch.

```bash
gpustack download-tools [OPTIONS]
```

## Configurations

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                   |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `----tools-download-base-url` value | (empty)                                | Base URL to download dependency tools.                                        |
| `--save-archive` value              | (empty)                                | Path to save downloaded tools as a tar archive.                               |
| `--load-archive` value              | (empty)                                | Path to load downloaded tools from a tar archive, instead of downloading.     |
| `--system` value                    | Default is the current OS.             | Operating system to download tools for. Options: `linux`, `windows`, `macos`. |
| `--arch` value                      | Default is the current architecture.   | Architecture to download tools for. Options: `amd64`, `arm64`.                |
| `--device` value                    | Default is the current device.         | Device to download tools for. Options: `cuda`, `mps`, `npu`, `musa`, `cpu`.   |
