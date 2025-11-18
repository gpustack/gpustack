---
hide:
  - toc
---

# gpustack reload-config

Reload runtime-safe configuration values.

```bash
gpustack reload-config [OPTIONS]
```

## Configurations

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------ | -------------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--set` value                  | (empty)                                | Set a single whitelisted field using `key=value` (keys in hyphen-case). Values are coerced by field type: booleans accept `true/false`, `1/0`, `yes/no`, `y/n`, `on/off`; lists accept comma-separated strings; dicts require JSON strings. Multiple `--set` flags are allowed; later ones override earlier ones and override `--file` values. |
| `--file` value                 | (empty)                                | Load configuration from a YAML file. Only whitelisted fields are applied. Keys are normalized to snake_case. Values provided via `--set` override those from the file.                                                                                                                                                                         |
| `--list`                       | `False`                                | Show whitelisted fields and values. When present, other options are ignored.                                                                                                                                                                                                                                                                     |
