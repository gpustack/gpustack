---
hide:
  - toc
---

# gpustack download-tools { #gpustack-download-tools }

gguf-parser や fastfetch などの依存ツールをダウンロードします。

```bash
gpustack download-tools [OPTIONS]
```

## 構成 { #configurations }

| <div style="width:180px">フラグ</div> | <div style="width:100px">既定値</div> | 説明 |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `----tools-download-base-url` value | (empty) | 依存ツールのダウンロード元のベース URL。 |
| `--save-archive` value | (empty) | ダウンロードしたツールを tar アーカイブとして保存するパス。 |
| `--load-archive` value | (empty) | ダウンロードする代わりに、ダウンロード済みのツールを読み込む tar アーカイブのパス。 |
| `--system` value | Default is the current OS. | ツールのダウンロード対象の OS。選択肢は `linux`、`windows`、`macos` です。 |
| `--arch` value | Default is the current architecture. | ツールのダウンロード対象のアーキテクチャ。選択肢は `amd64`、`arm64` です。 |
