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
| `--tools-download-base-url` 値 | （空） | 依存ツールのダウンロード元のベース URL。 |
| `--save-archive` 値 | （空） | ダウンロードしたツールを tar アーカイブとして保存するパス。 |
| `--load-archive` 値 | （空） | ダウンロードする代わりに、ダウンロード済みのツールを読み込む tar アーカイブのパス。 |
| `--system` 値 | 現在の OS。 | ツールのダウンロード対象の OS。選択肢は `linux`、`windows`、`macos` です。 |
| `--arch` 値 | 現在のアーキテクチャ。 | ツールのダウンロード対象のアーキテクチャ。選択肢は `amd64`、`arm64` です。 |
