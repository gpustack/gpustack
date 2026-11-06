---
hide:
  - toc
---

# gpustack list-images { #gpustack-list-images }

イメージを一覧表示します。

```bash
gpustack list-images [OPTIONS]

```

## 構成 { #configurations }

| <div style="width:220px">フラグ</div> | <div style="width:120px">既定値</div> | 説明 |
| ----------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | （空） | バックエンド名で gpustack/runner イメージを絞り込みます |
| `--backend-version` `BACKEND_VERSION` | （空） | バックエンドのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX` | （空） | バックエンドのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--backend-variant` `BACKEND_VARIANT` | （空） | バックエンドのバリアントで gpustack/runner イメージを絞り込みます |
| `--service` `{voxbox,vllm,mindie,sglang}` | （空） | サービス名で gpustack/runner イメージを絞り込みます |
| `--service-version` `SERVICE_VERSION` | （空） | サービスのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX` | （空） | サービスのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--repository` `REPOSITORY` | （空） | リポジトリ名でイメージを絞り込みます |
| `--platform` `{linux/amd64,linux/arm64}` | （空） | プラットフォームでイメージを絞り込みます |
| `--format {text,json}` | `text` | 出力形式。`text` は読みやすいテキスト形式、`json` は JSON 配列形式です |
