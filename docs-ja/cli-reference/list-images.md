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
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (empty) | バックエンド名で gpustack/runner イメージを絞り込みます |
| `--backend-version` `BACKEND_VERSION` | (empty) | バックエンドのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX` | (empty) | バックエンドのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--backend-variant` `BACKEND_VARIANT` | (empty) | バックエンドのバリアントで gpustack/runner イメージを絞り込みます |
| `--service` `{voxbox,vllm,mindie,sglang}` | (empty) | サービス名で gpustack/runner イメージを絞り込みます |
| `--service-version` `SERVICE_VERSION` | (empty) | サービスのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX` | (empty) | サービスのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--repository` `REPOSITORY` | (empty) | リポジトリ名でイメージを絞り込みます |
| `--platform` `{linux/amd64,linux/arm64}` | (empty) | プラットフォームでイメージを絞り込みます |
| `--format {text,json}` | `text` | 出力形式。`text` は読みやすいテキスト形式、`json` は JSON 配列形式です |
