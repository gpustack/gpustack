---
hide:
  - toc
---

# gpustack save-images { #gpustack-save-images }

https://github.com/containers/skopeo を使用して、イメージを OCI/Docker Archive としてローカルパスに保存します。

```bash
gpustack save-images [OPTIONS] [output]

```

## 構成 { #configurations }

| <div style="width:220px">フラグ</div> | <div style="width:120px">既定値</div> | 説明 |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | （空） | バックエンド名で gpustack/runner イメージを絞り込みます |
| `--backend-version` `BACKEND_VERSION` | （空） | バックエンドのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX` | （空） | バックエンドのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--backend-variant` `BACKEND_VARIANT` | （空） | バックエンドのバリアントで gpustack/runner イメージを絞り込みます |
| `--service` `{voxbox,vllm,mindie,sglang}` | （空） | サービス名で gpustack/runner イメージを絞り込みます |
| `--service-version` `SERVICE_VERSION` | （空） | サービスのバージョンの完全一致で gpustack/runner イメージを絞り込みます |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX` | （空） | サービスのバージョンの接頭辞で gpustack/runner イメージを絞り込みます |
| `--repository` `REPOSITORY` | （空） | リポジトリ名でイメージを絞り込みます |
| `--platform` `{linux/amd64,linux/arm64}` | （空） | プラットフォームでイメージを絞り込みます |
| `--deprecated` |  | 非推奨のイメージも一覧に含めます |
| `--max-workers` `MAX_WORKERS` | `1` | イメージの並列コピーに使用する Worker スレッドの最大数 |
| `--max-retries` `MAX_RETRIES` | `1` | イメージコピーの最大再試行回数 |
| `--source`, `--src` `SOURCE` | `docker.io` | 転送元のレジストリ |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE` | （空） | 転送元レジストリ内の名前空間 |
| `--source-username`, `--src-user` `SOURCE_USERNAME` | (env: `SOURCE_USERNAME`) | 転送元レジストリの認証に使用するユーザー名 |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD` | (env: `SOURCE_PASSWORD`) | 転送元レジストリの認証に使用するパスワードまたはトークン |
| `--archive-format` `{oci,docker}` | `oci` | 保存するアーカイブの形式 |
| `output` | （現在の作業ディレクトリー） | イメージを保存する出力ディレクトリ |
