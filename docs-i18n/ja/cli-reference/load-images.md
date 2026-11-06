---
hide:
  - toc
---

# gpustack load-images { #gpustack-load-images }

https://github.com/containers/skopeo を使用して、OCI/Docker Archive からローカルのコンテナイメージストレージにイメージを読み込みます。

```bash
gpustack load-images [OPTIONS] [input]

```

## 構成 { #configurations }

| <div style="width:220px">フラグ</div> | <div style="width:120px">既定値</div> | 説明 |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--repository` `REPOSITORY` | （空） | リポジトリ名でイメージを絞り込みます |
| `--platform` `{linux/amd64,linux/arm64}` | （空） | プラットフォームでイメージを絞り込みます |
| `--deprecated` |  | 非推奨のイメージも一覧に含めます |
| `--max-workers` `MAX_WORKERS` | `1` | イメージの並列コピーに使用する Worker スレッドの最大数 |
| `--max-retries` `MAX_RETRIES` | `1` | イメージコピーの最大再試行回数 |
| `--destination`, `--dest` `DESTINATION` | `docker.io` (env: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | 転送先レジストリを上書きします |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (env: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`) | 転送先レジストリ内の名前空間を上書きします |
| `--archive-format` `{oci,docker}` | `oci` | 保存するアーカイブの形式 |
| `--storage` `{docker,podman}` | `docker` | イメージの読み込み先となるコンテナイメージストレージ |
| `input` | （現在の作業ディレクトリー） | イメージを読み込む入力ディレクトリ |
