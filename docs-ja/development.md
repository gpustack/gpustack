---
translation_status: translated
---

# 開発ガイド { #development-guide }

## 前提条件 { #prerequisites }

1. Python（バージョン 3.10～3.12）をインストールします。
2. データベース（PostgreSQL または MySQL）を起動します。

たとえば、Docker で PostgreSQL を起動するには次を実行します。
```bash
docker run --name my-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d --restart=always postgres
```

## 環境のセットアップ { #set-up-environment }

```bash
make install
```

## 実行 { #run }

開発時はゲートウェイモードを `disabled` にし、データベースに接続します。

```bash
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

## ビルド { #build }

```bash
make build
```

`dist` 内の成果物を確認します。

## テスト { #test }

```bash
make test
```

## パッケージ化 { #package }

コンテナーイメージのビルドには、[Buildx](https://docs.docker.com/build/install-buildx/) プラグインを備えた Docker が必要です。

```bash
make package
```

`docker buildx version` が失敗する場合は、パッケージマネージャー（`apt-get install docker-buildx-plugin` または `yum install docker-buildx-plugin`）でプラグインをインストールするか、リリースバイナリを配置します。

```bash
mkdir -p /usr/local/lib/docker/cli-plugins
VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/download/${VER}/buildx-${VER}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
docker buildx version
```

未認証の GitHub API は、IP ごとに 1 時間あたり 60 リクエストまでです。制限に達すると `VER` が空になり、ダウンロードが 404 になります。[リリースページ](https://github.com/docker/buildx/releases)からバージョンを選び、手動で設定してください。

`sudo` でパッケージ化する場合は、`HOME` の変更により Docker CLI からプラグインが見えなくなるため、`~/.docker/cli-plugins` ではなく `/usr/local/lib/docker/cli-plugins` にインストールします。

## 依存関係の更新 { #update-dependencies }

```bash
uv add <something>
```

または、

```bash
uv add --dev <something>
```

開発・テスト用の依存関係にはこちらを使用します。
