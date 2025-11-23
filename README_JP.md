<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>
<br>

<p align="center">
    <a href="https://docs.gpustack.ai" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/badge/ドキュメント-GPUStack-blue?logo=readthedocs&logoColor=white"></a>
    <a href="./LICENSE" target="_blank">
        <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue"></a>
    <a href="./docs/assets/wechat-group-qrcode.jpg" target="_blank">
        <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white"></a>
    <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white"></a>
    <a href="https://twitter.com/intent/follow?screen_name=gpustack_ai" target="_blank">
        <img alt="Follow on X(Twitter)" src="https://img.shields.io/twitter/follow/gpustack_ai?logo=X"></a>
</p>
<br>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

GPUStack は、AI モデルを実行するためのオープンソース GPU クラスタマネージャーです。

### 主な機能

- **幅広い GPU 互換性:** Apple Mac、Windows PC、Linux サーバー上のさまざまなベンダーの GPU をシームレスにサポート。
- **豊富なモデルサポート:** LLM、VLM、画像モデル、音声モデル、埋め込みモデル、リランクモデルを含む幅広いモデルをサポート。
- **柔軟な推論バックエンド:** vLLM、SGLang、MindIE、vox-box と統合。
- **マルチバージョンバックエンドサポート:** 異なるモデルの多様なランタイム要件を満たすために、推論バックエンドの複数バージョンを同時実行。
- **分散推論:** ベンダーやランタイム環境をまたぐ異種 GPU を含む、シングルノードおよびマルチノードのマルチ GPU 推論をサポート。
- **スケーラブルな GPU アーキテクチャ:** インフラストラクチャに GPU やノードを追加することで簡単にスケールアップ。
- **堅牢なモデル安定性:** 自動障害回復、マルチインスタンス冗長性、推論リクエストのロードバランシングで高可用性を確保。
- **インテリジェントなデプロイ評価:** モデルリソース要件、バックエンドとアーキテクチャの互換性、OS の互換性、その他のデプロイ関連要因を自動的に評価。
- **自動スケジューリング:** 利用可能なリソースに基づいてモデルを動的に割り当て。
- **OpenAI 互換 API:** OpenAI の API 仕様と完全に互換性があり、シームレスな統合を実現。
- **ユーザーと API キー管理:** ユーザーと API キーの管理を簡素化。
- **リアルタイム GPU 監視:** GPU 性能と使用率をリアルタイムで追跡。
- **トークンとレートメトリクス:** トークン使用量と API リクエストレートを監視。

## インストール

> GPUStack は現在 Linux のみをサポートしています。Windows を使用する場合は WSL2 を利用し、Docker Desktop は使用しないでください。

NVIDIA GPU を使用している場合は、NVIDIA ドライバー、Docker、および NVIDIA Container Toolkit をインストールしてください。その後、以下のコマンドで GPUStack サーバーを起動します：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    gpustack/gpustack
```

もし Docker Hub からイメージをダウンロードできない、またはダウンロードが非常に遅い場合は、提供している `Quay.io` ミラーを利用できます。Registry を `quay.io` に指定してミラーを使用してください：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```

他のプラットフォームへのインストール方法や詳細な設定オプションについては、[インストール要件](docs/installation/requirements.md) を参照してください。

GPUStack の起動ログを確認するには：

```bash
sudo docker logs -f gpustack
```

サーバー起動後、次のコマンドでデフォルト管理者パスワードを取得できます：

```bash
sudo docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

ブラウザで http://your_host_ip にアクセスし、ユーザー名 admin と取得したパスワードでログインします。

## モデルのデプロイ

1. GPUStack UI の Catalog ページに移動します。

2. モデルリストから Qwen3 0.6B モデルを選択します。

3. デプロイ互換性チェックが完了したら、Save ボタンをクリックしてデプロイします。

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

4. モデルのダウンロードとデプロイが開始されます。ステータスが Running になると、デプロイ成功です。

![model is running](docs/assets/quick-start/model-running.png)

5. ナビゲーションメニューから Playground - Chat を選択し、右上の Model ドロップダウンで qwen3-0.6B が選択されていることを確認してチャットを開始します。

![quick chat](docs/assets/quick-start/quick-chat.png)

## API でモデルを使用する

1. ユーザーアバターをホバーし、API Keys ページに移動後、New API Key をクリックします。

2. Name を入力し、Save をクリックします。

3. 生成された API キーをコピーして安全な場所に保管してください（一度しか表示されません）。

4. OpenAI 互換エンドポイントにアクセスできます。例：

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3-0.6B",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## サポートされているアクセラレータ

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU (Experimental)
- [x] MThreads GPU (Experimental)
- [x] Iluvatar GPU (Experimental)
- [x] MetaX GPU (Experimental)
- [x] Cambricon MLU (Experimental)

## サポートされているモデル

GPUStack は [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[MindIE](https://www.hiascend.com/en/software/mindie)、および [vox-box](https://github.com/gpustack/vox-box) をバックエンドとして使用し、さらにコンテナ上で実行可能でサービス API を提供できる任意のカスタム推論バックエンドもサポートすることで、幅広いモデルに対応しています。

以下のソースからのモデルがサポートされています：

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. ローカルファイルパス

各組み込み推論バックエンドがサポートするモデルについては、[Built-in Inference Backends](https://docs.gpustack.ai/latest/user-guide/built-in-inference-backends/) ドキュメントの Supported Models セクションを参照してください。

## OpenAI 互換 API

GPUStack は`/v1`パスの下で以下の OpenAI 互換 API を提供します：

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

例えば、公式の[OpenAI Python API ライブラリ](https://github.com/openai/openai-python)を使用して API を利用できます：

```python
from openai import OpenAI
client = OpenAI(base_url="http://your_gpustack_server_url/v1", api_key="your_api_key")

completion = client.chat.completions.create(
  model="llama3.2",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

GPUStack ユーザーは UI で独自の API キーを生成できます。

## ドキュメント

完全なドキュメントについては、[公式ドキュメントサイト](https://docs.gpustack.ai)を参照してください。

## ビルド

1. Python（バージョン 3.10 から 3.12）をインストールします。

2. `make build`を実行します。

ビルドされた wheel パッケージは`dist`ディレクトリにあります。

## コントリビューション

GPUStack への貢献に興味がある場合は、[コントリビューションガイド](./docs/contributing.md)をお読みください。

## コミュニティに参加

問題がある場合や提案がある場合は、サポートのために[コミュニティ](https://discord.gg/VXYJzuaqwD)に参加してください。

## ライセンス

Copyright (c) 2024 The GPUStack authors

Apache License, Version 2.0（以下「ライセンス」）に基づいてライセンスされています。
このライセンスの詳細については、[LICENSE](./LICENSE)ファイルを参照してください。

適用法で要求されるか、書面で合意されない限り、
ライセンスに基づいて配布されるソフトウェアは「現状のまま」で配布され、
明示または黙示を問わず、いかなる種類の保証や条件もありません。
ライセンスに基づく許可と制限を規定する特定の言語については、
ライセンスを参照してください。
