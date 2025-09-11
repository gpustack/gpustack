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

![demo](https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-demo.gif)

GPUStack は、AI モデルを実行するためのオープンソース GPU クラスタマネージャーです。

### 主な機能

- **幅広い GPU 互換性:** Apple Mac、Windows PC、Linux サーバー上のさまざまなベンダーの GPU をシームレスにサポート。
- **豊富なモデルサポート:** LLM、VLM、画像モデル、音声モデル、埋め込みモデル、リランクモデルを含む幅広いモデルをサポート。
- **柔軟な推論バックエンド:** vLLM、Ascend MindIE、llama-box（llama.cpp と stable-diffusion.cpp）、vox-box と統合。
- **マルチバージョンバックエンドサポート:** 異なるモデルの多様なランタイム要件を満たすために、推論バックエンドの複数バージョンを同時実行。
- **分散推論:** ベンダーやランタイム環境をまたぐ異種 GPU を含む、シングルノードおよびマルチノードのマルチ GPU 推論をサポート。
- **スケーラブルな GPU アーキテクチャ:** インフラストラクチャに GPU やノードを追加することで簡単にスケールアップ。
- **堅牢なモデル安定性:** 自動障害回復、マルチインスタンス冗長性、推論リクエストのロードバランシングで高可用性を確保。
- **インテリジェントなデプロイ評価:** モデルリソース要件、バックエンドとアーキテクチャの互換性、OS の互換性、その他のデプロイ関連要因を自動的に評価。
- **自動スケジューリング:** 利用可能なリソースに基づいてモデルを動的に割り当て。
- **軽量な Python パッケージ:** 最小限の依存関係と低い運用オーバーヘッド。
- **OpenAI 互換 API:** OpenAI の API 仕様と完全に互換性があり、シームレスな統合を実現。
- **ユーザーと API キー管理:** ユーザーと API キーの管理を簡素化。
- **リアルタイム GPU 監視:** GPU 性能と使用率をリアルタイムで追跡。
- **トークンとレートメトリクス:** トークン使用量と API リクエストレートを監視。

## インストール

### Linux

NVIDIA GPU を使用している場合は、Docker と NVIDIA Container Toolkit をインストールしてください。その後、以下のコマンドで GPUStack サーバーを起動します：

```bash
docker run -d --name gpustack \
      --restart=unless-stopped \
      --gpus all \
      --network=host \
      --ipc=host \
      -v gpustack-data:/var/lib/gpustack \
      gpustack/gpustack
```

詳細なインストール手順やその他の GPU ハードウェアプラットフォームについては、インストールドキュメント を参照してください。

サーバー起動後、次のコマンドでデフォルト管理者パスワードを取得できます：

```bash
cat /var/lib/gpustack/initial_admin_password
```

ブラウザで http://your_host_ip にアクセスし、ユーザー名 admin と取得したパスワードでログインします。

### macOS & Windows

macOS および Windows 向けにデスクトップインストーラーが用意されています。インストールの詳細は [ドキュメント](https://docs.gpustack.ai/latest/installation/desktop-installer/) をご覧ください。

## モデルのデプロイ

1. GPUStack UI の Catalog ページに移動します。

2. モデルリストから Qwen3 モデルを選択します。

3. デプロイ互換性チェックが完了したら、Save ボタンをクリックしてデプロイします。

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

4. モデルのダウンロードとデプロイが開始されます。ステータスが Running になると、デプロイ成功です。

![model is running](docs/assets/quick-start/model-running.png)

5. ナビゲーションメニューから Playground - Chat を選択し、右上の Model ドロップダウンで qwen3 が選択されていることを確認してチャットを開始します。

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
    "model": "qwen3",
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

## サポートされているプラットフォーム

- [x] Linux
- [x] macOS
- [x] Windows

## サポートされているアクセラレータ

- [x] NVIDIA CUDA（[Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 以上）
- [x] Apple Metal（M 系チップ）
- [x] AMD ROCm
- [x] Ascend CANN
- [x] Hygon DTK
- [x] Moore Threads MUSA
- [x] Iluvatar Corex
- [x] Cambricon MLU
- [x] Insi Mars

## サポートされているモデル

GPUStack は[vLLM](https://github.com/vllm-project/vllm)、[Ascend MindIE](https://www.hiascend.com/en/software/mindie)、[llama-box](https://github.com/gpustack/llama-box)（バンドルされた[llama.cpp](https://github.com/ggml-org/llama.cpp)と[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)サーバー）、[vox-box](https://github.com/gpustack/vox-box)をバックエンドとして使用し、幅広いモデルをサポートしています。以下のソースからのモデルがサポートされています：

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. ローカルファイルパス

### モデル例

| **カテゴリ**                  | **モデル**                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **大規模言語モデル（LLM）**   | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma)    |
| **ビジョン言語モデル（VLM）** | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL3](https://huggingface.co/models?search=internvl3) |
| **拡散モデル**                | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                                   |
| **埋め込みモデル**            | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings), [Qwen3-Embedding](https://huggingface.co/models?search=qwen/qwen3-embedding)                                                                       |
| **リランカーモデル**          | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker), [Qwen3-Reranker](https://huggingface.co/models?search=qwen/qwen3-reranker)                                                                |
| **音声モデル**                | [Whisper](https://huggingface.co/models?search=Systran/faster)（音声認識）、[CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice)（音声合成）                                                                                                                                                                                  |

サポートされているモデルの完全なリストについては、[推論バックエンド](https://docs.gpustack.ai/latest/user-guide/inference-backends/)ドキュメントのサポートされているモデルセクションを参照してください。

## OpenAI 互換 API

GPUStack は`/v1-openai`パスの下で以下の OpenAI 互換 API を提供します：

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
client = OpenAI(base_url="http://your_gpustack_server_url/v1-openai", api_key="your_api_key")

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
