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
    <a href="./docs/assets/wechat-assistant.png" target="_blank">
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

GPUStackは、AIモデルを実行するためのオープンソースGPUクラスタマネージャーです。

### 主な機能

- **幅広いGPU互換性:** Apple Mac、Windows PC、Linuxサーバー上のさまざまなベンダーのGPUをシームレスにサポート。
- **豊富なモデルサポート:** LLM、VLM、画像モデル、音声モデル、埋め込みモデル、リランクモデルを含む幅広いモデルをサポート。
- **柔軟な推論バックエンド:** llama-box（llama.cppとstable-diffusion.cpp）、vox-box、vLLM、Ascend MindIEと統合。
- **マルチバージョンバックエンドサポート:** 異なるモデルの多様なランタイム要件を満たすために、推論バックエンドの複数バージョンを同時実行。
- **分散推論:** ベンダーやランタイム環境をまたぐ異種GPUを含む、シングルノードおよびマルチノードのマルチGPU推論をサポート。
- **スケーラブルなGPUアーキテクチャ:** インフラストラクチャにGPUやノードを追加することで簡単にスケールアップ。
- **堅牢なモデル安定性:** 自動障害回復、マルチインスタンス冗長性、推論リクエストのロードバランシングで高可用性を確保。
- **インテリジェントなデプロイ評価:** モデルリソース要件、バックエンドとアーキテクチャの互換性、OSの互換性、その他のデプロイ関連要因を自動的に評価。
- **自動スケジューリング:** 利用可能なリソースに基づいてモデルを動的に割り当て。
- **軽量なPythonパッケージ:** 最小限の依存関係と低い運用オーバーヘッド。
- **OpenAI互換API:** OpenAIのAPI仕様と完全に互換性があり、シームレスな統合を実現。
- **ユーザーとAPIキー管理:** ユーザーとAPIキーの管理を簡素化。
- **リアルタイムGPU監視:** GPU性能と使用率をリアルタイムで追跡。
- **トークンとレートメトリクス:** トークン使用量とAPIリクエストレートを監視。

## インストール

### LinuxまたはmacOS

GPUStackは、systemdまたはlaunchdベースのシステムでサービスとしてインストールするスクリプトを提供しており、デフォルトポートは80です。この方法でGPUStackをインストールするには、以下を実行します：

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

### Windows

管理者としてPowerShellを実行し（PowerShell ISEの使用は**避けてください**）、以下のコマンドを実行してGPUStackをインストールします：

```powershell
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

### その他のインストール方法

手動インストール、Dockerインストール、または詳細な構成オプションについては、[インストールドキュメント](https://docs.gpustack.ai/latest/installation/installation-script/)を参照してください。

## はじめに

1. **llama3.2**モデルを実行してチャットする：

```bash
gpustack chat llama3.2 "tell me a joke."
```

2. **stable-diffusion-v3-5-large-turbo**モデルで画像を生成する：

> ### 💡 ヒント
>
> このコマンドはHugging Faceからモデル（約12GB）をダウンロードします。ダウンロード時間はネットワーク速度に依存します。モデルを実行するために十分なディスクスペースとVRAM（12GB）があることを確認してください。問題が発生した場合は、このステップをスキップして次に進むことができます。

```bash
gpustack draw hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:stable-diffusion-v3-5-large-turbo-Q4_0.gguf \
"A minion holding a sign that says 'GPUStack'. The background is filled with futuristic elements like neon lights, circuit boards, and holographic displays. The minion is wearing a tech-themed outfit, possibly with LED lights or digital patterns. The sign itself has a sleek, modern design with glowing edges. The overall atmosphere is high-tech and vibrant, with a mix of dark and neon colors." \
--sample-steps 5 --show
```

コマンドが完了すると、生成された画像がデフォルトビューアに表示されます。プロンプトとCLIオプションを実験して出力をカスタマイズできます。

![Generated Image](https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/quickstart-minion.png)

3. ブラウザで`http://your_host_ip`を開いてGPUStack UIにアクセスします。ユーザー名`admin`とデフォルトパスワードでGPUStackにログインします。デフォルト設定のパスワードを取得するには、以下のコマンドを実行します：

**LinuxまたはmacOS**

```bash
cat /var/lib/gpustack/initial_admin_password
```

**Windows**

```powershell
Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
```

4. ナビゲーションメニューで`Playground - Chat`をクリックします。これでUIプレイグラウンドでLLMとチャットできます。

![Playground Screenshot](https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/playground-screenshot.png)

5. ナビゲーションメニューで`API Keys`をクリックし、`New API Key`ボタンをクリックします。

6. `Name`を入力し、`Save`ボタンをクリックします。

7. 生成されたAPIキーをコピーして安全な場所に保存します。作成時にのみ一度だけ表示されることに注意してください。

8. これでAPIキーを使用してOpenAI互換APIにアクセスできます。例えば、curlを使用する場合：

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1-openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## サポートされているプラットフォーム

- [x] macOS
- [x] Linux
- [x] Windows

## サポートされているアクセラレータ

- [x] NVIDIA CUDA（[Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0以上）
- [x] Apple Metal（M系チップ）
- [x] AMD ROCm
- [x] Ascend CANN
- [x] Hygon DTK
- [x] Moore Threads MUSA
- [x] Iluvatar Corex

以下のアクセラレータは将来のリリースでサポートする予定です。

- [ ] Intel oneAPI
- [ ] Qualcomm AI Engine

## サポートされているモデル

GPUStackは[llama-box](https://github.com/gpustack/llama-box)（バンドルされた[llama.cpp](https://github.com/ggml-org/llama.cpp)と[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)サーバー）、[vLLM](https://github.com/vllm-project/vllm)、[Ascend MindIE](https://www.hiascend.com/en/software/mindie)、[vox-box](https://github.com/gpustack/vox-box)をバックエンドとして使用し、幅広いモデルをサポートしています。以下のソースからのモデルがサポートされています：

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. ローカルファイルパス

### モデル例：

| **カテゴリ**                 | **モデル**                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **大規模言語モデル（LLM）**  | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma)        |
| **ビジョン言語モデル（VLM）** | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL2.5](https://huggingface.co/models?search=internvl2_5) |
| **拡散モデル**               | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                                       |
| **埋め込みモデル**           | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings)                                                                                                                                                         |
| **リランカーモデル**         | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker)                                                                                                                                                |
| **音声モデル**               | [Whisper](https://huggingface.co/models?search=Systran/faster)（音声認識）、[CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice)（音声合成）                                                                                                                                                                                       |

サポートされているモデルの完全なリストについては、[推論バックエンド](https://docs.gpustack.ai/latest/user-guide/inference-backends/)ドキュメントのサポートされているモデルセクションを参照してください。

## OpenAI互換API

GPUStackは`/v1-openai`パスの下で以下のOpenAI互換APIを提供します：

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

例えば、公式の[OpenAI Python APIライブラリ](https://github.com/openai/openai-python)を使用してAPIを利用できます：

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

GPUStackユーザーはUIで独自のAPIキーを生成できます。

## ドキュメント

完全なドキュメントについては、[公式ドキュメントサイト](https://docs.gpustack.ai)を参照してください。

## ビルド

1. Python（バージョン3.10から3.12）をインストールします。

2. `make build`を実行します。

ビルドされたwheelパッケージは`dist`ディレクトリにあります。

## コントリビューション

GPUStackへの貢献に興味がある場合は、[コントリビューションガイド](./docs/contributing.md)をお読みください。

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