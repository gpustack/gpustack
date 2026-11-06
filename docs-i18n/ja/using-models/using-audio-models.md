---
translation_status: translated
---

# 音声モデルの利用 { #using-audio-models }

GPUStack は、音声認識（Speech-to-Text）と音声合成（Text-to-Speech）の両方に対応しています。音声認識モデルは各言語の音声をテキストに変換し、音声合成モデルはテキストを自然で表現力のある音声に変換します。

このガイドでは、GPUStack で音声認識モデルと音声合成モデルをデプロイして利用する方法を説明します。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- GPUStack がインストール済みで、稼働していること。未導入の場合は[クイックスタート](../quickstart.md)をご覧ください。
- モデルファイルをダウンロードするために Hugging Face または ModelScope にアクセスできること。

## 音声認識モデルの実行 { #running-speech-to-text-model }

### 手順 1：音声認識モデルのデプロイ { #step-1-deploy-speech-to-text-model }

モデルカタログからモデルをデプロイするには、次の手順を実行します。

1. GPUStack UI の `Model Catalog` ページに移動します。
2. カテゴリーフィルターで `Speech-to-Text` を選択し、`Whisper-Large-V3-Turbo` モデルを選びます。
3. デフォルト設定のまま `Save` ボタンをクリックし、モデルをデプロイします。

![モデルのデプロイ](../assets/using-models/using-audio-models/deploy-stt-model.png)

デプロイ後は、`Deployments` ページで状態を確認できます。成功したら、デプロイメントの三点アイコンから `Open in Playground` を選択し、プレイグラウンドでモデルを使い始めます。

![モデル一覧](../assets/using-models/using-audio-models/stt-model-list.png)

### 手順 2：音声認識モデルの利用 { #step-2-interact-with-speech-to-text-model }

`Speech to Text` プレイグラウンドで次の操作を行います。

1. `Upload` ボタンで音声ファイルをアップロードするか、`Microphone` ボタンで録音します。
2. `Generate Text Content` ボタンをクリックして文字起こしを生成します。

![生成結果](../assets/using-models/using-audio-models/inference-stt-model.png)

### 手順 3：API 経由のストリーミング出力 { #step-3-streaming-output-via-api }

API で文字起こしをストリーミング取得することもできます。次は curl の例です。

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -F model="whisper-large-v3-turbo" \
  -F file="@/path/to/audio-file;type=audio/mpeg" \
  -F language="en" \
  -F stream="true"
```

文字起こしの結果が利用可能になるたびに、ストリーミングで返されます。

!!! note

    文字起こしのストリーミングは、音声認識モデルを `vLLM` バックエンドにデプロイした場合のみ対応しています。`VoxBox` バックエンドはストリーミングに対応していません。

## 音声合成モデルの実行 { #running-text-to-speech-model }

### 手順 1：音声合成モデルのデプロイ { #step-1-deploy-text-to-speech-model }

モデルカタログからモデルをデプロイするには、次の手順を実行します。

1. GPUStack UI の `Model Catalog` ページに移動します。
2. カテゴリーフィルターで `Text-to-Speech` を選択し、`Qwen3-TTS-12Hz-1.7B-CustomVoice` モデルを選びます。
3. デフォルト設定のまま `Save` ボタンをクリックし、モデルをデプロイします。

![モデルのデプロイ](../assets/using-models/using-audio-models/deploy-tts-model.png)

デプロイ後は、`Deployments` ページで状態を確認できます。成功したら、デプロイメントの三点アイコンから `Open in Playground` を選択し、プレイグラウンドでモデルを使い始めます。

![モデル一覧](../assets/using-models/using-audio-models/tts-model-list.png)

### 手順 2：音声合成モデルの利用 { #step-2-interact-with-text-to-speech-model }

`Text to Speech` プレイグラウンドで次の操作を行います。

1. `Voice` ドロップダウンで希望の音声を選択します。
2. （任意）`Instructions` を入力し、生成する音声のスタイルを指定します。
3. 音声に変換するテキストを入力します。
4. `Submit` ボタンをクリックして音声を生成します。

![生成結果](../assets/using-models/using-audio-models/inference-tts-model.png)

### 手順 3：API 経由のストリーミング出力 { #step-3-streaming-output-via-api_1 }

API で音声出力をストリーミング取得することもできます。次は curl の例です。

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -d '{
    "model": "qwen3-tts-12hz-1.7b-customvoice",
    "voice": "Vivian",
    "task_type": "CustomVoice",
    "language": "Auto",
    "input": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "stream": true,
    "response_format": "pcm"
  }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

音声が直接ストリーミングされ、`play` コマンドで再生されます。形式は PCM、サンプリングレートは 24kHz、16 ビット符号付き、モノラルです。

!!! note

    音声のストリーミングは、音声合成モデルを `vLLM` バックエンドにデプロイした場合のみ対応しています。`VoxBox` バックエンドはストリーミングに対応していません。

## Qwen3-TTS による音声クローニング { #voice-cloning-using-qwen3-tts }

GPUStack は、音声合成モデルによる音声クローニングにも対応しています。利用方法は次のとおりです。

### 手順 1：音声クローニングモデルのデプロイ { #step-1-deploy-voice-cloning-model }

1. GPUStack UI の `Model Catalog` ページに移動します。
2. カテゴリーフィルターで `Text-to-Speech` を選択し、`Qwen3-TTS-12Hz-1.7B-Base` モデルを選びます。
3. デフォルト設定のまま `Save` ボタンをクリックし、モデルをデプロイします。

### 手順 2：プレイグラウンドでの音声クローニング { #step-2-use-voice-cloning-in-playground }

デプロイに成功したら、デプロイメントの三点アイコンから `Open in Playground` を選択します。プレイグラウンドで次の手順を実行します。

1. `Reference Audio` に音声ファイルをアップロードするか、音声 URL を入力し、複製元の声を指定します。たとえば `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-0115/APS-en_33.wav` を入力できます。
2. `Use Speaker Embedding Only (no ICL)` を選択します。
3. 合成するテキストを入力します。例：`Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.`
4. `Submit` ボタンをクリックし、複製した声で音声を生成します。

![音声クローニング](../assets/using-models/using-audio-models/voice-clone.png)
