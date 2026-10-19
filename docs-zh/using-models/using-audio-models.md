---
translation_status: translated
---

# 使用音频模型 { #using-audio-models }

GPUStack 支持语音转文本和文本转语音模型。语音转文本模型可以将不同语言的音频输入转换为文字；文本转语音模型则将文字转换为自然、有表现力的语音。

本指南介绍如何在 GPUStack 中部署和使用这两类模型。

## 前提条件 { #prerequisites }

开始前，请确认：

- GPUStack 已安装并运行；如未完成，请参考[快速入门](../quickstart.md)。
- 可以访问 Hugging Face 或 ModelScope，以下载模型文件。

## 运行语音转文本模型 { #running-speech-to-text-model }

### 步骤 1：部署语音转文本模型 { #step-1-deploy-speech-to-text-model }

按以下步骤从 Model Catalog 部署模型：

1. 在 GPUStack 界面中打开 `Model Catalog` 页面。
2. 在分类筛选中选择 `Speech-to-Text`，然后选择 `Whisper-Large-V3-Turbo` 模型。
3. 保持默认设置，点击 `Save` 部署模型。

![部署模型](../assets/using-models/using-audio-models/deploy-stt-model.png)

部署后，可在 `Deployments` 页面查看模型部署状态。部署成功后，点击部署项的省略号图标，选择 `Open in Playground`，即可在 Playground 中使用模型。

![模型列表](../assets/using-models/using-audio-models/stt-model-list.png)

### 步骤 2：使用语音转文本模型 { #step-2-interact-with-speech-to-text-model }

在 `Speech to Text` Playground 中：

1. 点击 `Upload` 上传音频文件，或点击 `Microphone` 录音。
2. 点击 `Generate Text Content` 生成转写文本。

![生成结果](../assets/using-models/using-audio-models/inference-stt-model.png)

### 步骤 3：通过 API 获取流式输出 { #step-3-streaming-output-via-api }

也可以通过 API 获取流式转写结果。以下是使用 curl 的示例：

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

该请求会在转写结果产生时持续返回内容。

!!! note

    只有部署在 `vLLM` 后端上的语音转文本模型支持流式转写；`VoxBox` 后端不支持流式输出。

## 运行文本转语音模型 { #running-text-to-speech-model }

### 步骤 1：部署文本转语音模型 { #step-1-deploy-text-to-speech-model }

按以下步骤从 Model Catalog 部署模型：

1. 在 GPUStack 界面中打开 `Model Catalog` 页面。
2. 在分类筛选中选择 `Text-to-Speech`，然后选择 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 模型。
3. 保持默认设置，点击 `Save` 部署模型。

![部署模型](../assets/using-models/using-audio-models/deploy-tts-model.png)

部署后，可在 `Deployments` 页面查看模型部署状态。部署成功后，点击部署项的省略号图标，选择 `Open in Playground`，即可在 Playground 中使用模型。

![模型列表](../assets/using-models/using-audio-models/tts-model-list.png)

### 步骤 2：使用文本转语音模型 { #step-2-interact-with-text-to-speech-model }

在 `Text to Speech` Playground 中：

1. 在 `Voice` 下拉菜单中选择所需音色。
2. 可选：填写 `Instructions`，指导模型生成所需风格的语音。
3. 输入待转换为语音的文本。
4. 点击 `Submit` 生成音频。

![生成结果](../assets/using-models/using-audio-models/inference-tts-model.png)

### 步骤 3：通过 API 获取流式输出 { #step-3-streaming-output-via-api_1 }

也可以通过 API 获取流式音频输出。以下是使用 curl 的示例：

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

该请求会持续输出音频，并使用 `play` 命令直接播放。音频采用 PCM 格式，采样率为 24kHz、16 位有符号编码、单声道。

!!! note

    只有部署在 `vLLM` 后端上的文本转语音模型支持流式语音；`VoxBox` 后端不支持流式输出。

## 使用 Qwen3-TTS 克隆音色 { #voice-cloning-using-qwen3-tts }

GPUStack 也支持通过文本转语音模型克隆音色，操作如下：

### 步骤 1：部署音色克隆模型 { #step-1-deploy-voice-cloning-model }

1. 在 GPUStack 界面中打开 `Model Catalog` 页面。
2. 在分类筛选中选择 `Text-to-Speech`，然后选择 `Qwen3-TTS-12Hz-1.7B-Base` 模型。
3. 保持默认设置，点击 `Save` 部署模型。

### 步骤 2：在 Playground 中克隆音色 { #step-2-use-voice-cloning-in-playground }

部署成功后，点击部署项的省略号图标，选择 `Open in Playground`，然后：

1. 在 `Reference Audio` 中上传音频文件或输入音频 URL，提供待克隆的参考音色。例如，可输入 `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-0115/APS-en_33.wav`。
2. 勾选 `Use Speaker Embedding Only (no ICL)`。
3. 输入要合成的文本，例如：`Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.`。
4. 点击 `Submit`，使用克隆的音色生成语音。

![音色克隆](../assets/using-models/using-audio-models/voice-clone.png)
