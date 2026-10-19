---
translation_status: translated
---

# 音频 Playground { #audio-playground }

音频 Playground 是用于测试和探索 GPUStack 文本转语音（TTS）及语音转文本（STT）API 的专用空间。它支持以交互方式将文本转换为音频、将音频转换为文本，自定义参数，并查看便于集成 API 的代码示例。

## 文本转语音 { #text-to-speech }

切换到 "Text to Speech" 选项卡，测试 TTS 模型。

### 输入文本 { #text-input }

输入要转换的文本，然后单击 `Submit` 按钮生成对应语音。

![文本转语音](../../assets/playground/text-to-speech.png)

### 选择模型 { #select-model }

单击 Playground 界面右上角的模型下拉菜单，选择 GPUStack 中可用的 TTS 模型。

### 自定义参数 { #customize-parameters }

自定义音频输出的声音和格式。

!!! tip

    不同模型支持的声音可能不同。

### 查看代码 { #view-code }

尝试不同输入文本和参数后，单击 `View Code` 按钮，查看如何使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。

## 语音转文本 { #speech-to-text }

切换到 "Speech to Text" 选项卡，测试 STT 模型。

### 提供音频文件 { #provide-audio-file }

可以通过两种方式提供需要转录的音频：

1. **上传音频文件**
2. **在线录音**

!!! note

    如果无法在线录音，可能由以下原因导致：

    1. 通过 HTTPS 或 `http://localhost` 访问时，必须在浏览器中启用麦克风权限。
    2. 通过 `http://{host IP}` 访问时，必须将 URL 加入浏览器的可信列表。

          **示例：**
          在 Chrome 中进入 `chrome://flags/`，将 GPUStack URL 添加到 "Insecure origins treated as secure"，并启用此选项。

![语音转文本](../../assets/playground/audio-permission.png)

![语音转文本](../../assets/playground/speech-to-text.png)

### 选择模型 { #select-model_1 }

单击 Playground 界面右上角的模型下拉菜单，选择 GPUStack 中可用的 STT 模型。

### 复制文本 { #copy-text }

复制模型生成的转录结果。

### 自定义参数 { #customize-parameters_1 }

为音频文件选择合适的语言，以提高转录准确率。

### 查看代码 { #view-code_1 }

尝试不同音频文件和参数后，单击 `View Code` 按钮，查看如何使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。
