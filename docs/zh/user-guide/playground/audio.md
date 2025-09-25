# 音频演示台

音频演示台是用于测试和试验 GPUStack 的文本转语音（TTS）与语音转文本（STT）API 的专用区域。它允许用户交互式地将文本转换为音频、将音频转换为文本，自定义参数，并查看代码示例以便无缝集成 API。

## 文本转语音

切换到 “Text to Speech” 选项卡以测试 TTS 模型。

### 文本输入

输入要转换的文本，然后点击 `Submit` 按钮生成相应语音。

![文本转语音](../../assets/playground/text-to-speech.png)

### 清空文本

点击 `Clear` 按钮重置文本输入并移除已生成的语音。

### 选择模型

在 Playground 界面右上角点击模型下拉框，选择 GPUStack 中可用的 TTS 模型。

### 自定义参数

自定义音色和音频输出格式。

!!! tip

    不同模型支持的音色可能有所不同。

### 查看代码

在尝试不同的输入文本和参数后，点击 `View Code` 按钮，查看使用相同输入调用 API 的方法。示例代码提供 `curl`、`Python` 和 `Node.js` 版本。

## 语音转文本

切换到 “Speech to Text” 选项卡以测试 STT 模型。

### 提供音频文件

你可以通过以下两种方式提供待转写的音频：

1. 上传音频文件。
2. 在线录音。

!!! note

    如果无法使用在线录音，可能是以下原因之一：

    1. 通过 HTTPS 或 `http://localhost` 访问时，需要在浏览器中启用麦克风权限。
    2. 通过 `http://{host IP}` 访问时，需要将该 URL 添加到浏览器的信任列表。

          **示例：**
          在 Chrome 中，进入 `chrome://flags/`，将 GPUStack 的 URL 添加到 "Insecure origins treated as secure"，并启用此选项。

![语音转文本](../../assets/playground/audio-permission.png)

![语音转文本](../../assets/playground/speech-to-text.png)

### 选择模型

在 Playground 界面右上角点击模型下拉框，选择 GPUStack 中可用的 STT 模型。

### 复制文本

复制模型生成的转写结果。

### 自定义参数

为你的音频文件选择合适的语言，以优化转写准确度。

### 查看代码

在尝试不同的音频文件和参数后，点击 `View Code` 按钮，查看使用相同输入调用 API 的方法。示例代码提供 `curl`、`Python` 和 `Node.js` 版本。