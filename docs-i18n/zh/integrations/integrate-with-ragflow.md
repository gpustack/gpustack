---
translation_status: translated
---

# 与 RAGFlow 集成 { #integrate-with-ragflow }

RAGFlow 可接入 GPUStack，使用本地部署的 LLM、视觉模型、向量嵌入、重排序、语音转文本和文本转语音能力。RAGFlow 会自动发现 GPUStack 中正在运行的模型，只需连接一次，再选择要添加的模型即可。

!!! note

    本指南需要 RAGFlow v1.0.0-rc3 之后的版本，该版本新增了 GPUStack 配置卡片。

## 部署模型 { #deploying-models }

1. 在 GPUStack 界面中打开 `Deployments` 页面，点击 `Deploy Model` 部署所需模型。请保持勾选 `Enable Model Route`，因为 RAGFlow 只能发现正在运行且已配置路由的模型。模型示例如下：

    | 模型 | 在 RAGFlow 中的类型 |
    | --- | --- |
    | `qwen3-4b` | Chat |
    | `qwen2.5-vl-3b-instruct` | Chat + Vision |
    | `bge-m3` | Embedding |
    | `bge-reranker-v2-m3` | Rerank |
    | `faster-whisper-small` | Speech to text |
    | `cosyvoice2-0.5b` | Text to speech |

2. 等待每个模型的副本都就绪：

    ![GPUStack 部署列表](../assets/integrations/integration-ragflow-gpustack-deployments.png)

## 创建 API 密钥 { #create-an-api-key }

1. 打开 `Access Control` > `API Keys` 页面，点击 `Add API Key`。

2. 填写名称，在 `Access Permissions` 下保持勾选 `Model Access` 并选择 `All models`。`Expiration` 默认为 `1 month`，如果 RAGFlow 需要长期使用该密钥，请选择更长的有效期。然后点击 `Save`。

    ![GPUStack API 密钥](../assets/integrations/integration-ragflow-gpustack-api-key.png)

3. 复制 API 密钥并保存，供后续使用。之后 `API Keys` 页面只会显示脱敏后的值。

## 在 RAGFlow 中接入 GPUStack { #integrating-gpustack-into-ragflow }

1. 打开 RAGFlow 界面，点击右上角头像进入 `Model providers`，然后在 `Available models` 下选择 `GPUStack`。

2. 填写以下字段，然后点击 `Connect`：

    - Instance name：用于在 RAGFlow 中标识该 GPUStack 服务器的任意名称，例如 `gpustack`。
    - GPUStack server URL：`http://your-gpustack-url`，不要附加 `/v1` 路径。如使用自定义端口，请包含端口号。不要使用 `localhost`，因为它指向 RAGFlow 容器。如果 GPUStack 与 RAGFlow 运行在同一台主机上，可以使用 `http://host.docker.internal:<port>`。
    - API key：填写前面复制的 API 密钥。

3. RAGFlow 会按类型分组列出正在运行的模型及其上下文长度。取消勾选不需要添加的模型。GPUStack 不会报告模型是否支持图片输入，因此请将 `qwen2.5-vl-3b-instruct` 这类视觉模型设为 `Chat + Vision`：

    ![在 RAGFlow 中连接 GPUStack](../assets/integrations/integration-ragflow-connect-gpustack.png)

    如果连接失败，RAGFlow 会在表单下方显示原因：

    - `GPUStack rejected this API key.`：请在 `Access Control` > `API Keys` 中检查 API 密钥。
    - `Cannot reach this server from RAGFlow.`：请在 RAGFlow 容器内检查 URL 和端口。例如，在 RAGFlow 的 `docker` 目录下运行以下命令，应返回 `"ok"`：

        ```bash
        docker compose exec ragflow-cpu curl http://your-gpustack-url/healthz
        ```

4. 点击列表底部的 `Add 6 models`（数字随所选模型数量变化）。添加的模型会显示在实例中：

    ![在 RAGFlow 中添加模型](../assets/integrations/integration-ragflow-add-model.png)

    之后如果在 GPUStack 中部署了新模型，点击实例上的 `Refresh` 即可发现并添加。

5. 点击 `Set default models`，为每种类型选择模型。每项选择都会立即保存：

    ![在 RAGFlow 中设置默认模型](../assets/integrations/integration-ragflow-set-default-models.png)

## 使用模型 { #using-the-models }

现在可以在 `Dataset` 和 `Chat` 中使用模型。以下是一个简单示例：

1. 进入 `Dataset` 创建数据集。默认嵌入模型已自动选中：

    ![在 RAGFlow 中创建数据集](../assets/integrations/integration-ragflow-create-dataset.png)

2. 点击 `Add file` > `Upload file`，打开 `Parse on creation`，选择文件并点击 `Save`。解析完成后，文件会显示切片数量：

    ![在 RAGFlow 中上传文件](../assets/integrations/integration-ragflow-upload-file.png)

3. 打开 `Retrieval testing`，将 `Rerank model` 设为 `bge-reranker-v2-m3`，输入问题并点击 `Run`：

    ![在 RAGFlow 中设置重排序模型](../assets/integrations/integration-ragflow-set-rerank-model.png)

4. 在 `Chat` 中创建并打开一个聊天。点击右上角的设置图标，在 `Datasets` 下选择数据集，确认 `Model` 中的对话模型，然后点击 `Save`：

    ![在 RAGFlow 中配置聊天](../assets/integrations/integration-ragflow-create-assistant.png)

    `Retrieval testing` 中选择的重排序模型不会应用到聊天。如需为聊天的检索启用重排序，请在同一设置中展开 `Advanced settings` > `Naive`，将 `Rerank model` 设为 `bge-reranker-v2-m3` 并点击 `Save`。

5. 发送消息，即可与模型交互并查询数据集：

    ![在 RAGFlow 中与模型对话](../assets/integrations/integration-ragflow-chat-with-model.png)

6. 如需朗读回答，请在 `Advanced settings` 下打开 `Text to speech` 并点击 `Save`。将鼠标悬停在回答上并点击扬声器图标，播放时该图标会变为暂停图标：

    ![在 RAGFlow 中朗读回答](../assets/integrations/integration-ragflow-read-aloud.png)

7. 如需语音输入，请点击消息框中的麦克风图标，说话后再次点击该图标结束录音。RAGFlow 会使用默认的语音转文本模型转写录音，并将其作为你的消息发送。

8. 如需针对图片对话，请在聊天设置中将 `Model` 切换为 `qwen2.5-vl-3b-instruct` 并清空 `Datasets`，避免回答中混入无关文档的引用。点击 `Save`，新建对话，通过回形针图标附加图片并发送问题：

    ![在 RAGFlow 中使用视觉模型对话](../assets/integrations/integration-ragflow-chat-with-vlm.png)
