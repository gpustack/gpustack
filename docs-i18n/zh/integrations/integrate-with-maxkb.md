---
translation_status: translated
---

# 与 MaxKB 集成 { #integrate-with-maxkb }

MaxKB 可接入 GPUStack，使用本地部署的 **LLM、向量嵌入模型和重排序模型**构建基于知识库的 AI 助手。

## 部署模型 { #deploying-models }

1. 在 GPUStack 界面中打开 `Deployments` 页面，点击 `Deploy Model` 部署所需模型，例如：

* `qwen3.5-35b-a3b`

  ![](../assets/integrations/maxkb/01.png)
  ![](../assets/integrations/maxkb/02.png)

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/03.png)
  ![](../assets/integrations/maxkb/04.png)

* `qwen3-reranker-4b`

  ![](../assets/integrations/maxkb/05.png)
  ![](../assets/integrations/maxkb/06.png)

2. 部署后，可在 **Playground** 中测试模型。

![](../assets/integrations/maxkb/07.png)
![](../assets/integrations/maxkb/09.png)
![](../assets/integrations/maxkb/08.png)

## 获取模型访问信息 { #obtain-model-access-information }

1. 在 GPUStack 侧边栏打开 **Routes** 页面。
2. 点击路由旁的 **More actions menu**，选择 **API Access Info**。

![](../assets/integrations/maxkb/10.png)
![](../assets/integrations/maxkb/11.png)

记录以下信息：

```
Base URL
Model Name
API Key
```

示例：

```
Base URL: http://your-gpustack-url/v1

Model Name:
qwen3.5-35b-a3b
qwen3-embedding-4b
qwen3-reranker-4b

API Key:
gpustack_xxxxxxxxxxxxx
```

!!! note

    可按照界面提示创建 API 密钥。

## 部署 MaxKB { #deploy-maxkb }

可以使用 Docker 部署 MaxKB：

```bash
docker run -d \
  --name maxkb \
  --restart always \
  -p 8080:8080 \
  -v ~/.maxkb:/opt/maxkb \
  1panel/maxkb
```

默认凭据：

```
admin / MaxKB@123..
```

![](../assets/integrations/maxkb/50.png)

首次登录后，请按照提示修改密码。

## 在 MaxKB 中接入 GPUStack { #integrating-gpustack-into-maxkb }

1. 在 MaxKB 顶部导航栏打开 **Model**。

![](../assets/integrations/maxkb/51.png)

2. 点击 **Add Model**，配置模型。

![](../assets/integrations/maxkb/52.png)
![](../assets/integrations/maxkb/53.png)
![](../assets/integrations/maxkb/54.png)

模型配置如下：

* **Base Model**：必须与 GPUStack 中部署的模型名称一致。
* **API URL**：`http://your-gpustack-url/v1`
* **API Key**：在 GPUStack 中创建的 API 密钥。

!!! note

    **输入 Base Model 并按 Enter 后**，才会显示 `API URL` 和 `API Key` 字段。

3. 使用同样的方法添加向量嵌入模型和重排序模型：

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/56.png)

* `qwen3-reranker-4b`

  对 **qwen3-reranker-4b** 启用 **Generic Proxy**。

  ![](../assets/integrations/maxkb/55.png)

  这是因为 MaxKB 使用以下端点：

  ```
  /v2/rerank
  ```

  ![](../assets/integrations/maxkb/57.png)

配置完成后，模型应出现在模型列表中。

![](../assets/integrations/maxkb/58.png)

## 创建知识库 { #create-a-knowledge-base }

1. 打开 **Knowledge** 页面。
2. 点击 **Create**，选择 **Web Knowledge**。

![](../assets/integrations/maxkb/59.png)

3. 输入文档 URL 或其他数据源，MaxKB 会自动抓取并解析内容。

![](../assets/integrations/maxkb/60.png)

抓取完成后：

![](../assets/integrations/maxkb/61.png)

## 创建 AI 智能体 { #create-an-ai-agent }

1. 打开 **Agent** 页面。
2. 点击 **Create**，创建新智能体。

![](../assets/integrations/maxkb/62.png)
![](../assets/integrations/maxkb/63.png)

3. 为智能体配置：

* 对话模型
* 知识库
* 检索设置

4. 点击 **Publish** 发布并启用智能体。

![](../assets/integrations/maxkb/64.png)

## 与知识库对话 { #chat-with-the-knowledge-base }

打开聊天界面，开始与助手交互。

![](../assets/integrations/maxkb/65.png)
![](../assets/integrations/maxkb/66.png)

助手现在可以根据接入的知识库和 GPUStack 上部署的模型回答问题。
