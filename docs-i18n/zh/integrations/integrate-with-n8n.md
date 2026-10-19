---
translation_status: translated
---

# 与 n8n 集成 { #integrating-with-n8n }

[n8n](https://n8n.io/) 是采用 fair-code 模式分发、可扩展的工作流自动化工具。它可以连接任何具有 API 的应用，以处理和自动执行任务；借助 AI 节点，也可以将大语言模型集成到自动化工作流中。

## 部署模型 { #deploy-the-model }

请参考 GPUStack 文档中的**[模型部署](../user-guide/model-deployment-management.md#deploy-model)**部分完成模型部署。

## API 访问信息 { #api-access-info }

1. 登录 GPUStack Web 界面。
2. 打开 **Routes** 页面。
3. 在目标模型右侧菜单中选择 **API Access Info**。

![](../assets/integrations/n8n-01.png)
![](../assets/integrations/n8n-02.png)

记录以下信息；如果尚未创建 API 密钥，请按照页面提示创建：

* **Access URL**
* **API Key**

## 安装 n8n { #install-n8n }

按照 n8n 官方文档完成自托管安装，或直接使用 n8n Cloud：
[https://docs.n8n.io/hosting/](https://docs.n8n.io/hosting/)

## 在 n8n 中接入 GPUStack { #integrating-gpustack-in-n8n }

GPUStack 提供 OpenAI 兼容 API，可以直接使用 n8n 的 OpenAI 节点进行配置：

1. 在 n8n 中添加 **Credential**。

   ![](../assets/integrations/n8n-03.png)
   ![](../assets/integrations/n8n-04.png)
   ![](../assets/integrations/n8n-05.png)

   添加成功后的效果如下：

   ![](../assets/integrations/n8n-05-02.png)

2. 使用 GPUStack Credential。

   ![](../assets/integrations/n8n-06.png)
   ![](../assets/integrations/n8n-07.png)

## 构建高级 AI 工作流 { #build-advanced-ai-workflows }

参考 n8n 官方文档，将不同应用节点与 AI 组合，构建自动化流程：
[https://docs.n8n.io/advanced-ai/](https://docs.n8n.io/advanced-ai/)
