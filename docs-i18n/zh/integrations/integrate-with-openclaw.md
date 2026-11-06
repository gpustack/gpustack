---
translation_status: translated
---

# 与 OpenClaw 集成 { #integrating-with-openclaw }

[OpenClaw](https://openclaw.ai/) 是运行在本地设备上的个人 AI 助手，支持通过 Telegram、WhatsApp、Slack、Discord 等常用渠道进行交互。

## 部署模型 { #deploy-the-model }

请参考 GPUStack 文档中的[**模型部署**](../user-guide/model-deployment-management.md#deploy-model)部分完成模型部署。

## API 访问信息 { #api-access-info }

1. 登录 GPUStack Web 界面。
2. 打开 **Routes** 页面。
3. 在目标模型右侧菜单中选择 **API Access Info**。

![](../assets/integrations/openclaw-01.png)

![](../assets/integrations/openclaw-02.png)

记录以下信息；如果尚未创建 API 密钥，请按照页面提示创建：

* **Access URL**
* **Model Name**
* **API Key**

## 安装 OpenClaw { #install-openclaw }

按照 OpenClaw 官方文档完成安装：
[https://docs.openclaw.ai/install](https://docs.openclaw.ai/install)

## 在 OpenClaw 中配置 GPUStack { #configure-gpustack-in-openclaw }

1. 启动交互式配置向导：

   ```bash
   openclaw onboard --install-daemon
   ```

2. 在 **Model / Auth Provider** 选项中选择 **Custom Provider**。

   ![](../assets/integrations/openclaw-03.png)

3. 按提示填写 GPUStack 提供的信息：

    * **API Base URL**：Access URL
    * **API Key**：API Key
    * **Model ID**：Model Name

   ![](../assets/integrations/openclaw-04.png)

完成以上步骤后，OpenClaw 会通过 GPUStack 调用对应模型进行推理。

## 配置渠道 { #configure-channels }

按照 OpenClaw 官方文档配置所需通信渠道：
[https://docs.openclaw.ai/channels](https://docs.openclaw.ai/channels)
