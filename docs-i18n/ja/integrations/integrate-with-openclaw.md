---
translation_status: translated
---

# OpenClaw との連携 { #integrating-with-openclaw }

[OpenClaw](https://openclaw.ai/) は、ローカルデバイスで動作し、Telegram、WhatsApp、Slack、Discord などのチャンネルで対話できる個人用 AI アシスタントです。

## モデルのデプロイ { #deploy-the-model }

GPUStack ドキュメントの[**モデルのデプロイ**](../user-guide/model-deployment-management.md#deploy-model)に従い、モデルをデプロイします。

## API アクセス情報 { #api-access-info }

1. GPUStack Web UI にログインします。
2. **Routes** ページに移動します。
3. 対象モデルの右側のメニューで **API Access Info** を選択します。

![](../assets/integrations/openclaw-01.png)

![](../assets/integrations/openclaw-02.png)

次の情報を記録します（API キーが未作成の場合は、画面の案内に従って作成してください）。

* **Access URL**
* **Model Name**
* **API Key**

## OpenClaw のインストール { #install-openclaw }

OpenClaw の公式ドキュメントに従ってインストールします。
[https://docs.openclaw.ai/install](https://docs.openclaw.ai/install)

## OpenClaw での GPUStack の設定 { #configure-gpustack-in-openclaw }

1. 対話式の設定ウィザードを起動します。

   ```bash
   openclaw onboard --install-daemon
   ```

2. **Model / Auth Provider** で **Custom Provider** を選択します。

   ![](../assets/integrations/openclaw-03.png)

3. 案内に従い、GPUStack の情報を入力します。

    * **API Base URL**：Access URL
    * **API Key**：API キー
    * **Model ID**：モデル名

   ![](../assets/integrations/openclaw-04.png)

以上の手順により、OpenClaw は GPUStack を使い、対応するモデルで推論を実行します。

## チャンネルの設定 { #configure-channels }

OpenClaw の公式ドキュメントに従い、使用する通信チャンネルを設定します。
[https://docs.openclaw.ai/channels](https://docs.openclaw.ai/channels)
