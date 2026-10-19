---
translation_status: translated
---

# n8n との連携 { #integrating-with-n8n }

[n8n](https://n8n.io/) は、fair-code モデルで配布される拡張可能なワークフロー自動化ツールです。API を持つアプリケーションを接続し、タスクを操作・自動化できます。高度な AI ノードにより、大規模言語モデルを自動化ワークフローに簡単に組み込めます。

## モデルのデプロイ { #deploy-the-model }

GPUStack ドキュメントの**[モデルのデプロイ](../user-guide/model-deployment-management.md#deploy-model)**に従い、モデルをデプロイします。

## API アクセス情報 { #api-access-info }

1. GPUStack Web UI にログインします。
2. **Routes** ページに移動します。
3. 対象モデルの右側のメニューで **API Access Info** を選択します。

![](../assets/integrations/n8n-01.png)
![](../assets/integrations/n8n-02.png)

次の情報を記録します（API キーが未作成の場合は、画面の案内に従って作成してください）。

* **Access URL**
* **API Key**

## n8n のインストール { #install-n8n }

n8n の公式ドキュメントに従って自己ホスト環境にインストールするか、n8n Cloud を直接使用します。
[https://docs.n8n.io/hosting/](https://docs.n8n.io/hosting/)

## n8n での GPUStack の統合 { #integrating-gpustack-in-n8n }

GPUStack は OpenAI 互換 API を提供するため、n8n の OpenAI ノードで直接設定できます。

1. n8n に **Credential** を追加します。

   ![](../assets/integrations/n8n-03.png)
   ![](../assets/integrations/n8n-04.png)
   ![](../assets/integrations/n8n-05.png)

   追加に成功すると、次のように表示されます。

   ![](../assets/integrations/n8n-05-02.png)

2. GPUStack の Credential を使用します。

   ![](../assets/integrations/n8n-06.png)
   ![](../assets/integrations/n8n-07.png)

## 高度な AI ワークフローの構築 { #build-advanced-ai-workflows }

さまざまなアプリのノードと AI を組み合わせた強力な自動処理については、n8n の公式ドキュメントをご覧ください。
[https://docs.n8n.io/advanced-ai/](https://docs.n8n.io/advanced-ai/)
