---
translation_status: translated
---

# MaxKB との連携 { #integrate-with-maxkb }

MaxKB を GPUStack と連携させると、ローカルにデプロイした **LLM、埋め込みモデル、リランキングモデル**を使って、ナレッジに基づく AI アシスタントを構築できます。

## モデルのデプロイ { #deploying-models }

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、必要なモデルをデプロイします。モデルの例：

* `qwen3.5-35b-a3b`

  ![](../assets/integrations/maxkb/01.png)
  ![](../assets/integrations/maxkb/02.png)

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/03.png)
  ![](../assets/integrations/maxkb/04.png)

* `qwen3-reranker-4b`

  ![](../assets/integrations/maxkb/05.png)
  ![](../assets/integrations/maxkb/06.png)

2. デプロイ後は **Playground** でモデルをテストできます。

![](../assets/integrations/maxkb/07.png)
![](../assets/integrations/maxkb/09.png)
![](../assets/integrations/maxkb/08.png)

## モデルのアクセス情報の取得 { #obtain-model-access-information }

1. GPUStack のサイドバーから **Routes** ページを開きます。

2. ルートの横の **More actions menu** をクリックし、**API Access Info** を選択します。

![](../assets/integrations/maxkb/10.png)
![](../assets/integrations/maxkb/11.png)

次の情報を記録します。

```
Base URL
Model Name
API Key
```

例：

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

    UI の案内に従って API キーを作成できます。

## MaxKB のデプロイ { #deploy-maxkb }

MaxKB は Docker でデプロイできます。

```bash
docker run -d \
  --name maxkb \
  --restart always \
  -p 8080:8080 \
  -v ~/.maxkb:/opt/maxkb \
  1panel/maxkb
```

デフォルトの認証情報：

```
admin / MaxKB@123..
```

![](../assets/integrations/maxkb/50.png)

初回ログイン後、案内に従ってパスワードを変更します。

## MaxKB への GPUStack の統合 { #integrating-gpustack-into-maxkb }

1. MaxKB UI の上部ナビゲーションで **Model** を開きます。

![](../assets/integrations/maxkb/51.png)

2. **Add Model** をクリックし、モデルを設定します。

![](../assets/integrations/maxkb/52.png)
![](../assets/integrations/maxkb/53.png)
![](../assets/integrations/maxkb/54.png)

モデル設定時の注意点：

* **Base Model**：GPUStack にデプロイしたモデル名と一致させます。
* **API URL**: `http://your-gpustack-url/v1`
* **API Key**：GPUStack で作成した API キー。

!!! note

    **Base Model を入力して Enter を押した後**に、`API URL` と `API Key` フィールドが表示されます。

3. 同じ方法で埋め込みモデルとリランキングモデルを追加します。

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/56.png)

* `qwen3-reranker-4b`

  **qwen3-reranker-4b** では **Generic Proxy** を有効にします。

  ![](../assets/integrations/maxkb/55.png)

  MaxKB が次のエンドポイントを使用するため、この設定が必要です。

  ```
  /v2/rerank
  ```

  ![](../assets/integrations/maxkb/57.png)

設定後、モデル一覧にモデルが表示されます。

![](../assets/integrations/maxkb/58.png)

## ナレッジベースの作成 { #create-a-knowledge-base }

1. **Knowledge** ページに移動します。

2. **Create** をクリックし、**Web Knowledge** を選択します。

![](../assets/integrations/maxkb/59.png)

3. ドキュメントの URL またはほかのデータソースを入力します。MaxKB が内容を自動でクロールし、解析します。

![](../assets/integrations/maxkb/60.png)

クロールが完了した状態：

![](../assets/integrations/maxkb/61.png)

## AI エージェントの作成 { #create-an-ai-agent }

1. **Agent** ページに移動します。

2. **Create** をクリックして新しいエージェントを作成します。

![](../assets/integrations/maxkb/62.png)
![](../assets/integrations/maxkb/63.png)

3. 次の項目を設定します。

* チャットモデル
* ナレッジベース
* 検索設定

4. **Publish** をクリックしてエージェントを有効にします。

![](../assets/integrations/maxkb/64.png)

## ナレッジベースとのチャット { #chat-with-the-knowledge-base }

チャット画面を開き、アシスタントと対話します。

![](../assets/integrations/maxkb/65.png)
![](../assets/integrations/maxkb/66.png)

接続したナレッジベースと GPUStack にデプロイしたモデルに基づき、アシスタントが質問に回答できるようになります。
