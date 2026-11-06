---
translation_status: translated
---

# Cherry Studio との連携 { #integrate-with-cherry-studio }

Cherry Studio を GPUStack と連携させると、ローカルでホストする LLM、埋め込み、リランキングを利用できます。

## モデルのデプロイ { #deploying-models }

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、必要なモデルをデプロイします。モデルの例：

    - qwen3-instruct-2507
    - qwen2.5-vl-7b
    - bge-m3
    - bge-reranker-v2-m3

   ![GPUStack のモデル](../assets/integrations/integration-gpustack-models2.png)

2. モデルの操作メニューで `API Access Info` を開き、連携方法を確認します。

   ![API アクセス情報](../assets/integrations/integration-gpustack-api-access-info2.png)

## API キーの作成 { #create-an-api-key }

1. `Access Control` > `API Keys` ページで `New API Key` をクリックします。

2. 名前を入力し、`Save` をクリックします。

3. API キーをコピーし、後で使えるよう保存します。

## Cherry Studio への GPUStack の統合 { #integrating-gpustack-into-cherry-studio }

1. Cherry Studio の `Settings` → `Model Provider` で GPUStack を見つけて有効にし、次のように設定します。

    - `API Key`：前の手順でコピーした API キーを入力します。

    - `API Host`：`API Access Info` パネルの `Access URL` を指定します。

   ![Cherry Studio の GPUStack 設定](../assets/integrations/integration-cherrystudio-01.png)

2. GPUStack プロバイダー設定で「Manage」をクリックし、必要なモデルを有効にします。

   ![モデルの管理](../assets/integrations/integration-cherrystudio-02.png)

   ![モデルの有効化](../assets/integrations/integration-cherrystudio-03.png)

3. （任意）API をテストします。

   ![API のテスト](../assets/integrations/integration-cherrystudio-04.png)

設定後、Cherry Studio のホームページに戻ってモデルを使い始めます。

## LLM の利用 { #using-llms }

![LLM の利用](../assets/integrations/integration-cherrystudio-05.png)

## マルチモーダルモデルの利用 { #using-multimodal-models }

1. マルチモーダルモデルを選択します。

   ![マルチモーダルモデルの選択](../assets/integrations/integration-cherrystudio-13.png)

2. マルチモーダルの質問をします。

   ![マルチモーダルの質問応答](../assets/integrations/integration-cherrystudio-14.png)

   ![マルチモーダルの質問応答](../assets/integrations/integration-cherrystudio-15.png)

## 埋め込みとリランキングによるナレッジベースの質問応答の改善 { #use-embeddings-and-reranking-to-improve-knowledge-base-qa }

1. ナレッジベースの設定ページを開きます。

   ![ナレッジベースの設定](../assets/integrations/integration-cherrystudio-06.png)

2. ナレッジベースを追加します。

   ![ナレッジベースの追加](../assets/integrations/integration-cherrystudio-08.png)

3. コンテンツをナレッジベースに追加します（ここでは「Notes」を例に使用）。

   ![ナレッジベースへのコンテンツ追加](../assets/integrations/integration-cherrystudio-09.png)

   ![ナレッジベースの内容](../assets/integrations/integration-cherrystudio-10.png)

4. ホームページに戻り、ナレッジベースの質問応答を使用します。

   ![ナレッジベースの質問応答](../assets/integrations/integration-cherrystudio-11.png)

   ![ナレッジベースの質問応答例](../assets/integrations/integration-cherrystudio-12.png)
