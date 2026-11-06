---
translation_status: translated
---

# RAGFlow との連携 { #integrate-with-ragflow }

RAGFlow を GPUStack と連携させると、ローカルにデプロイした LLM、ビジョンモデル、埋め込み、リランキング、音声認識、音声合成を利用できます。RAGFlow は GPUStack で実行中のモデルを自動で検出するため、一度接続すれば、追加するモデルを選ぶだけです。

!!! note

    このガイドには、GPUStack 設定カードが追加された RAGFlow v1.0.0-rc3 より後のバージョンが必要です。

## モデルのデプロイ { #deploying-models }

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、必要なモデルをデプロイします。RAGFlow が検出できるのは実行中でルートを持つモデルのみのため、`Enable Model Route` は選択したままにしてください。モデルの例：

    | モデル | RAGFlow での種類 |
    | --- | --- |
    | `qwen3-4b` | Chat |
    | `qwen2.5-vl-3b-instruct` | Chat + Vision |
    | `bge-m3` | Embedding |
    | `bge-reranker-v2-m3` | Rerank |
    | `faster-whisper-small` | Speech to text |
    | `cosyvoice2-0.5b` | Text to speech |

2. 各モデルのレプリカの準備が完了するまで待ちます：

    ![GPUStack のデプロイ一覧](../assets/integrations/integration-ragflow-gpustack-deployments.png)

## API キーの作成 { #create-an-api-key }

1. `Access Control` > `API Keys` ページで `Add API Key` をクリックします。

2. 名前を入力し、`Access Permissions` で `Model Access` を選択したまま `All models` にします。`Expiration` の既定値は `1 month` のため、RAGFlow でこのキーを使い続ける場合はより長い期間を選択してください。その後 `Save` をクリックします。

    ![GPUStack の API キー](../assets/integrations/integration-ragflow-gpustack-api-key.png)

3. API キーをコピーして保存します。以降、`API Keys` ページにはマスクされた値のみが表示されます。

## RAGFlow への GPUStack の統合 { #integrating-gpustack-into-ragflow }

1. RAGFlow UI を開き、右上のアバターをクリックして `Model providers` を開き、`Available models` から `GPUStack` を選択します。

2. 次の項目を入力し、`Connect` をクリックします：

    - Instance name：RAGFlow でこの GPUStack サーバーを識別する任意の名前（例：`gpustack`）。
    - GPUStack server URL：`http://your-gpustack-url`（`/v1` パスは付けません）。カスタムポートを使用する場合はポートも含めます。`localhost` は RAGFlow コンテナを指すため使用しないでください。GPUStack と RAGFlow が同じホストで動作している場合は `http://host.docker.internal:<port>` を使用できます。
    - API key：先ほどコピーした API キー。

3. RAGFlow は実行中のモデルを種類ごとにグループ化し、コンテキスト長とともに一覧表示します。追加しないモデルのチェックを外します。GPUStack はモデルが画像を受け付けるかどうかを報告しないため、`qwen2.5-vl-3b-instruct` などのビジョンモデルは `Chat + Vision` に設定してください：

    ![RAGFlow で GPUStack に接続](../assets/integrations/integration-ragflow-connect-gpustack.png)

    接続に失敗すると、RAGFlow はフォームの下に理由を表示します：

    - `GPUStack rejected this API key.`：`Access Control` > `API Keys` で API キーを確認してください。
    - `Cannot reach this server from RAGFlow.`：RAGFlow コンテナ内から URL とポートを確認してください。たとえば、RAGFlow の `docker` ディレクトリで次のコマンドを実行すると、`"ok"` が返るはずです：

        ```bash
        docker compose exec ragflow-cpu curl http://your-gpustack-url/healthz
        ```

4. 一覧の下部にある `Add 6 models` をクリックします（数字は選択したモデル数に応じて変わります）。追加したモデルがインスタンスに表示されます：

    ![RAGFlow でモデルを追加](../assets/integrations/integration-ragflow-add-model.png)

    後から GPUStack にモデルを追加でデプロイした場合は、インスタンスの `Refresh` をクリックすると検出して追加できます。

5. `Set default models` をクリックし、種類ごとにモデルを選択します。選択はすぐに保存されます：

    ![RAGFlow で既定のモデルを設定](../assets/integrations/integration-ragflow-set-default-models.png)

## モデルの利用 { #using-the-models }

これで `Dataset` と `Chat` でモデルを利用できます。簡単な例を示します：

1. `Dataset` でデータセットを作成します。既定の埋め込みモデルはすでに選択されています：

    ![RAGFlow でデータセットを作成](../assets/integrations/integration-ragflow-create-dataset.png)

2. `Add file` > `Upload file` をクリックし、`Parse on creation` をオンにして、ファイルを選択し `Save` をクリックします。解析が完了すると、ファイルにチャンク数が表示されます：

    ![RAGFlow でファイルをアップロード](../assets/integrations/integration-ragflow-upload-file.png)

3. `Retrieval testing` を開き、`Rerank model` を `bge-reranker-v2-m3` に設定し、質問を入力して `Run` をクリックします：

    ![RAGFlow でリランクモデルを設定](../assets/integrations/integration-ragflow-set-rerank-model.png)

4. `Chat` でチャットを作成して開きます。右上の設定アイコンをクリックし、`Datasets` でデータセットを選択し、`Model` のチャットモデルを確認して `Save` をクリックします：

    ![RAGFlow でチャットを設定](../assets/integrations/integration-ragflow-create-assistant.png)

    `Retrieval testing` で選択したリランクモデルはチャットには適用されません。チャットの検索結果をリランクするには、同じ設定で `Advanced settings` > `Naive` を展開し、`Rerank model` を `bge-reranker-v2-m3` に設定して `Save` をクリックします。

5. メッセージを送信すると、モデルと対話してデータセットを照会できます：

    ![RAGFlow でモデルと対話](../assets/integrations/integration-ragflow-chat-with-model.png)

6. 回答を読み上げるには、`Advanced settings` で `Text to speech` をオンにして `Save` をクリックします。回答にカーソルを合わせてスピーカーアイコンをクリックします。再生中は一時停止アイコンに変わります：

    ![RAGFlow で回答を読み上げ](../assets/integrations/integration-ragflow-read-aloud.png)

7. 音声入力を使うには、メッセージ欄のマイクアイコンをクリックして話し、もう一度アイコンをクリックして停止します。RAGFlow は既定の音声認識モデルで録音を文字起こしし、メッセージとして送信します。

8. 画像について対話するには、チャット設定で `Model` を `qwen2.5-vl-3b-instruct` に切り替え、回答に無関係なドキュメントの引用が混ざらないよう `Datasets` をクリアします。`Save` をクリックし、新しい会話を開始して、クリップアイコンで画像を添付し、質問を送信します：

    ![RAGFlow でビジョンモデルと対話](../assets/integrations/integration-ragflow-chat-with-vlm.png)
