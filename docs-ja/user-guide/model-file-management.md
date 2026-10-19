# モデルファイルの管理 { #model-file-management }

GPUStack では、管理者がモデルファイルをダウンロードして管理できます。

## モデルファイルの追加 { #add-model-file }

現在、GPUStack は [Hugging Face](https://huggingface.co)、[ModelScope](https://modelscope.cn)、ローカルパスのモデルに対応しています。モデルファイルを追加するには、`Model Files` ページに移動します。

### Hugging Face モデルの追加 { #add-a-hugging-face-model }

1. `Add Model File` ボタンをクリックし、ドロップダウンから `Hugging Face` を選択します。
2. 左上の検索バーで、`Qwen/Qwen3-0.6B` などのモデル名を検索します。
3. *（オプション）* GGUF モデルでは、`Available Files` から必要な量子化形式を選択します。
4. モデルファイルをダウンロードする対象の Worker を選択します。
5. *（オプション）* GPUStack のキャッシュディレクトリではなく任意のパスにダウンロードするには、`Local Directory` を指定します。
6. `Save` ボタンをクリックします。

### ModelScope モデルの追加 { #add-a-modelscope-model }

1. `Add Model File` ボタンをクリックし、ドロップダウンから `ModelScope` を選択します。
2. 左上の検索バーで、`Qwen/Qwen3-0.6B` などのモデル名を検索します。
3. *（オプション）* GGUF モデルでは、`Available Files` から必要な量子化形式を選択します。
4. モデルファイルをダウンロードする対象の Worker を選択します。
5. *（オプション）* GPUStack のキャッシュディレクトリではなく任意のパスにダウンロードするには、`Local Directory` を指定します。
6. `Save` ボタンをクリックします。

### ローカルパスのモデルの追加 { #add-a-local-path-model }

Worker 上のローカルパスからモデルを追加できます。パスには、Hugging Face のモデルフォルダーなどのディレクトリ、または GGUF モデルなどのファイルを指定できます。

1. `Add Model File` ボタンをクリックし、ドロップダウンから `Local Path` を選択します。
2. `Model Path` を入力します。
3. 対象の Worker を選択します。
4. `Save` ボタンをクリックします。

## ダウンロードの再試行 { #retry-download }

モデルファイルのダウンロードに失敗した場合や、非常に低速な状態で停止している場合は、再試行できます。

1. `Model Files` ページに移動します。
2. モデルファイルを探します。
3. `Operations` 列の省略記号ボタンをクリックし、`Retry Download` を選択します。
4. GPUStack が、指定されたソースからモデルファイルのダウンロードを再試行します。

## モデルのデプロイ { #deploy-model }

モデルファイルからモデルをデプロイできます。モデルは特定の Worker に保存されているため、GPUStack は `worker-name` キーを使う Worker セレクターを追加し、適切にスケジュールされるようにします。

!!! tip

    ノード間でモデルをフェイルオーバーさせるには、クラスター内のすべてのノードが同じパスでモデルファイルにアクセスできることを確認し、Worker セレクターから `worker-name` ラベルを手動で削除してください。

1. `Model Files` ページに移動します。
2. デプロイするモデルファイルを探します。
3. `Operations` 列の `Deploy` ボタンをクリックします。
4. `Name`、`Backend`、`Backend Version`、`Replicas` などのデプロイパラメーターを確認または調整します。
5. `Save` ボタンをクリックします。

## モデルファイルの削除 { #delete-model-file }

1. `Model Files` ページに移動します。
2. 削除するモデルファイルを探します。
3. `Operations` 列の省略記号ボタンをクリックし、`Delete` を選択します。
4. *（オプション）* `Also delete the file from disk` にチェックを入れます。
5. `Delete` ボタンをクリックして確認します。
