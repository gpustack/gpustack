# ベンチマーク { #benchmarking }

GPUStack は、実行中のモデルインスタンスを対象にベンチマークを実行できます。Worker が専用のベンチマークコンテナイメージで実行し、結果とログは Worker に保存されます。

## 前提条件 { #prerequisites }

- モデルインスタンスが実行中で、正常に動作していること。

## ベンチマークの作成 { #create-benchmark }

1. `Benchmarks` ページに移動します。
2. `Add Benchmark` をクリックします。
3. インスタンスを選択し、構成を入力します。
4. `Save` ボタンをクリックします。

### ShareGPT ワークロードの長さ { #sharegpt-workload-lengths }

組み込みの ShareGPT データセットでは、入力トークンの最小長、最大長、または両方を必要に応じて設定できます。GPUStack は、トークナイザーで数えた長さが指定範囲内の初回プロンプトを保持します。出力トークン長は任意です。指定すると、保持されたすべてのプロンプトに同じ最大長を要求します。省略すると、各プロンプトはデータセット内の参照回答の長さを使用します。範囲に一致するプロンプトがない場合は、ベンチマークリクエストを送信する前に実行が失敗します。

## ベンチマーク結果の表示 { #view-benchmark-results }

1. `Benchmarks` ページに移動します。
2. 表示するベンチマークを探します。
3. ベンチマーク名をクリックして、結果と構成のスナップショットを表示します。

## ベンチマーク結果のエクスポート { #export-benchmark-results }

1. `Benchmarks` ページに移動します。
2. エクスポートするベンチマークを探します。
3. エクスポートする実行を 1 つ以上選択します。
4. `Export` ボタンをクリックして結果をダウンロードします。

## ベンチマークの編集 { #edit-benchmark }

1. `Benchmarks` ページに移動します。
2. 編集するベンチマークを探します。
3. `Operations` 列の `Edit` ボタンをクリックします。
4. 必要に応じて名前と説明を変更します。
5. `Save` ボタンをクリックします。

## ベンチマークの削除 { #delete-benchmark }

1. `Benchmarks` ページに移動します。
2. 削除するベンチマークを探します。
3. `Operations` 列の `Delete` ボタンをクリックします。
4. 削除を確認します。

## プライベート CA を使用する HTTPS Server でのベンチマーク { #benchmarks-on-an-https-server-with-a-private-ca }

ベンチマークコンテナは、Server URL を通じて GPUStack Server に進捗を報告するため、HTTPS デプロイでは Server の証明書を検証する必要があります。ベンチマークは別のイメージで実行され、CA を自身でインポートしないため、Worker が信頼する CA 証明書バンドルを渡します。これには `/usr/local/share/ca-certificates/` にマウントしたプライベート CA も含まれます（[追加の信頼する CA](../installation/installation.md#additional-trusted-cas)を参照）。Worker でその CA を利用できれば、追加の構成は必要ありません。

検証に失敗しても、モデルインスタンスへの負荷は平文 HTTP で生成するため、ベンチマーク自体には影響しません。ただし、実行完了まで進捗が 0% のままになり、Worker のログに原因が記録されます。解決するには、Server の CA を Worker にインポートしてください。
