# ベンチマーク { #benchmarking }

GPUStack は、実行中のモデルインスタンスを対象にベンチマークを実行できます。Worker が専用のベンチマークコンテナイメージで実行し、結果とログは Worker に保存されます。

## 前提条件 { #prerequisites }

- モデルインスタンスが実行中で、正常に動作していること。

### ルートベンチマークのゲートウェイアドレス { #route-benchmark-gateway-address }

Server は実行する Worker が所属するクラスターの `server_url`、次に Server の `server_external_url` を優先します（ベース URL に `/v1` は含めません）。それ以外の場合、Server の `gateway_mode` により Worker 側の解決ポリシーが決まります。

- `embedded`：組み込み Worker はローカル管理接続のホストと Higress のポートを使用します。独立した Worker には、明示的に設定した `advertise_address` または信頼されたリクエストの入口がフォールバックとして必要です。
- `disabled`：Worker の Server 接続 URL をネイティブ API ルーティングに再利用します。測定には Server プロキシのオーバーヘッドが含まれます。
- `external` / `incluster`：Server の `trusted_hosts` で明示的に許可されている場合のみ、作成リクエストのオリジンを使用します。

リクエストのオリジンへのフォールバックではワイルドカードを使用できません。ホスト名のみの指定では標準の HTTP/HTTPS ポートだけが許可されます。カスタムポートは `gateway.example:8443` のように明示してください。前段のプロキシで URL のパスプレフィックスを取り除いてから Higress に転送する必要があります。

Worker は明示された入口を使用するか、ポリシーとローカル構成からアドレスを解決します。ベンチマークの開始前に最終 URL を報告し、スナップショットに固定します。変更するにはベンチマークを再作成してください。進捗報告には引き続き Worker の管理アドレスを使用します。

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

進捗報告とルート推論は Worker の TLS ポリシーに従います。Worker で `GPUSTACK_INSECURE_TLS` または `insecure_tls` 構成オプション（`--insecure-tls`）が有効でない限り、証明書検証は有効です。ベンチマーク専用の TLS 設定はありません。

いずれかの入口が HTTPS を使用し、検証が有効な場合、Worker は信頼する CA バンドルをファイル内容としてベンチマークコンテナに注入します。Worker にインポートしたプライベート CA も含まれます（[追加の信頼する CA](../installation/installation.md#additional-trusted-cas)を参照）。進捗クライアントは `--progress-ca-cert` を使用し、HTTPS 推論は `SSL_CERT_FILE` を通じて注入されたバンドルを使用してバックエンドの検証を有効にします。管理アドレスが HTTP でも、HTTPS ゲートウェイ用の CA は注入されます。

Worker が明示的に検証をスキップする場合、ベンチマークは両方の通信で検証をスキップし、CA バンドルを注入しません。検証が必要な HTTPS 推論でバンドルを読み取れない場合、別の信頼ストアを使用せず、起動を拒否します。進捗報告のみの場合は引き続きベストエフォートで動作し、報告の失敗をログに記録しても測定結果は破棄しません。
