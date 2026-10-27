# クラウド認証情報の管理 { #cloud-credential-management }

GPUStack はクラウド認証情報を管理し、外部クラウドプロバイダーに安全に接続できます。クラウド認証情報には、API アクセスに必要なプロバイダー情報、キー、オプションが含まれます。

## 対応プロバイダー { #supported-providers }

`DigitalOcean` と `SHUIHUA FUTURE` に対応しています。

## クラウド認証情報の作成 { #create-cloud-credential }

1. `Cloud Credentials` ページに移動します。
2. `Add Cloud Credential` ドロップダウンをクリックし、プロバイダーを選択します。
3. 次の情報を入力します。

   - `Name`：一意の認証情報名。
   - プロバイダーのシークレット。どちらのプロバイダーでも、必要な認証情報フィールドはこれだけです。
     - `DigitalOcean` — `Access Token`：DigitalOcean の `Applications & API` ページで生成した API トークン。トークンのスコープは `Full Access` にする必要があります。
     - `SHUIHUA FUTURE` — `API Key`：[Shuihua コンソール](https://hub.do.top/cn/signin)で作成する、`amp_live_` で始まるキー。
   - `Description`：クラウド認証情報の補足説明。

4. `Save` ボタンをクリックします。

## クラウド認証情報の更新 { #update-cloud-credential }

1. `Cloud Credentials` ページに移動します。
2. 編集する認証情報を探します。
3. `Edit` ボタンをクリックします。
4. 必要に応じて `Name`、`Access Token`、`Description` を更新します。
5. `Save` ボタンをクリックします。

## クラウド認証情報の削除 { #delete-cloud-credential }

1. `Cloud Credentials` ページに移動します。
2. 削除する認証情報を探します。
3. 操作列の省略記号ボタンをクリックし、`Delete` を選択します。
4. 削除を確認します。
