# GPU サービスのストレージタイプ { #gpu-service-storage-types }

GPU サービスのストレージタイプでは、GPU サービスインスタンス用の共有ファイルシステムストレージバックエンド（NFS または S3）を定義できます。

管理者は複数の Kubernetes クラスターにまたがるストレージ定義を管理できます。定義したストレージタイプは、[GPU サービスのストレージ](gpuservice-storage.md)の作成時に選択できます。

内部的には、各ストレージタイプは対象クラスター上の Kubernetes `StorageClass` として実現されます。

## ストレージタイプの閲覧 { #browse-storage-types }

`GPU Service` > `Storage Types` ページに移動すると、すべてのストレージタイプとその詳細を閲覧できます。

![ストレージタイプ一覧のスクリーンショット](../assets/gpuservice/storage-types/list.png)

ストレージタイプは表示名または名前で絞り込めます。`Name` 列には、いずれかの値が表示されます。`Name` 列の並べ替えは、表示されているラベルに基づきます。

## ストレージタイプの追加 { #adding-a-storage-type }

`Storage Types` ページで `Add Storage Type` をクリックして、作成フォームを開きます。

![ストレージタイプ追加フォームのスクリーンショット](../assets/gpuservice/storage-types/add.png)

`Name` と、必要に応じて `Display Name` を入力し、`Type` を選択します。残りの項目は、選択したタイプによって異なります。

**NFS** — NFS Server がエクスポートする共有をマウントします。

- **NFS Server**（必須）：NFS Server のホスト名または IP アドレス。
- **共有パス**（必須）：NFS Server がエクスポートする共有パス。
- **サブディレクトリ**：ボリュームに使用する共有内のサブディレクトリ。空の場合は共有のルートを使用します。
- **マウント権限**：マウントしたディレクトリに適用する権限。`0777` などを指定します。既定値は `0` で、NFS Server 側に設定された権限を尊重します。
- **マウントオプション**：NFS のマウントオプションを個別のエントリーとして追加します。既定値は `hard`、`vers=4`、`rsize=1048576`、`wsize=1048576`、`noatime`、`nodiratime` です。

**S3** — S3 互換オブジェクトストレージのバケットをマウントします。

- **エンドポイント**（必須）：S3 互換ストレージのエンドポイント URL。
- **リージョン**：該当する場合は、S3 ストレージのリージョン。
- **バケット**：使用するバケット名。空の場合は、`gpu-instance-pv-{id}` という名前の既定のバケットを作成します。
- **アクセスキー**（必須）：ストレージの認証に使用するアクセスキー。
- **シークレットキー**（必須）：ストレージの認証に使用するシークレットキー。書き込み専用で、API 応答では返しません。
- **Insecure**：エンドポイントへの接続時に TLS/SSL 証明書の検証をスキップします。既定では無効です。
- **マウントオプション**：基盤となる [GeeseFS](https://github.com/yandex-cloud/geesefs) ドライバーのオプションを個別のエントリーとして追加します。既定値は、多くのワークロードに適した**大きなファイルへの集中的な書き込み**プリセットです。

S3 ボリュームは [GeeseFS](https://github.com/yandex-cloud/geesefs) でマウントされ、`Mount Options` は直接渡されます。アクセスパターンに合うプリセットを選択し、エンドポイントが Yandex Object Storage でない場合は、**Yandex 以外の S3 互換サービス**のオプションを追加してください。

??? note "一般的な GeeseFS マウントオプションのプリセット"

    | 用途 | マウントオプション |
    | --- | --- |
    | 大きなファイルへの集中的な書き込み（既定） | `--no-checksum` `--memory-limit=4000` `--max-flushers=32` `--max-parallel-parts=32` `--part-sizes=25` |
    | 大きなファイルの順次読み取り | `--read-ahead-large=200000` `--large-read-cutoff=10240` `--read-ahead-parallel=40000` `--memory-limit=8000` |
    | 小さなファイルのランダム読み取り | `--read-ahead-small=64` `--small-read-cutoff=64` `--read-ahead=1024` `--stat-cache-ttl=300s` `--entry-limit=200000` |
    | 書き込みの高可用性 | `--sdk-max-retries=10` `--read-retry-attempts=5` `--fsync-on-close` `--cache=/mnt/disk-cache` |
    | Yandex 以外の S3 互換サービス | `--list-type=2` `--no-specials` |

必須項目を入力した後、`Save` をクリックしてストレージタイプを作成します。

## ストレージタイプの編集 { #editing-a-storage-type }

作成後に変更できるのは表示名だけです。

!!! note

    ストレージタイプ構成のその他の項目の編集は、今後のリリースで対応する予定です。

## ストレージタイプの削除 { #deleting-a-storage-type }

ストレージタイプの `Delete` をクリックして確認すると、一覧から削除されます。

!!! warning

    ストレージタイプを削除する前に、それを使用して作成した[GPU サービスのストレージ](gpuservice-storage.md)をすべて削除してください。
