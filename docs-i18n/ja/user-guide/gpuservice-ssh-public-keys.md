# GPU サービスの SSH 公開鍵 { #gpu-service-ssh-public-keys }

GPU サービスの SSH 公開鍵を管理すると、GPU サービスインスタンスに安全に、パスワードなしでアクセスできます。

インスタンスの `SSH Access` 構成でこれらのキーを割り当てます。[GPU サービスインスタンス](gpuservice-instances.md)を参照してください。

## SSH 公開鍵の閲覧 { #browse-ssh-public-keys }

`GPU Service` > `SSH Public Keys` ページに移動すると、登録済みのキーをすべて閲覧できます。

![SSH 公開鍵一覧のスクリーンショット](../assets/gpuservice/ssh-public-keys/list.png)

SSH 公開鍵は表示名または名前で絞り込めます。`Name` 列には、いずれかの値が表示されます。`Name` 列の並べ替えは、表示されているラベルに基づきます。

## SSH 公開鍵の追加 { #adding-an-ssh-public-key }

`SSH Public Keys` ページで `Add SSH Public Key` をクリックして、作成フォームを開きます。

![SSH 公開鍵追加フォームのスクリーンショット](../assets/gpuservice/ssh-public-keys/add.png)

`Name` を入力し、`SSH Public Key` フィールドにキーを貼り付けます。各キーは `ssh-` で始まる必要があります。1 行につき 1 つのキーを入力すると、複数のキーを登録できます。`Save` をクリックして完了します。

!!! tip

    公開鍵は、RSA の場合は `cat ~/.ssh/id_rsa.pub`、Ed25519 の場合は `cat ~/.ssh/id_ed25519.pub` で表示できます。

## SSH 公開鍵の編集 { #editing-an-ssh-public-key }

キーの `Edit` をクリックして構成を開き、設定を更新して `Save` をクリックします。

![SSH 公開鍵編集フォームのスクリーンショット](../assets/gpuservice/ssh-public-keys/edit.png)

## SSH 公開鍵の削除 { #deleting-an-ssh-public-key }

キーの `Delete` をクリックして確認すると、一覧から削除されます。

!!! note

    GPU サービスインスタンスがすでに使用している SSH 公開鍵は、ここで削除してもそのインスタンスからは削除されません。キーの内容は Worker 側のクラスターに同期済みであり、個々のエントリーを後から取り除くことはできません。実行中のインスタンスからキーを失効させるには、そのインスタンスを開いて `Edit` をクリックし、`SSH Access` 構成でキーの選択を解除してください。
