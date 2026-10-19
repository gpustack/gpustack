# GPU サービスのストレージ { #gpu-service-storage }

[GPU サービスのストレージタイプ](gpuservice-storage-types.md)に基づいて、永続ストレージを割り当て、複数の GPU サービスインスタンスにマウントできます。

インスタンス間で共有する必要があるデータ、モデル、その他のファイルに役立ちます。

GPUStack は複数の Kubernetes クラスターを管理し、クラスター間で GPU サービスのストレージを共有するための統一されたインターフェースを提供します。

内部的には、各ストレージ項目は対象クラスターの Kubernetes `PersistentVolumeClaim` として実現されます。

## ストレージの閲覧 { #browse-storage }

`GPU Service` > `Storage` ページに移動すると、すべてのストレージとその詳細を閲覧できます。

![ストレージ一覧のスクリーンショット](../assets/gpuservice/storage/list.png)

ストレージは表示名または名前で絞り込めます。`Name` 列には、いずれかの値が表示されます。`Name` 列の並べ替えは、表示されているラベルに基づきます。

## ストレージの追加 { #adding-storage }

`Storage` ページで `Add Storage` をクリックして、作成フォームを開きます。

![ストレージ追加フォームのスクリーンショット](../assets/gpuservice/storage/add.png)

`Name` を入力し、[GPU サービスのストレージタイプ](gpuservice-storage-types.md)ページで定義した `Storage Type` を選択して、`Capacity (GB)` を設定します。続いて `Save` をクリックします。

## ストレージの編集 { #editing-storage }

作成後に変更できるのは表示名だけです。

!!! note

    ストレージ構成のその他の項目の編集は、今後のリリースで対応する予定です。

## ストレージの削除 { #deleting-storage }

ストレージ項目の `Delete` をクリックして確認すると、一覧から削除されます。

!!! warning

    ストレージが GPU サービスインスタンスにまだ接続されている場合、クラスター側の `PersistentVolumeClaim` はすぐには削除されません。削除は遅延ポリシーに従うため、関連するインスタンスが削除されるまで `PersistentVolumeClaim` は保持され、その後にクリーンアップされます。
