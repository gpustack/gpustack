# ユーザー管理 { #user-management }

GPUStack には `Admin` と `User` の 2 つのユーザーロールがあります。

管理者は、クラスター、リソース、モデル、ユーザー、システム設定を管理できます。

ユーザーは、自分の API キーを管理し、モデル API にアクセスできます。

## 既定の管理者 { #default-admin }

GPUStack は初回起動時に、既定の管理者ユーザーを作成します。

初期パスワードは `<data-dir>/initial_admin_password` に保存されます。

既定の構成では、このファイルは Server コンテナ内の `/var/lib/gpustack/initial_admin_password` にあります。

`GPUStack` の起動時に `--bootstrap-password` フラグを使用すると、既定の管理者のパスワードを指定できます。

## パスワードの要件 { #password-requirements }

UI または API で設定するパスワードには、大文字、小文字、数字、および次の特殊文字のいずれかを、それぞれ 1 文字以上含める必要があります。

```
! @ # $ % ^ & * _ + .
```

UI では、さらにパスワードの長さを 6～64 文字に制限しています。

## ユーザーの作成 { #create-user }

1. `Users` ページに移動します。
2. `Add User` ボタンをクリックします。
3. `Name`、`Full Name`、`Password` を入力し、ユーザーの `Role` を選択します。
4. `Save` ボタンをクリックします。

## ユーザーの更新 { #update-user }

1. `Users` ページに移動します。
2. 編集するユーザーを探します。
3. `Operations` 列の `Edit` ボタンをクリックします。
4. 必要に応じて属性を更新します。
5. `Save` ボタンをクリックします。

## ユーザーの無効化 { #deactivate-user }

1. `Users` ページに移動します。
2. 無効化するユーザーを探します。
3. `Operations` 列の `Deactivate Account` ボタンをクリックします。

## ユーザーの有効化 { #activate-user }

1. `Users` ページに移動します。
2. 有効化するユーザーを探します。
3. `Operations` 列の `Activate Account` ボタンをクリックします。

## ユーザーの削除 { #delete-user }

1. `Users` ページに移動します。
2. 削除するユーザーを探します。
3. `Operations` 列の省略記号ボタンをクリックし、`Delete` を選択します。
4. 削除を確認します。

!!! note

    管理者ユーザーは UI から無効化または削除できません。
