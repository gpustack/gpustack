# 画像 Playground { #image-playground }

画像 Playground は、GPUStack の画像生成 API をテストして試すための専用の場所です。さまざまなモデルの能力を対話的に調べ、パラメーターを調整し、API との連携に使うコード例を確認できます。

## 画像の生成 { #generate-image }

### プロンプト { #prompt }

プロンプトを入力するかランダムに生成し、`Submit` ボタンをクリックして画像を生成します。

![画像生成](../../assets/playground/create-image.png)

## 画像の編集 { #edit-image }

画像をアップロードし、変更する領域を塗って指定します。続いてプロンプトを入力し、`Submit` をクリックします。領域を塗らなかった場合は、画像全体が変更されます。

![画像編集](../../assets/playground/image-edit.png)

### マスクの保存 { #save-mask }

`Save Mask` をクリックすると、塗った領域を別の画像として保存できます。

### 画像のダウンロード { #download-image }

`Download Image` をクリックすると、編集した画像を保存できます。

## モデルの選択 { #select-model }

Playground UI の右上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能なモデルを選択できます。

## パラメーター { #parameters }

| パラメーター | 既定値 | 説明 |
| --- | --- | --- |
| `Size` | `512x512` | 生成する画像のサイズ。形式は 'widthxheight' です。 |
| `Seed` | (empty) | 乱数シード。 |

## コードの表示 { #view-code }

プロンプトとパラメーターを試した後、`View Code` ボタンをクリックすると、同じ入力で API を呼び出す方法を確認できます。`curl`、`Python`、`Node.js` のコード例を提供しています。
