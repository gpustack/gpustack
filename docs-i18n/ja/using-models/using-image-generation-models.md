---
translation_status: translated
---

# 画像生成モデルの利用 { #using-image-generation-models }

GPUStack は**画像生成モデル**のデプロイと実行に対応しています。このガイドでは、**Z-Image-Turbo** テキスト画像生成モデルをデプロイし、GPUStack UI でテキストプロンプトから画像を生成する方法を説明します。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- 24 GB 以上の VRAM を持つ GPU（NVIDIA RTX 4090 など）。
- GPUStack がインストール済みで、稼働していること。未導入の場合は[クイックスタート](../quickstart.md)をご覧ください。
- モデルファイルを取得するために Hugging Face または ModelScope にアクセスできること。

## 手順 1：Z-Image-Turbo モデルのデプロイ { #step-1-deploy-the-z-image-turbo-model }

次の手順でモデルをデプロイします。

1. GPUStack UI の `Model Catalog` ページに移動します。
2. カテゴリーフィルターで `Image` を選択します。
3. `Z-Image-Turbo` のモデルカードをクリックし、デプロイページを開きます。
4. デフォルト設定のまま `Save` ボタンをクリックしてデプロイします。

![モデルのデプロイ](../assets/using-models/using-image-generation-models/deploy-model.png)

デプロイ後は、`Deployments` ページで状態を確認できます。成功したら、デプロイメントの三点アイコンから `Open in Playground` を選択し、プレイグラウンドでモデルを使い始めます。

![モデル一覧](../assets/using-models/using-image-generation-models/model-list.png)

## 手順 2：モデルによる画像生成 { #step-2-use-the-model-for-image-generation }

`Image` プレイグラウンドで次の操作を行います。

1. `Size` ドロップダウンで、希望する画像解像度を選択します。
2. 生成する画像を説明するプロンプトを入力します。`Random Prompt` ボタンでランダムなプロンプトを表示し、発想のヒントにすることもできます。
3. `Submit` ボタンをクリックして画像を作成します。

生成された画像が UI に表示されます。結果はランダム性、シード値、プロンプトの内容によって変わります。

![生成結果](../assets/using-models/using-image-generation-models/image-playground.png)

## まとめ { #conclusion }

この設定により、Z-Image-Turbo モデルでテキストプロンプトから独自の魅力的な画像を生成できます。プロンプトや設定を変えて、さまざまな画像を作成してください。
