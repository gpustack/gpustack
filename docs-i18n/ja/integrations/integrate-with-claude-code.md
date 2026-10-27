---
translation_status: translated
---

# Claude Code との連携 { #integrate-with-claude-code }

Claude Code は Anthropic のエージェント型コーディングツールです。GPUStack のモデルデプロイメントは Anthropic API と互換性があるため、Claude Code を簡単に接続してコード生成に利用できます。このガイドでは、GPUStack との連携手順を説明し、Flappy Bird ゲームの作成を依頼して動作を確認します。

## 前提条件 { #prerequisites }

- 合計 100 GB 以上の VRAM を持つ 1 基以上の GPU。
- GPUStack がインストール済みで稼働していること。
- モデルファイルを取得するために Hugging Face または ModelScope にアクセスできること。

!!! note

    このガイドでは、NVIDIA H200 GPU を 1 基使用し、FP8 量子化の `Qwen3-Coder-Next` モデルを提供します。

## モデルのデプロイ { #deploy-the-model }

1. GPUStack UI の **Model Catalog** ページに移動します。

2. `Qwen3-Coder-Next` を検索し、デフォルト設定でデプロイします。

   ![モデルカタログ](../assets/integrations/claude-code/catalog.png)

3. デプロイ後は、Anthropic 互換 API で推論できます。

   ![準備完了のモデル](../assets/integrations/claude-code/model-ready.png)

## API キーの作成 { #create-an-api-key }

1. **Access Control** > **API Keys** ページで **Add API Key** をクリックします。

2. 名前を入力し、**Save** をクリックします。

3. API キーをコピーし、後で使えるよう保存します。

   ![API キー](../assets/integrations/claude-code/api-key.png)

## Claude Code のインストール { #install-claude-code }

[Claude Code ドキュメント](https://github.com/anthropics/claude-code)に従い、システムにインストールします。

## CC-Switch のインストール（任意） { #install-cc-switch-optional }

異なるモデルプロバイダーを簡単に切り替えるには、CC-Switch などのツールを使用できます。この手順は任意です。Claude Code を手動で設定したり、別のツールを使用したりしてもかまいません。

[CC-Switch](https://github.com/farion1231/cc-switch) のドキュメントに従ってインストールします。

## Claude Code での GPUStack の設定 { #configure-claude-code-with-gpustack }

1. CC-Switch を開き、次の設定でカスタムプロバイダーを追加します。

   - **Provider Name**：`GPUStack`
   - **API Endpoint**：GPUStack サーバーの URL。
   - **API Key**：先ほど作成した API キー。

2. すべてのモデルで `qwen3-coder-next` を使用するよう設定します。

   ![CC-Switch の設定](../assets/integrations/claude-code/cc-switch.png)

## 連携のテスト { #test-the-integration }

1. 新しいディレクトリを作成し、Claude Code を起動します。

   ```bash
   mkdir flappybird && cd flappybird
   claude
   ```

2. `/status` コマンドで設定が正しいことを確認します。

3. サンプルのタスクとして、Flappy Bird ゲームの作成を Claude Code に依頼します。

   ```
   Write a Flappy Bird game.
   ```

   ![Claude Code の結果](../assets/integrations/claude-code/claude.png)

4. 完了したら、生成された HTML ファイルをブラウザーで開き、結果を確認します。

   ![Flappy Bird ゲーム](../assets/integrations/claude-code/flappybird.gif)

## まとめ { #conclusion }

このガイドでは、Claude Code と GPUStack を連携し、Flappy Bird ゲームを生成しました。さらに複雑なコーディングタスクを試し、GPUStack の効率的なモデル提供機能を活用できます。
