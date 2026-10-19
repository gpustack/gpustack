---
translation_status: translated
template: overview.html
---

<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>

<br>

<p align="center">
  <a href="https://github.com/gpustack/gpustack/blob/main/LICENSE" target="_blank">
    <img alt="ライセンス" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue">
  </a>
  <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white">
  </a>
</p>

<!-- github-buttons -->

# 概要 { #overview }

GPUStack は、AI モデルを効率的にデプロイするためのオープンソース GPU クラスターマネージャーです。vLLM、SGLang、TensorRT-LLM、独自の推論エンジンを構成・オーケストレーションし、GPU クラスター全体の性能を最適化します。

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle .icon-blue } __複数 GPU クラスターの管理__

    ---

    オンプレミスサーバー、Kubernetes クラスター、クラウドプロバイダーなど、さまざまな環境の GPU クラスターを管理します。

-   :material-engine-outline:{ .lg .middle .icon-green } __プラグイン可能な推論エンジン__

    ---

    vLLM、SGLang、TensorRT-LLM などの高性能推論エンジンを自動で構成します。必要に応じてカスタム推論エンジンも追加できます。

-   :material-rocket-launch-outline:{ .lg .middle .icon-orange } __モデル公開当日のサポート__

    ---

    GPUStack のプラグイン可能なエンジンアーキテクチャにより、新しいモデルを公開当日にデプロイできます。

-   :material-speedometer:{ .lg .middle .icon-red } __推論性能の最適化__

    ---

    低レイテンシまたは高スループット向けに事前調整されたモードを提供します。拡張 KV キャッシュ（LMCache、HiCache）と投機的デコーディング（EAGLE3、MTP）に対応しています。

-   :material-shield-check-outline:{ .lg .middle .icon-purple } __エンタープライズ水準の運用__

    ---

    自動障害復旧、負荷分散、監視、認証、アクセス制御をサポートします。

</div>

## アーキテクチャ { #architecture }

GPUStack は、開発チーム、IT 組織、サービスプロバイダーが Model-as-a-Service を大規模に提供できるようにします。LLM、音声、画像、動画モデル向けの業界標準 API に対応しています。ユーザー認証とアクセス制御、GPU 性能・使用率のリアルタイム監視、トークン使用量と API リクエストレートの詳細な計測機能を内蔵しています。

下図は、1 台の GPUStack サーバーでオンプレミスとクラウドにまたがる複数の GPU クラスターを管理する構成を示しています。GPUStack スケジューラーは GPU を割り当ててリソース使用率を最大化し、適切な推論エンジンを選択して性能を最適化します。管理者は、統合された Grafana と Prometheus のダッシュボードからシステムの健全性とメトリクスを把握できます。

![GPUStack v2 アーキテクチャ](assets/gpustack-v2-architecture.png)

## クラスター全体をひと目で把握 { #cluster-visibility-at-a-glance }

[GPUStack Enterprise](https://gpustack.ai/enterprise/) の GPU クラスタートポロジービューでは、すべてのワーカー、GPU、モデルデプロイメントを 1 か所で俯瞰できます。使用率、割り当て、健全性の状態がリアルタイムに表示されます。

![GPUStack クラスタートポロジー](assets/cluster-topology.png)

## 最適化された推論性能 { #optimized-inference-performance }

GPUStack はエンジンの自動選択とパラメーターの最適化により、導入直後から優れた推論性能を提供します。下図は、vLLM のデフォルト設定に対するスループットの向上を示しています。

![H200 スループット比較](assets/h200-throughput-comparison.png)

ベンチマーク手法と結果の詳細は、[推論性能ラボ](performance-lab/overview.md)をご覧ください。

## 対応アクセラレーター { #supported-accelerators }

GPUStack は、AI 推論用のさまざまなアクセラレーターに対応しています。

<div class="logo-tile-grid">
    <div class="logo-tile" data-tooltip="NVIDIA GPU">
        <img src="../assets/logos/nvidia.png" alt="NVIDIA" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="AMD GPU">
        <img src="../assets/logos/amd.png" alt="AMD" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Ascend NPU">
        <img src="../assets/logos/ascend.png" alt="Ascend" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Hygon DCU">
        <img src="../assets/logos/hygon.png" alt="Hygon" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="MThreads GPU">
        <img src="../assets/logos/mthreads.png" alt="MThreads" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Iluvatar GPU">
        <img src="../assets/logos/iluvatar.png" alt="Iluvatar" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="MetaX GPU">
        <img src="../assets/logos/metax.png" alt="MetaX" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Cambricon MLU">
        <img src="../assets/logos/cambricon.png" alt="Cambricon" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="T-Head PPU">
        <img src="../assets/logos/thead.png" alt="T-Head" class="off-glb">
    </div>
</div>

要件とセットアップ手順の詳細は、[インストール要件](installation/requirements.md)をご覧ください。
