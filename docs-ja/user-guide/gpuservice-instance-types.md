# GPU サービスのインスタンスタイプ { #gpu-service-instance-types }

GPU サービスのインスタンスタイプは、[GPU サービスインスタンス](gpuservice-instances.md)の作成元となる、CPU 専用またはアクセラレーター付きのコンピューティング構成を表します。

[GPUStack Operator](https://github.com/gpustack/gpustack-operator) は、AMD、Ascend、Cambricon、Hygon、Iluvatar、MetaX、Moore Threads、NVIDIA、T-Head のアクセラレーターに対応しています。各メーカーのアクセラレーターは、**排他**（デバイス全体）、**共有**、または**論理スライス**（独立した VRAM と演算枠を持つソフトウェア分割）で割り当てられます。NVIDIA、T-Head、Hygon は、ハードウェアでデバイスを分割する**物理パーティション分割**（MIG）にも対応しています。メーカーごとの詳細は、Operator の[アクセラレーター対応表](https://github.com/gpustack/gpustack-operator#accelerator-support)を参照してください。

## インスタンスタイプの閲覧 { #browsing-instance-types }

`GPU Service` > `Instance Types` を開きます。表示権限のあるすべての GPU サービスクラスターのインスタンスタイプが一覧表示され、各行に所属クラスターが示されます。

![複数クラスターのインスタンスタイプ一覧](../assets/gpuservice/instance-types/list.png)

- **名前 / フレーバー**：タイプ名（または表示名）と所属フレーバー。`NVIDIA-H100-80GB-HBM3` などのデバイスモデル、または `CPU-only` です。
- **単位 CPU / 単位 RAM / ストレージ**：タイプの 1 単位あたりに付与するホストリソース。アクセラレーターのフレーバーでは、1 単位はデバイス全体です。論理スライスと物理パーティションは、作成時にこれらを基準としてサイズが決まります。
- **プラットフォーム**：OS と CPU アーキテクチャ。互換性のあるイメージを選ぶ際に役立ちます。
- **クラスター**：タイプが所属するクラスター。
- **状態**：`Active` のタイプは *Add GPU Instance* フォームで選択でき、`Inactive` のものは選択できません。

ツールバーの `Filter by cluster` セレクターで単一のクラスターに絞り込めます。検索欄では、`Name` 列に表示される表示名または名前で検索できます。絞り込み、並べ替え、ページ分割はすべて Server 側で行われ、`Name` 列の並べ替えは表示ラベルに基づきます。行の無効化、有効化、削除の操作は、常にその行の Cluster 列にあるクラスターに対して実行します。

一覧は各クラスターからのリアルタイム読み取りではなく、コントロールプレーンの記録です。クラスターにアクセスできない間は、最後に観測した値を表示します。現在の空き容量は、以下のリアルタイム容量表示を基準にしてください。

各タイプのリアルタイム容量は、`kubectl` でも確認できます。アクセラレーターの 4 つの表示は、**EX**（排他）、**SH**（共有）、**SL**（論理スライス）、**PT**（物理パーティション）です。アクセラレーターの各表示も `CPU` 列も、`onceMaxRequest/remaining` の形式です。

```bash
kubectl get instancetypes
```

```
NAME                                          ENTRANCE                          UNIT(CPU/RAM)/STORAGE   ACCELERATOR(EX/SH/SL/PT)   CPU     PHASE
gpustack--generic-linux-amd64                 gpustack-fnv64-3b93966fd73eb9ec   1/2Gi/100Gi             0/0 0/0 0/0 0/0            16/40   Active
gpustack--nvidia-h100-80gb-hbm3-linux-amd64   gpustack-fnv64-e4768a65ca0ce96b   12/192Gi/100Gi          1/1 10/10 100/100 1/7      0/0     Active
gpustack--nvidia-l40s-linux-amd64             gpustack-fnv64-a730f1dca9e26fca   12/128Gi/100Gi          1/1 10/10 100/100 0/0      0/0     Active
```

この新しく追加したクラスターの表示では、H100 タイプは排他デバイス 1 つ（`EX 1/1`）、共有スロット 10 個（`SH 10/10`）、論理スライスの全枠（`SL 100/100`）、MIG パーティションスロット 7 個（`PT 1/7`、MIG を有効にした 2 台目の H100 ノード由来）を提供しています。L40S は、パーティションを除いて同様に単一デバイスを提供します（`PT 0/0` — L40S は MIG 非対応）。汎用の CPU 専用タイプには、アクセラレーター容量はありません。

## ノードから導出されるインスタンスタイプ { #derived-instance-types }

GPU サービスクラスターで `Derive Instance Types from Nodes` が **Enabled**（既定）になっている場合、Operator は各ノードのデバイスを検出し、アクセラレーターモデルごとのタイプと CPU 専用タイプを自動的に作成します。上の一覧も、このように導出されたものです。

導出タイプの単位 CPU/RAM は、Operator が管理する製品別のプリセットに基づきます。[インスタンスタイプの単位リソースのリファレンス](https://docs.gpustack.ai/gpustack-operator/main/docs/reference/instance-type-unit-resources/)に記載されています。ストレージは常に 100 GiB、導出される CPU 専用タイプは常に 1 CPU / 2 GiB RAM です。

導出タイプは Operator が所有します。

- **UI からは削除できません。** `Delete` 操作は無効です。

![導出インスタンスタイプでは Delete 操作が無効](../assets/gpuservice/instance-types/derived-delete-disabled.png)

- `kubectl` で削除しても、次の再同期時に Operator が再作成します。単位リソースは作成時に一度だけ記録されるため、Operator のアップグレード後に導出タイプのサイズを変更する場合は、この方法で再作成することがサポートされています。

導出タイプを完全に削除するには、先にクラスターの `Derive Instance Types from Nodes` を **Disabled** に設定します（クラスターの[高度なオプション](gpuservice-instances.md#advanced-options)を参照）。Operator はタイプの作成を停止し、既存の導出タイプを削除できるようになります。

## インスタンスタイプの追加 { #adding-an-instance-type }

より大きな CPU 専用構成が必要な場合など、導出タイプが合わない場合は、カスタムタイプを追加します。`Add Instance Type` をクリックしてフォームに入力します。

![インスタンスタイプ追加フォーム](../assets/gpuservice/instance-types/add-form.png)

- **クラスター**：タイプを作成するクラスター。
- **名前**：タイプ名。基盤リソース名の一部になるため、小文字で Kubernetes の規則に適合する必要があります。クラスター内で一意でなければならず、導出タイプを含む既存タイプの名前を再利用すると、上書きではなく拒否されます。
- **表示名**：UI に表示する、任意のわかりやすい名前。
- **フレーバー**：`CPU-only`、または選択したクラスターに存在するアクセラレーターのフレーバー。選択肢は Cluster フィールドに応じて変わります。
- **OS / Arch**：タイプの対象プラットフォーム。
- **単位 CPU / 単位 RAM / ストレージ**：1 単位あたりに付与するホストリソース。

`Save` をクリックします。新しいタイプが一覧と *Add GPU Instance* フォームに表示されます。

![一覧に追加された CPU-Large カスタムタイプ](../assets/gpuservice/instance-types/added.png)

!!! note

    Operator はカスタムインスタンスタイプを変更しません。導出処理はタイプを作成するだけで、自分がその時点で作成したもの以外には触れません。

## インスタンスタイプの無効化と有効化 { #deactivating-and-activating-an-instance-type }

タイプを削除せず、新しいインスタンスの作成を停止するには無効化します。タイプの行の無効化操作をクリックして確認します。

![インスタンスタイプの無効化確認](../assets/gpuservice/instance-types/deactivate-confirm.png)

タイプは `Inactive` になり、*Add GPU Instance* フォームで選択できなくなります。

![Inactive のインスタンスタイプ](../assets/gpuservice/instance-types/inactive.png)

再び有効化するには、同じ操作をクリックして確認します。

![インスタンスタイプの有効化確認](../assets/gpuservice/instance-types/activate-confirm.png)

## インスタンスタイプの削除 { #deleting-an-instance-type }

タイプの行の操作メニューを開き、`Delete` をクリックします。

![インスタンスタイプの操作メニューの Delete](../assets/gpuservice/instance-types/delete-menu.png)

削除を確認します。

![インスタンスタイプの削除確認](../assets/gpuservice/instance-types/delete-confirm.png)

カスタムタイプは完全に削除されます。導出が有効な間、導出タイプは削除できません。[導出インスタンスタイプ](#derived-instance-types)を参照してください。

## 物理パーティション分割（MIG） { #physical-partitioning-mig }

物理パーティション分割（MIG）されたデバイスには、専用のインスタンスタイプは作成されません。パーティションはデバイスのインスタンスタイプの容量（`PT` 表示）に現れ、[インスタンスの追加](gpuservice-instances.md#instance-type-selection)時にパーティションプロファイルを選択します。

Operator は、NVIDIA、T-Head、Hygon のアクセラレーターの MIG に対応しています。MIG モード自体は、**ノード単位で管理者が管理する**プロパティです。Operator は観測するだけで、有効化、無効化、再構成は行いません。ノードでパーティションを提供するには、ベンダーのツールで MIG を有効にし、そのノードの Operator のデバイスマネージャーを再起動します。NVIDIA ノードの例：

```bash
# On the node, per GPU or for all GPUs:
sudo nvidia-smi -i <id> -mig 1
# Then, from anywhere with cluster access:
kubectl -n gpustack-system rollout restart ds/gpustack-operator-device-manager-nvidia
```

前提条件、制限、ノード再起動時に MIG インスタンスが保持されないことなどを含む完全な手順は、ベンダーごとの MIG 運用手順を参照してください。

- [NVIDIA MIG の運用](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/nvidia-mig/)
- [T-Head MIG の運用](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/thead-mig/)
- [Hygon MIG の運用](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/hygon-mig/) — Hygon のモードはノード全体に適用され、分割されたノードはパーティションだけを提供します。
