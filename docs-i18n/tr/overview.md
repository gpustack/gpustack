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
    <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue">
  </a>
  <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white">
  </a>
</p>

<!-- github-buttons -->

# Genel Bakış { #overview }

GPUStack, AI modellerini sunmak ve GPU örnekleri sağlamak için açık kaynaklı bir GPU küme yöneticisidir. vLLM, SGLang, TensorRT-LLM veya kendi çıkarım motorlarınızı yapılandırıp düzenler ve talep üzerine SSH ile erişilebilen GPU örnekleri başlatmanıza olanak tanır.

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle .icon-blue } __Çok Kümeli GPU Yönetimi__

    ---

    Şirket içi sunucular, Kubernetes ve bulut sağlayıcıları genelinde GPU kümelerini yönetir. Model dağıtımları ve GPU örnekleri için yapılandırılabilir VRAM ve hesaplama kaynağı tahsisleriyle GPU dilimlemeyi destekler.

-   :material-engine-outline:{ .lg .middle .icon-green } __Takılabilir Çıkarım Motorları__

    ---

    vLLM, SGLang ve TensorRT-LLM gibi yüksek performanslı çıkarım motorlarını otomatik olarak yapılandırır. Gerektiğinde özel çıkarım motorları da ekleyebilirsiniz.

-   :material-speedometer:{ .lg .middle .icon-red } __Performans İçin Optimize Edilmiş__

    ---

    Düşük gecikme veya yüksek verim için önceden ayarlanmış modlar sunar. Prefill/decode ayrıştırmasını, platform tarafından yönetilen paylaşımlı KV önbelleğini ve spekülatif kod çözmeyi destekler.

-   :material-chart-line:{ .lg .middle .icon-green } __Yerleşik Performans Testleri__

    ---

    Auto Tune, eşzamanlılık düzeylerini veya istek hızlarını otomatik olarak değiştirerek en yüksek verimi veya hizmet düzeyi hedeflerinizi (SLO) karşılayan en yüksek eşzamanlılık düzeyini bulur.

-   :material-server-outline:{ .lg .middle .icon-blue } __GPU Örnekleri__

    ---

    Geliştirme, ince ayar ve etkileşimli iş yükleri için talep üzerine SSH ile erişilebilen GPU örnekleri başlatır.

-   :material-shield-check-outline:{ .lg .middle .icon-purple } __Kurumsal Düzeyde İşlemler__

    ---

    Otomatik arıza kurtarma, yük dengeleme, izleme, kimlik doğrulama ve erişim kontrolü için destek sunar.

</div>

## Mimari { #architecture }

GPUStack, geliştirme ekiplerinin, BT organizasyonlarının ve hizmet sağlayıcılarının geniş ölçekte Hizmet Olarak Model sunmasına olanak tanır. LLM, ses, görüntü ve video modelleri için endüstri standardı API'leri destekler. Platform, yerleşik kullanıcı kimlik doğrulaması ve erişim kontrolünü, GPU performansının ve kullanımının gerçek zamanlı izlenmesini ve belirteç kullanımı ile API istek oranlarının ayrıntılı ölçümünü içerir.

Aşağıdaki şekil, tek bir GPUStack sunucusunun hem şirket içi hem de bulut ortamlarında birden fazla GPU kümesini nasıl yönetebileceğini göstermektedir. GPUStack zamanlayıcı, kaynak kullanımını en üst düzeye çıkarmak için GPU'ları tahsis eder ve optimum performans için uygun çıkarım motorlarını seçer. Yöneticiler ayrıca entegre Grafana ve Prometheus kontrol panelleri aracılığıyla sistem sağlığına ve ölçümlere ilişkin tam görünürlük elde eder.

![gpustack-v2-mimarisi](assets/gpustack-v2-architecture.png)

## Bir Bakışta Küme Görünürlüğü { #cluster-visibility-at-a-glance }

[GPUStack Enterprise](https://gpustack.ai/enterprise/)'deki GPU Küme Topolojisi görünümü, gerçek zamanlı kullanım, tahsis ve sağlık durumuyla birlikte tüm filonuzun (her çalışan, GPU ve model dağıtımı) tek bir yerde canlı, kuşbakışı görünümünü sağlar.

![gpustack-küme-topolojisi](assets/cluster-topology.png)

## Optimize Edilmiş Çıkarım Performansı { #optimized-inference-performance }

GPUStack'in otomatik motor seçimi ve parametre optimizasyonu, kutudan çıktığı haliyle güçlü çıkarım performansı sunar. Aşağıdaki şekil, varsayılan vLLM yapılandırmalarına göre aktarım hızı iyileştirmelerini göstermektedir:

![h200-iş hacmi karşılaştırması](assets/h200-throughput-comparison.png)

Ayrıntılı kıyaslama yöntemleri ve sonuçları için [Çıkarım Performans Laboratuvarımızı](performance-lab/overview.md) ziyaret edin.

GPUStack'in yerleşik performans testleriyle dağıtımınızın ne kadar yük kaldırabileceğini keşfedin. **Auto Tune**, yükü uyarlanabilir biçimde ayarlayarak en yüksek verimi veya hizmet düzeyi hedeflerinizi (SLO) karşılayan en yüksek eşzamanlılık düzeyini otomatik olarak bulur. Ayrıntılı verim ve gecikme grafikleri kapasite planlamasına yol gösterir.

![gpustack-benchmark](assets/benchmark.png)

## Desteklenen Hızlandırıcılar { #supported-accelerators }

GPUStack, yapay zeka çıkarımı için çok çeşitli hızlandırıcıları destekler:

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

Ayrıntılı gereksinimler ve kurulum talimatları için [Yükleme Gereksinimleri](installation/requirements.md) belgelerine bakın.
