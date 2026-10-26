---
translation_status: translated
---

# Yerleşik Çıkarım Arka Uçları { #built-in-inference-backends }

GPUStack aşağıdaki çıkarım arka uçlarını destekler:

- [vLLM](#vllm)
- [SGLang](#sglang)
- [MindIE](#mindie)
- [VoxBox](#voxbox)

Bir model dağıtımını başlatırken, kullanıcı arayüzü aşağıdaki kriterlere göre otomatik olarak bir arka ucu önceden seçecektir:

- Model bilinen bir `Text-to-Speech` veya `Speech-to-Text` modeli ise `VoxBox` kullanılır.
- Model bilinen bir `Image` modeli ise `SGLang` kullanılır.
- Aksi halde `vLLM` kullanılır.

Bu ön seçim yalnızca dağıtım formundaki varsayılan değeri doldurur. Dağıtım için kullanılan gerçek arka uç, dağıtım isteğini gönderirken kullanıcı tarafından açıkça seçilen arka uç olacaktır. Kullanıcılar yukarıda sağlanan desteklenen arka uçlar listesinden seçim yapabilir.

!!! Note

    Desteklenen tüm çıkarım arka uçları için, [DockerHub](https://hub.docker.com/r/gpustack/runner) adresinde önceden oluşturulmuş Docker görsellerimiz mevcuttur. Kullanıcılar modelleri dağıttığında sistem ilgili görüntüleri otomatik olarak çekip çalıştıracaktır.

## vLLM { #vllm }

[vLLM](https://github.com/vllm-project/vllm), yüksek verimli ve bellek açısından verimli bir LLM çıkarımı arka ucudur. LLM'ı üretimde çalıştırmak için popüler bir seçimdir.

vLLM, aşağıdakiler de dahil olmak üzere en son teknolojiye sahip açık kaynak modellerinin çoğunu sorunsuz bir şekilde destekler:

- Transformatör benzeri LLM'ler (ör. `Qwen3`)
- Uzmanlar Karması (MoE) LLM'ler (ör. `DeepSeek V3/R1`)
- Çok modlu LLM'lar (ör. `Qwen3-VL`)
- Vektör gömme modelleri (ör. `Qwen3-Embedding`)
- Yeniden Sıralama Modelleri (ör. `Qwen3-Reranker`)

Varsayılan olarak GPUStack, modelin meta verilerine dayalı olarak model örneği için VRAM gereksinimini tahmin eder.

Parametreleri ihtiyaçlarınıza uyacak şekilde özelleştirebilirsiniz. Aşağıdaki vLLM parametreleri faydalı olabilir:

- `--gpu-memory-utilization` (varsayılan: 0.9): Model örneği için kullanılacak GPU belleğinin oranı.
- `--max-model-len`: Model bağlam uzunluğu. Geniş bağlamlı modeller için GPUStack, özellikle kaynakların kısıtlı olduğu ortamlarda model dağıtımını basitleştirmek amacıyla bu parametreyi otomatik olarak `8192` olarak ayarlar. Bu parametreyi ihtiyaçlarınıza uyacak şekilde özelleştirebilirsiniz.
- `--tensor-parallel-size`: Tensör paralel kopyalarının sayısı. Varsayılan olarak GPUStack, bu parametreyi mevcut GPU kaynaklarına ve modelin bellek gereksinimi tahminine göre ayarlar. Bu parametreyi ihtiyaçlarınıza uyacak şekilde özelleştirebilirsiniz.

Daha fazla ayrıntı için lütfen [vLLM CLI Referansı](https://docs.vllm.ai/en/stable/cli/serve/)'e bakın.

### Desteklenen Modeller { #supported-models }

Desteklenen modeller için lütfen vLLM [belgelerine](https://docs.vllm.ai/en/stable/models/supported_models.html) bakın.

### Desteklenen Özellikler { #supported-features }

#### Omni-Modal Desteği { #omni-modal-support }

[vLLM-Omni](https://github.com/vllm-project/vllm-omni), çok modlu çıkarım için tasarlanmış vLLM'nin bir uzantısıdır ve şunları destekler:

- **Omni Models**: Metin, ses, görüntü ve videonun eşzamanlı işlenmesi (ör. `Qwen3-Omni`)
- **Resim Görevleri**: Resim oluşturma ve düzenleme (ör. `Z-Image-Turbo`)
- **Video Görevleri**: Video oluşturma ve düzenleme (ör. `Wan2.2`)
- **Ses Görevleri**: Konuşma sentezi, ses klonlama ve daha fazlası (ör. `Qwen3-TTS`)

GPUStack, çok modlu modellerin dağıtımı ve yönetimi için kusursuz bir deneyim sunmak üzere vLLM-Omni ile entegre olur. Bir model vLLM arka ucu aracılığıyla dağıtıldığında GPUStack, meta verilerine dayalı olarak modelin omni-modal olup olmadığını otomatik olarak algılar ve vLLM-Omni için gerekli parametreleri ayarlar.

#### Çalışanlar Arasında Dağıtılmış Çıkarım (Deneysel) { #distributed-inference-across-workers-experimental }

Tek bir çalışanın bir modeli barındırmak için yeterli VRAM'i veya hesaplaması olmadığında, vLLM bir model örneğini birden fazla çalışana bölebilir ve çıkarımı işbirliği içinde çalıştırabilir. GPUStack, bu düğümler arası süreçleri tek bir model örneği olarak düzenler ve yine de harici olarak yalnızca bir OpenAI uyumlu uç noktayı kullanıma sunar.

Etkinleştirmek için bir modeli dağıtırken `Allow Distributed Inference Across Workers` seçeneğini işaretleyin.

GPUStack, vLLM dağıtılmış çıkarımı için vLLM'nin `--distributed-executor-backend` parametresi tarafından seçilen iki yürütme yolu sağlar. Genellikle manuel olarak ayarlamanıza gerek yoktur; GPUStack, kapsayıcı görüntüsüne ve vLLM sürümüne göre otomatik olarak bir tane seçer.

##### Ray Arka Uç { #ray-backend }

- **Kullanıldığında**: GPUStack'in resmi runner görsellerindeki varsayılan yol — runner görselleri, vLLM sürümünden bağımsız olarak her zaman Ray paketiyle birlikte gönderilir. Ayrıca, arka uç parametrelerinde `--distributed-executor-backend` açıkça `ray`'e veya `mp` olmayan herhangi bir değere (ör. `external_launcher`) ayarlandığında da kullanılır.
- **Çalışma şekli**: GPUStack, küme oluşturmak için katılan her düğümde bir [Ray](https://ray.io) yardımcı süreci başlatır. Lider düğüm ana vLLM sürecini ve API sunucusunu çalıştırırken Ray düğümler arasındaki görev planlamasını koordine eder.

##### MultiProcessing Arka Ucu { #multiprocessing-backend }

- **Kullanıldığında**: `--distributed-executor-backend=mp` arka uç parametrelerinde açıkça ayarlandığında; veya vLLM ≥ `0.18.0` üzerinde özel bir görüntü (`-custom` son ekine sahip arka uç sürümü) çalıştırırken — vLLM, Ray'i `0.18.0`'deki varsayılan bağımlılıklarından çıkardı, bu nedenle özel bir görüntü Ray içermeyebilir ve GPUStack, başlatma hatalarını önlemek için `mp`'ye geri döner.
- **Çalışma şekli**: Ray yardımcı süreci olmadan vLLM'nin çok süreçli, çok düğümlü yürütmesi doğrudan kullanılır. GPUStack, küme yapısına göre `--nnodes`, `--node-rank`, `--master-addr` ve `--master-port` gibi topoloji argümanlarını ekler.
- **API sunucusu olmayan takipçiler**: Takipçi olarak planlanan örnekler otomatik olarak `--headless` alır; böylece yalnızca dağıtılmış hesaplamaya katılır ve **OpenAI API sunucusu başlatmazlar**. HTTP uç noktasını yalnızca lider sunar; tüm dış istekler liderden girip takipçilere iletilir. Takipçi örneğin günlüklerinde `--headless` görmek normaldir; bu örnekler bu nedenle API portu kullanmaz.
- **Topoloji şekli**: GPUStack, paralellik yapılandırmasına (TP / PP / DP) bağlı olarak `dp_only`, `mp_only` veya `nested` parametre düzenlerinden birini otomatik olarak seçer.


!!! warning "Bilinen Sınırlamalar"

    1. Model dosyalarına tüm katılımcı çalışanlarda aynı yoldan erişilebilir olmalıdır. Paylaşılan bir dosya sistemi kullanmalı veya model dosyalarını tüm katılımcı çalışanlarda aynı yola indirmelisiniz.
    2. Ray modunda, her çalışan aynı anda yalnızca bir dağıtılmış vLLM modeli örneğine atanabilir.

Otomatik planlama aşağıdaki koşullarla desteklenir:

- Katılımcı çalışanlar aynı sayıda GPU'ya sahiptir.
- Çalışandaki tüm GPU'lar gpu_memory_utilization gereksinimini (varsayılan 0.9) karşılamalıdır.
- Tensör paralelliği ile GPU'lar dikkat kafalarının sayısına göre bölünür.
- Toplam VRAM talebi, tahmini VRAM talebinden daha yüksektir.

Yukarıdaki koşullar karşılanmazsa model örneği otomatik olarak planlanmayacaktır. Ancak model yapılandırmasında istediğiniz çalışanları/GPU'ları seçerek bunu manuel olarak planlayabilirsiniz.

### Parametre Referansı { #parameters-reference }

vLLM için desteklenen parametrelerin tam listesine [buradan](https://docs.vllm.ai/en/stable/cli/serve/) bakın.

## SGLang { #sglang }

[SGLang](https://github.com/sgl-project/sglang), büyük dil modelleri ve vizyon dili modelleri için yüksek performanslı bir hizmet çerçevesidir.

Tek bir GPU'dan büyük dağıtılmış kümelere kadar çok çeşitli kurulumlarda düşük gecikme süresi ve yüksek verimli çıkarım sağlamak üzere tasarlanmıştır.

Varsayılan olarak GPUStack, model meta verilerine dayalı olarak model örneği için VRAM gereksinimini tahmin eder.

Gerektiğinde GPUStack, geniş bağlamlı modeller için çeşitli parametreleri de otomatik olarak ayarlar. Ortak SGLang parametreleri şunları içerir:

- `--mem-fraction-static` (varsayılan: `0.9`): GPU başına ayrılabilir VRAM oranı. Zamanlayıcı bu değeri kaynak eşleştirme ve aday seçimi için kullanır. Modelin `backend_parameters`'sini kullanarak bunu geçersiz kılabilirsiniz.
- `--context-length`: Model bağlam uzunluğu. Geniş bağlamlı modeller için, otomatik olarak tahmin edilen bağlam uzunluğu aygıt kapasitesini aşarsa GPUStack, kaynakların kısıtlı olduğu ortamlarda dağıtımı basitleştirmek için bu parametreyi `8192` olarak ayarlar. Bu parametreyi gerektiği gibi özelleştirebilirsiniz.
- `--tp-size`: Tensör paralel boyutu. Açıkça sağlanmadığında, GPUStack bu parametreyi seçilen GPU'lara göre çıkarır ve ayarlar.
- `--pp-size`: Boru hattı paralel boyutu. Çok düğümlü dağıtımlarda GPUStack, modele ve küme yapılandırmasına göre `--tp-size` ve `--pp-size` kombinasyonunu belirler.
- Çok düğümlü bağımsız değişkenler: `--nnodes`, `--node-rank`, `--dist-init-addr`. Dağıtılmış çıkarım etkinleştirildiğinde GPUStack, çok düğümlü iletişimi başlatmak için bu bağımsız değişkenleri enjekte eder.

Daha fazla ayrıntı için lütfen [SGLang belgelerine](https://docs.sglang.ai/index.html) bakın.

### Desteklenen Modeller { #supported-models_1 }

Desteklenen modeller için lütfen SGLang [belgelerine](https://docs.sglang.ai/supported_models/generative_models.html) bakın.

SGLang ayrıca görüntü modellerini de destekler. Doğruladıklarımız arasında şunlar yer alıyor: Qwen-Image, Flux 1 Dev. Daha fazla bilgi için lütfen [SGLang Difüzyon belgelerine](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) bakın.

### Desteklenen Özellikler { #supported-features_1 }

#### Çalışanlar Arasında Dağıtılmış Çıkarım (Deneysel) { #distributed-inference-across-workers-experimental_1 }

GPUStack'te birden fazla çalışan arasında dağıtılmış SGLang çıkarımını etkinleştirebilirsiniz.

!!! warning "Bilinen Sınırlamalar"

    1. Katılımcı düğümlerin tümü Linux çalıştırmalı ve görüntüler/ortamlar Python sürümü ve iletişim kitaplıklarında (örn. NCCL) uyumlu olmalıdır.
    2. Model dosyalarına tüm düğümlerde aynı yoldan erişilebilir olmalıdır. Paylaşılan bir dosya sistemi kullanın veya modeli her düğümde aynı yola indirin.

Otomatik planlamalı aday seçimi aşağıdaki koşulları dikkate alır:

- Katılımcı GPU'lar `--mem-fraction-static` gereksinimini karşılar (varsayılan 0.9).
- Tek düğümlü çoklu GPU veya çok düğümlü çoklu GPU senaryolarında, toplam etkin tahsis edilebilir VRAM, modelin gereksinimlerini karşılamalıdır.
- Model paralelliği gereksinimleri karşılanmalıdır (örneğin, tensör paralel boyutuna bölünebilen toplam dikkat kafası sayısı), aksi takdirde aday reddedilir.

Yukarıdaki koşullar karşılanmazsa, yapılandırmada çalışanları/GPU'ları seçerek model örneğini manuel olarak zamanlayabilirsiniz, ancak aşırı taahhüt riski belirtilebilir.

#### Difüzyon Modelleri (Deneysel) { #diffusion-models-experimental }

SGLang Diffusion, görüntülerin ve videoların üretim sürecini hızlandırmayı amaçlayan, özellikle difüzyon modelleri için tasarlanmış yüksek performanslı bir çıkarım motorudur. SGLang'ın paralel işleme teknikleri ve optimize edilmiş çekirdekleri sayesinde, ana akım temellere (ör. Hugging Face Diffusers) kıyasla 1.2 kata kadar daha yüksek üretim hızına ulaşır.

Daha fazla ayrıntı için lütfen [SGLang Difüzyonu](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)'e bakın.

#### Diğer Gelişmiş Özellikler { #other-advanced-features }

Ek gelişmiş özellikler mevcuttur, örneğin:

- [Spekülatif Kod Çözme](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- [Hiyerarşik KV Önbelleğe Alma](https://docs.sglang.ai/advanced_features/hicache.html)

Lütfen kullanım talimatları için resmi belgelere bakın.

### Parametre Referansı { #parameters-reference_1 }

SGLang için desteklenen parametrelerin tam listesine [buradan](https://docs.sglang.ai/advanced_features/server_arguments.html) bakın.

## MindIE { #mindie }

[MindIE](https://www.hiascend.com/en/software/mindie), [Ascend donanımı](https://www.hiascend.com/en/hardware/product) üzerinde yüksek performanslı bir çıkarım hizmetidir.

### Desteklenen Modeller { #supported-models_2 }

MindIE, [burada](https://www.hiascend.com/software/mindie/modellist) listelenen çeşitli modelleri destekler.

GPUStack içinde [büyük dil modellerini (LLM'ler)](https://www.hiascend.com/software/mindie/modellist) ve [çok modlu dil modellerini (VLM'ler)](https://www.hiascend.com/software/mindie/modellist) destekleyin.

Ancak _yerleştirme modelleri_ ve _çok modlu nesil modeller_ henüz desteklenmemektedir.

### Desteklenen Özellikler { #supported-features_2 }

MindIE, [burada](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0001.html) belirtilen çeşitli özelliklere sahiptir.

Şu anda GPUStack, [Quantization](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html), [Genişletilmiş Bağlam Boyutu](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0295.html), [Dağıtılmış Çıkarım](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0296.html), [Mixture of Experts(MoE)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0297.html) dahil olmak üzere bu yeteneklerin bir alt kümesini desteklemektedir. [Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0300.html), [Spekülatif Kod Çözme](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0301.html), [Çoklu Token Tahmini](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0500.html), [Önek Önbelleğe Alma](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0302.html), [İşlev Çağrısı](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0303.html), [Çok Modlu Anlama](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0304.html), [Çok Kafalı Gizli Dikkat(MLA)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0305.html), [Tensör Paralelliği](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0511.html), [Bağlam Paralelliği](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0508.html), [Sıra Paralellik](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0517.html), [Uzman Paralelliği](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0509.html), [Veri Paralelliği](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0424.html), [Tampon Yanıtı (Ascend MindIE 2.0.RC1'den beri)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0425.html). [KV Havuzu(Ascend MindIE 2.2.RC1'den beri)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0538.html).

!!! note

    1. Niceleme belirli bir ağırlığa ihtiyaç duyar ve modelin `config.json` ayarlanması gerekir, doğru ağırlığı hazırlamak için lütfen [referansı(kılavuz)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html) takip edin.

    2. Bazı özellikler birbirini dışlar, bu nedenle bunları kullanırken dikkatli olun. Örneğin, Önek Önbelleğe Alma etkinleştirildiğinde Bağlam Boyutunu Genişletme özelliği kullanılamaz.

### Parametre Referansı { #parameters-reference_2 }

MindIE'nin yapılandırılabilir [parametreleri](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_service0285.html) ve [ortam değişkenleri](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0416.html) vardır.

JSON'u doğrudan yapılandırmayı önlemek için GPUStack aşağıdaki gibi bir dizi komut satırı parametresi sağlar.

| Parametre | Varsayılan | Aralık | Kapsam | Açıklama |
|------------------------------------------------------|---------|--------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--log-level`                                        | Bilgi | | Günlük Yapılandırması | MindIE için günlük düzeyi. Seçenekler: `Verbose`, `Info`, `Warning`, `Warn`, `Error`, `Debug`.                                                                                                                                                                                          |
| `--max-link-num`                                     | 1000 | [1, 1000] | Sunucu Yapılandırması | Maksimum paralel istek sayısı.                                                                                                                                                                                                                                            |
| `--token-timeout`                                    | 60 | [1, 3600] | Sunucu Yapılandırması | Saniye cinsinden token oluşturma zaman aşımı.                                                                                                                                                                                                                                      |
| `--e2e-timeout`                                      | 60 | [1, 3600] | Sunucu Yapılandırması | E2E (isteğin kabul edilmesinden çıkarımın durdurulmasına kadar) saniye cinsinden zaman aşımı.                                                                                                                                                                                                            |
| `--openai-support`                                   | vllm | | Sunucu Yapılandırması | OpenAI API için uyumluluk modu.                                                                                                                                                                                                                                          |
| `--kv-pool-config`                                   | | | Arka Uç Yapılandırması | JSON formatında KV havuz yapılandırması. Örneğin: `{"backend":"<KV pool backend name>", "configPath":"/path/to/your/config/file"}`.                                                                                                                                            |
| `--max-seq-len`                                      | 8192 | (0, \*) | Model Dağıtım Yapılandırması | Model bağlam uzunluğu. Belirtilmemişse model yapılandırmasından türetilecektir.                                                                                                                                                                                                 |
| `--max-input-token-len`                              | | (0, `--max-seq-len`] | Model Dağıtım Yapılandırması | Maksimum giriş token'ı uzunluğu. Belirtilmezse `min(--max-seq-len, --max-prefill-tokens)`'den türetilecektir. |
| `--truncation`                                       | | | Model Dağıtım Yapılandırması | Minimum `--max-input-token-len` ve `--max-seq-len` - 1'i aştığında giriş token'ı uzunluğunu kesin. |
| `--cpu-mem-size`                                     | 0 | [0, \*) | Model Yapılandırması | GiB'de CPU takas alanı boyutu. `--max-preempt-count` belirtildiğinde çalışır.                                                                                                                                                                                                         |
| `--npu-memory-fraction` (env: `NPU_MEMORY_FRACTION`) | 0.9 | (0, 1] | Model Yapılandırması | Model yürütücüsü için kullanılacak NPU belleğinin oranı (0 ila 1). Örneğin: `0.5` %50 hafıza kullanımı anlamına gelir |
| `--trust-remote-code`                                | | | Model Yapılandırması | Uzak koda güvenin (model yükleme için).                                                                                                                                                                                                                                          |
| `--models`                                           | | | Model Yapılandırması | Uzman Paralellik Uygulama Yöntemi, Tensör Paralelliği LM Başlığı/Çıktı Dikkat Bölmesi gibi belirli belirli yapılandırmalar için JSON formatında yapılandırma modelleri.                                                                                                   |
| `--async-scheduler-wait-time`                        | 120 | [1, 3600] | Model Yapılandırması | Asenkron zamanlayıcının başlaması için bekleme süresi (saniye cinsinden).                                                                                                                                                                                                             |
| `--cache-block-size`                                 | 128 | | Zamanlama Yapılandırması | KV önbellek bloğu boyutu. 2'nin kuvveti olmalı. |
| `--max-prefill-batch-size`                           | 50 | [1, `--max-batch-size`] | Zamanlama Yapılandırması | Ön doldurma aşamasında toplu hale getirilen maksimum istek sayısı. `--max-batch-size`'den küçük olmalıdır.                                                                                                                                                                                  |
| `--max-prefill-tokens  `                             | | [1, `--max-seq-len`] | Zamanlama Yapılandırması | Her ön doldurma sırasında, mevcut gruptaki tüm giriş token'larının toplam sayısı `--max-prefill-tokens`'ı geçemez. Varsayılan `--max-seq-len` ile aynıdır.                                                                                                                        |
| `--prefill-time-ms-per-req`                          | 150 | [0, 1000] | Zamanlama Yapılandırması | İstek başına tahmini ön doldurma süresi (ms). Ön doldurma ve kod çözme aşaması arasında karar vermek için kullanılır.                                                                                                                                                                                       |
| `--prefill-policy-type`                              | 0 | | Planlama Yapılandırması | Ön doldurma aşaması stratejisi: <br> `0`: FCFS (İlk Gelen İlk Servis). <br> `1`: DURUM (FCFS ile aynı). <br> `2`: ÖNCELİK (öncelik sırası). <br> `3`: MLFQ (Çok Düzeyli Geri Bildirim Kuyruğu).                                                                                              |
| `--max-batch-size`                                   | 200 | [1, 5000] | Zamanlama Yapılandırması | Kod çözme aşamasında toplu hale getirilen maksimum istek sayısı.                                                                                                                                                                                                                         |
| `--max-iter-times`                                   | | [1, `--max-seq-len`] | Zamanlama Yapılandırması | Kod çözme aşaması için maksimum yineleme. Varsayılan `--max-seq-len` ile aynıdır.                                                                                                                                                                                                      |
| `--decode-time-ms-per-req`                           | 50 | [0, 1000] | Zamanlama Yapılandırması | İstek başına tahmini kod çözme süresi (ms). Toplu seçim için `--prefill-time-ms-per-req` ile kullanılır.                                                                                                                                                                              |
| `--decode-policy-type`                               | 0 | | Zamanlama Yapılandırması | Aşama stratejisini kodla: <br> `0`: FCFS <br> `1`: STATE (öncelikli veya değiştirilmiş isteklere öncelik verin) <br> `2`: ÖNCELİK <br> `3`: MLFQ |
| `--max-preempt-count`                                | 0 | [0, `--max-batch-size`] | Zamanlama Yapılandırması | Kod çözme sırasında izin verilen maksimum öncelikli istek sayısı. `--max-batch-size`'den küçük olmalıdır.                                                                                                                                                                             |
| `--support-select-batch`                             | | | Zamanlama Yapılandırması | Toplu seçimi etkinleştirin. `--prefill-time-ms-per-req` ve `--decode-time-ms-per-req`'ye göre yürütme önceliğini belirler. Açıkça devre dışı bırakmak için `--no-support-select-batch` tuşunu kullanın.                                                                                               |
| `--max-queue-delay-microseconds`                     | 5000 | [500, 1000000] | Zamanlama Yapılandırması | Mikrosaniye cinsinden maksimum kuyruk bekleme süresi.                                                                                                                                                                                                                                        |
| `--max-first-token-wait-time`                        | 2500 | [0, 3600000] | Zamanlama Yapılandırması | İlk token oluşturulması için beklenecek maksimum milisaniye.                                                                                                                                                                                                                    |
| `--override-generation-config`                       | | | | JSON formatında oluşturma yapılandırmasını geçersiz kılar veya ayarlar. Örneğin: `{"temperature": 0.5}`. Bu, model yapısının `generation_config.json`'si ile birleşecektir.                                                                                                              |
| `--enforce-eager`                                    | | | | İstekli modda operatörleri yayınlayın.                                                                                                                                                                                                                                                   |
| `--no-metrics`                                       | | | | `/metrics` uç noktasında ölçümleri göstermeyi devre dışı bırakın.                                                                                                                                                                                                                                |
| `--dtype`                                            | auto | | | Model ağırlıkları ve aktivasyonları için veri türü. <br> `auto`: model yapılandırmasının varsayılan veri türünü kullanın. <br> `half`/`float16`: FP16 için. <br> `bfloat16`: BF16 için. <br> `float`/`float32`: FP32 için.                                                                          |
| `--rope-scaling`                                     | | | Bağlam Boyutunu Genişletme | JSON formatında RoPE ölçeklendirme yapılandırması. Örneğin: `{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`. Bu, model yapısının `config.json`'si ile birleşecektir.                                                                                 |
| `--rope-theta`                                       | | | Bağlam Boyutunu Genişletme | RoPE theta yapılandırması. Bu, model yapısının `config.json`'si ile birleşecektir.                                                                                                                                                                                        |
| `--enable-split`                                     | | | Split Fuse | Bölünmüş sigortayı etkinleştirin, parçalı ön doldurma gibi bir şey. Açıkça devre dışı bırakmak için `--no-enable-split` tuşunu kullanın.                                                                                                                                                                               |
| `--policy-type`                                      | 0 | | Split Fuse | Bölünmüş sigorta stratejisi. <br> `0`: FCFS, ilk gelen ilk servis yapar. <br> `4`: SJF, önce en kısa iş. <br> `5`: LJF, önce en uzun iş. <br> `6`: MLFQ'ya Atla-Katıl, çok seviyeli geri bildirim kuyruğunu atla-Katıl. <br> `7`: SJF-MLFQ, önce en kısa iş ve çok seviyeli geri bildirim kuyruğu. |
| `--split-chunk-tokens`                               | 512 | [1, 8192] | Split Fuse | Bölünmüş sigorta için toplu token boyutu. 16'nın katları. |
| `--split-start-batch-size`                           | 16 | [0, `--max-batch-size`] | Split Fuse | Bölünmüş sigorta için bölmeye başlamak için parti boyutu.                                                                                                                                                                                                                                   |
| `--enable-memory-decoding`                           | | | Spekülatif Kod Çözme / Bellek Kod Çözme | Bellek kod çözme spekülasyonunu etkinleştirin. Açıkça devre dışı bırakmak için `--no-enable-memory-decoding` tuşunu kullanın.                                                                                                                                                                                    |
| `--memory-decoding-length`                           | 16 | [1, 16] | Spekülatif Kod Çözme / Bellek Kod Çözme | Bellek kod çözme spekülasyonunun uzunluğu.                                                                                                                                                                                                                                         |
| `--memory-decoding-dynamic-algo`                     | | | Spekülatif Kod Çözme / Bellek Kod Çözme | Bellek kod çözme spekülasyonu için dinamik algoritmayı etkinleştirin.                                                                                                                                                                                                                       |
| `--enable-lookahead`                                 | | | Spekülatif Kod Çözme / İleriye Yönelik | İleriye dönük spekülasyonu etkinleştirin. Açıkça devre dışı bırakmak için `--no-enable-lookahead` tuşunu kullanın.                                                                                                                                                                                                |
| `--lookahead-level`                                  | 4 | [3, 16] | Spekülatif Kod Çözme / İleriye Yönelik | İleriye dönük spekülasyon seviyesi.                                                                                                                                                                                                                                                |
| `--lookahead-window`                                 | 5 | [1, 16] | Spekülatif Kod Çözme / İleriye Yönelik | İleriye dönük spekülasyon için pencere boyutu.                                                                                                                                                                                                                                          |
| `--lookahead-guess-set-size`                         | 5 | [1, 16] | Spekülatif Kod Çözme / İleriye Yönelik | İleriye dönük spekülasyon için ayarlanan boyutu tahmin edin.                                                                                                                                                                                                                                       |
| `--enable-multi-token-prediction`                    | | | Çoklu Token Tahmini | Çoklu token tahminini etkinleştirin. Açıkça devre dışı bırakmak için `--no-enable-multi-token-prediction` tuşunu kullanın.                                                                                                                                                                                  |
| `--multi-token-prediction-tokens`                    | 1 | (0, \*) | Çoklu Tokenlu Tahmin | Çoklu tokenlu tahmin token'larının sayısı. Bu yalnızca `--enable-multi-token-prediction` etkinleştirildiğinde etkilidir.                                                                                                                                                              |
| `--enable-prefix-caching`                            | | | Önek Önbelleğe Alma | Önek önbelleğe almayı etkinleştirin. Açıkça devre dışı bırakmak için `--no-enable-prefix-caching` tuşunu kullanın.                                                                                                                                                                                                  |
| `--pipeline-parallel-size`, `-pp`                    | 1 | (0, \*) | Paralellik | Boru hattı paralel gruplarının sayısı.                                                                                                                                                                                                                                             |
| `--data-parallel-size`, `-dp`                        | -1 | | Paralellik | Dikkat katmanları için veri paralel gruplarının sayısı. `-1` veri paralelliğinin devre dışı bırakılması anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--context-parallel-size`, `-cp`                     | -1 | | Paralellik | Dikkat katmanları için bağlam paralel gruplarının sayısı. `-1` bağlam paralelliğini devre dışı bırakmak anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--tensor-parallel-size`, `-tp`                      | -1 | | Paralellik | Dikkat katmanları için tensör paralel gruplarının sayısı. `-1` dünya boyutunu tensör paralel boyutu olarak kullanmak anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--sequence-parallel-size`, `-sp`                    | -1 | `--tensor-parallel-size` | Paralellik | MLP katmanları için dizi paralel gruplarının sayısı. `-1` dizi paralelliğinin devre dışı bırakılması anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--moe-expert-parallel-size`, `-moe-ep`              | -1 | | Paralellik | Uzman paralel grup sayısı. `-1` MoE uzman paralelliğinin devre dışı bırakılması anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--moe-tensor-parallel-size`, `-moe-tp`              | -1 | | Paralellik | MoE MLP katmanları için tensör paralel gruplarının sayısı. `-1` MoE tensör paralel boyutu olarak dünya boyutunun kullanılması anlamına gelir, aksi takdirde 2'nin kuvveti olmalıdır. |
| `--enable-buffer-response`                           | | | Tampon Yanıtı | Tampon yanıtını etkinleştirin. Açıkça devre dışı bırakmak için `--no-enable-buffer-response` tuşunu kullanın.                                                                                                                                                                                                |
| `--prefill-expected-time-ms`                         | | | Arabellek Yanıtı | Milisaniye cinsinden İlk Belirteç Süresi (TTFT) için beklenen gecikme (SLO).                                                                                                                                                                                                          |
| `--decode-expected-time-ms`                          | | | Tampon Yanıtı | Milisaniye cinsinden Çıkış Token'ı Başına Süre (TPOT) için beklenen gecikme (SLO).                                                                                                                                                                                                        |

!!! note

    GPUStack, kullanıcıların model dağıtımı sırasında özel ortam değişkenleri eklemesine olanak tanır, ancak bazı değişkenler GPUStack yönetimiyle çakışabilir.

    Dolayısıyla GPUStack bu değişkenleri geçersiz kılacak/önleyecektir. Lütfen model örneği günlüklerinin çıktısını beklentilerinizle karşılaştırın.

## VoxBox { #voxbox }

[VoxBox](https://github.com/gpustack/vox-box), Metinden Konuşmaya ve Konuşmadan Metne modellerini dağıtmak için tasarlanmış bir çıkarım motorudur. Ayrıca OpenAI ses API'si ile tamamen uyumlu bir API sağlar.

### Desteklenen Modeller { #supported-models_3 }

| Model | Tür | Bağlantı | Desteklenen Platformlar |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Faster-whisper-large-v3         | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3) | AMD64,ARM64 |
| Faster-whisper-large-v2         | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2) | AMD64,ARM64 |
| Faster-whisper-large-v1         | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1) | AMD64,ARM64 |
| Faster-whisper-medium           | Konuşmayı Metne Dönüştürme | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium) | AMD64,ARM64 |
| Faster-whisper-medium.en        | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en) | AMD64,ARM64 |
| Faster-whisper-small            | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small) | AMD64,ARM64 |
| Faster-whisper-small.en         | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en) | AMD64,ARM64 |
| Faster-distil-whisper-large-v3  | Konuşmadan Metne Dönüştürme | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3) | AMD64,ARM64 |
| Faster-distil-whisper-large-v2  | Konuşmadan Metne Dönüştürme | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2) | AMD64,ARM64 |
| Faster-distil-whisper-medium.en | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | AMD64,ARM64 |
| Faster-whisper-tiny             | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny) | AMD64,ARM64 |
| Faster-whisper-tiny.en          | Konuşmadan Metne | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en) | AMD64,ARM64 |
| CosyVoice-300M-Instruct         | Metinden Konuşmaya | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct) | AMD64 |
| CosyVoice-300M-SFT              | Metinden Konuşmaya | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT) | AMD64 |
| CosyVoice-300M                  | Metinden Konuşmaya | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M) | AMD64 |
| CosyVoice-300M-25Hz             | Metinden Konuşmaya | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz) | AMD64 |
| CosyVoice2-0.5B                 | Metinden Konuşmaya | [Hugging Face](https://huggingface.co/gpustack/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B) | AMD64 |
| Dia-1.6B                        | Metinden Konuşmaya | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B) | AMD64 |

### Desteklenen Özellikler { #supported-features_3 }

#### GPU/CPU Boşaltmasına İzin Ver { #allow-gpucpu-offloading }

VoxBox, modellerin NVIDIA GPU'lara dağıtılmasını destekler. GPU kaynakları yetersizse modelleri otomatik olarak CPU'ya dağıtır.

!!! note "Akış Desteklenmiyor"

    VoxBox, Metinden Konuşmaya (`/v1/audio/speech`) veya Konuşmadan Metne (`/v1/audio/transcriptions`) yönelik akışı desteklemez. Akışlı ses API'lerini kullanmak için bunun yerine `vLLM` arka ucuna vLLM uyumlu bir ses modeli dağıtın.
