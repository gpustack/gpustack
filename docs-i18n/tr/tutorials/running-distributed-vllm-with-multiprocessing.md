---
translation_status: translated
---

# Dağıtılmış vLLM'yi Çoklu İşleme Arka Uçuyla Çalıştırma { #running-distributed-vllm-with-the-multiprocessing-backend }

Tek bir çalışanın bir modeli barındırmak için yeterli GPU belleği veya bilgi işlemi olmadığı durumlarda vLLM, işbirliğine dayalı çıkarım için bir model örneğini birden fazla çalışana bölebilir. GPUStack, bu düğümler arası işlemleri **tek** model örneği olarak düzenler ve yalnızca bir OpenAI uyumlu uç nokta ortaya çıkarır. vLLM, `--distributed-executor-backend` parametresi aracılığıyla iki yürütme yolu sunar: `ray` ve `mp` (MultiProcessing). Bu eğitimde, tensör paralelliği (TP) ve veri paralelliği (DP) kombinasyonuyla birden çok çalışan arasında dağıtılmış çıkarım yürütmek için **MP arka ucunun** nasıl kullanılacağı gösterilir.

Bu eğitimde, GPUStack'te çalışanlar arası dağıtılmış çıkarımı nasıl etkinleştireceğinizi, MP arka ucunu açıkça seçmeyi ve iki düğüm arasında tek bir model örneğini dağıtmak için tensör paralelliğini veri paralelliğiyle birleştirmeyi öğreneceksiniz.

!!! note

    - MP arka ucu, `Backend Parameters`'de açıkça `--distributed-executor-backend=mp` ayarladığınızda veya vLLM ≥ `0.18.0` ile özel bir imaj (`-custom` son ekine sahip bir arka uç sürümü) kullandığınızda etkili olur. `0.18.0`'den itibaren vLLM artık varsayılan olarak Ray'e bağlı değildir, bu nedenle GPUStack, başlatma hatalarını önlemek için otomatik olarak `mp`'e geri döner.
    - Ray ve MP yolları, başsız takipçi ve `dp_only` / `mp_only` / `nested` topolojileri arasındaki kavramsal farklılıklar için, [Yerleşik Çıkarım Arka Uçları](../user-guide/built-in-inference-backends.md)'deki `Distributed Inference Across Workers` bölümüne bakın.
    - Bu özellik şu anda deneme aşamasındadır.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- GPUStack kurulu ve çalışıyor.
- Her biri **aynı sayıda ve aynı model** GPU'ya sahip en az **2 Linux düğümü**. Bu eğitimde örnek olarak **2 düğüm × 8 GPU** (toplam 16 GPU) kullanılmaktadır; bu, NVIDIA, AMD ve Ascend gibi çeşitli satıcıların hızlandırıcılarına uygulanabilir.
- Daha iyi performans için düğümler arasında yüksek hızlı bir ara bağlantı (NVLink veya InfiniBand gibi) önerilir.
- Model dosyalarına, katılan her düğümde**aynı yoldan** erişilebilir olmalıdır. Paylaşılan bir dosya sistemi kullanın veya model dosyalarını her düğümde aynı yola önceden indirin.
- Seçilen model `tensor-parallel-size=8`, `data-parallel-size=2` düzenine uygundur (yani her veri paralel kopyası bir tam 8 GPU düğümünü kaplar).

!!! note

    Bu eğitimde, her veri paralel kopyasının (`tensor-parallel-size=8`) bir 8 GPU düğümünü tamamen kapladığı varsayılmaktadır. Modelinize ve donanımınıza göre `--tensor-parallel-size` ve `--data-parallel-size`'yi ayarlayın.

## Adım 1: GPUStack Çalışanlarını Ekleyin { #step-1-add-gpustack-workers }

[Installation](../installation/installation.md)'in ardından, her düğümü aynı GPUStack kümesine eklemek için join komutunu çalıştırın.

Katıldıktan sonra, GPUStack kullanıcı arayüzündeki `Workers` sayfasında her iki düğümün de Hazır durumda olduğunu ve GPU'larının listelendiğini doğrulayın.

![Ekran görüntüsü: İki düğümün hazır olduğunu gösteren çalışanlar sayfası](../assets/tutorials/running-distributed-vllm-with-multiprocessing/1-workers-ready.png)

## Adım 2: Modeli MP Arka Uç ile Dağıtın { #step-2-deploy-the-model-with-the-mp-backend }

1. `Deployments` sayfasına gidin.
2. `Deploy Model`'e tıklayın, model kaynağını seçin (`Local Path` gibi) ve tüm düğümlerde tutarlı olan model yolunu girin.
3. `Backend`'in `vLLM` olarak ayarlandığını doğrulayın.
4. Gelişmiş ayarlar altında, çalışanlar arası dağıtılmış çıkarımı etkinleştirmek için `Allow Distributed Inference Across Workers` seçeneğini işaretleyin.
5. `Backend Parameters`'e aşağıdaki parametreleri ekleyin:

    ```bash
    --distributed-executor-backend=mp
    --tensor-parallel-size=8
    --data-parallel-size=2
    --enable-expert-parallel
    ```

6. Uyumluluk kontrolünü geçtikten sonra dağıtmak için `Save` seçeneğine tıklayın.


!!! tip "Dağıtılmış örneğin tamamını tek bir dağıtım olarak dağıtın"

    Düğümler arası dağıtılmış bir örneği **tek** dağıtım olarak dağıtmanızı öneririz. Yukarıda açıklandığı gibi yalnızca bir dağıtım oluşturduğunuzda GPUStack şunları yapacaktır:

    - her iki çalışandaki GPU'ları seçin (bu örnekte 16) ve düğüm başına `--data-parallel-size-local` değerini otomatik olarak türetin, böylece **manüel olarak** `--data-parallel-size-local` ayarlamanıza gerek kalmaz;
    - çok düğümlü topoloji parametrelerini `--nnodes`, `--node-rank`, `--master-addr` ve `--master-port` enjekte edin;
    - takipçi örneklerine otomatik olarak `--headless` eklenir, böylece yalnızca dağıtılmış hesaplamaya katılırlar ve bir API uç noktası başlatmazlar. Takipçi kayıtlarında `--headless`'in görülmesi bekleniyor.

!!! note "Otomatik planlama sağlanamadığında"

    Otomatik planlama, bir koşulun karşılanmaması nedeniyle kaynakları tahsis edemiyorsa (örneğin, GPU belleği `gpu_memory_utilization`'i karşılamıyor), `Scheduling Mode`'yi `Manual` veya `Specify GPU` olarak değiştirin ve ardından iki çalışanın her birinde 8 GPU seçmek için `GPU Selector`'i kullanın (toplamda 16). Otomatik planlamanın tüm koşulları için [Yerleşik Çıkarım Arka Uçları](../user-guide/built-in-inference-backends.md) konusuna bakın.


!!! note "Manuel olarak birden fazla dağıtıma bölerken: GPU sayım kontrolünü atlamak için `GPUSTACK_SKIP_GPU_COUNT_CHECK` tuşunu kullanın"

    Bazı senaryolarda, düğüm başına bir dağıtım oluşturmak (örneğin, harici bir yük dengeleyiciyi kendiniz yönetmek için), GPU'ları manuel olarak seçip `--data-parallel-size-local 1`'i elle ayarlamak isteyebilirsiniz. Bu durumda, her dağıtım yalnızca 8 GPU seçer, oysa `tensor-parallel-size × data-parallel-size = 16`, dolayısıyla GPU sayısı dünya boyutuyla eşleşmediğinden GPUStack bir hata bildirir: `the selected gpu count (8) does not match the world size (16)`.

    Eğer amaçladığınız düzen buysa, kontrolü atlamak ve manuel olarak seçtiğiniz GPU sayısını korumak için her dağıtımın `Environment Variables`'inde `GPUSTACK_SKIP_GPU_COUNT_CHECK=1`'ı ayarlayın. Bu anahtar **yalnızca GPU'lar manuel olarak seçildiğinde** (`Scheduling Mode` `Manual` veya `Specify GPU`'dir) etkili olur ve otomatik planlamayı etkilemez.

## 3. Adım: Dağıtımı İzleyin { #step-3-monitor-deployment }

`Deployments` sayfasından dağıtım durumunu izleyin. Örneğin her çalışandaki GPU kullanımını imajlemek için `Distributed Across Workers`'in üzerine gelin. Gerçek zamanlı günlükleri görmek için `View Logs`'ye tıklayın:

- **leader** örneği modeli yükler ve OpenAI uyumlu uç noktayı ortaya çıkarır;
- **follower** örnek günlükleri `--headless`'yi gösterir; bu, yalnızca dağıtılmış hesaplamaya katıldığı ve bir API bağlantı noktasını işgal etmediği anlamına gelir. Bu bekleniyor.

Modelin yüklenmesi birkaç dakika sürebilir.

![Ekran görüntüsü: Çalışanlar arasındaki dağılımı gösteren dağıtım ayrıntıları](../assets/tutorials/running-distributed-vllm-with-multiprocessing/3-distributed-across-workers.png)

## Adım 4: Oyun Alanı aracılığıyla doğrulayın { #step-4-verify-via-the-playground }

Dağıtım çalışmaya başladıktan sonra bunu doğrudan GPUStack Playground'da doğrulayın. Tüm harici istekler, bunları takipçilere gönderen lider aracılığıyla girer; böylece kullanım, tek çalışanlı dağıtımla aynıdır:

1. `Playground` > `Chat` seçeneğine gidin.
2. Yalnızca bir model konuşlandırılmışsa varsayılan olarak seçilir; aksi takdirde sağ üst köşedeki açılır menüden modelinizi seçin.
3. Bir bilgi istemi girin ve modelle etkileşim kurun.

![Ekran görüntüsü: Oyun Alanında dağıtılmış modelle sohbet etme](../assets/tutorials/running-distributed-vllm-with-multiprocessing/4-playground-chat.png)

Artık vLLM'nin MP arka ucunu kullanarak bir GPUStack kümesindeki birden fazla düğümde dağıtılmış çıkarımı başarıyla dağıttınız ve çalıştırdınız.

Daha fazla yardıma ihtiyacınız olursa GPUStack topluluğu veya destek ekibiyle iletişime geçmekten çekinmeyin.
