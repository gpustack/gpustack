---
translation_status: translated
---

# SSS { #faq }

## Destek Matrisi { #support-matrix }

### Hibrit Küme Desteği { #hybrid-cluster-support }

GPUStack, NVIDIA, AMD, Ascend NPU'lar, Hygon DCU'lar, Moore Threads, Iluvatar, MetaX, Cambricon ve T-Head PPU'ları kapsayan heterojen kümeleri destekler ve hem AMD64 hem de ARM64 mimarilerinde çalışır.

### Dağıtılmış Çıkarım Desteği { #distributed-inference-support }

**Tek Düğümlü Çoklu GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

**Çok Düğümlü Çoklu GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

!!! tip

    İlgili belgeler:

    **vLLM**:[Dağıtılmış Çıkarım ve Sunma](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

    **SGLang**:[Çoklu Düğümlü Dağıtım](https://docs.sglang.io/references/multi_node_deployment/multi_node_index.html)

    **MindIE**: [Çok Düğümlü Çıkarım](https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html)

## Kurulum { #installation }

### Kayıtlı işçi adını nasıl değiştirebilirim? { #how-can-i-change-the-registered-worker-name }

GPUStack'i çalıştırırken `--worker-name` işaretini kullanarak bunu özel bir ad olarak ayarlayabilirsiniz:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-name New-Name
```

### Kayıtlı çalışan IP'sini nasıl değiştirebilirim? { #how-can-i-change-the-registered-worker-ip }

GPUStack'i çalıştırırken `--worker-ip` işaretini kullanarak bunu özel bir IP'ye ayarlayabilirsiniz:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-ip xx.xx.xx.xx
```

### GPUStack'in verileri nerede saklanıyor? { #where-are-gpustacks-data-stored }

GPUStack konteynerini çalıştırırken Docker birimi `--volume/-v` parametresi kullanılarak bağlanır. Varsayılan veri yolu Docker veri dizininin altındadır, özellikle de birimler alt dizinindedir ve varsayılan yol şöyledir:

```bash
/var/lib/docker/volumes/gpustack-data/_data
```

Aşağıdaki yöntemle kontrol edebilirsiniz:

```bash
docker volume ls
docker volume inspect gpustack-data
```

Özel bir yola değiştirmeniz gerekiyorsa, kapsayıcıyı çalıştırırken bağlama yapılandırmasını değiştirin. Örneğin, `/data/gpustack` ana bilgisayar dizinini bağlamak için:

```diff
sudo docker run -d --name gpustack \
    ...
    --volume /var/run/docker.sock:/var/run/docker.sock \
-    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/gpustack:/var/lib/gpustack \
    ...
    gpustack/gpustack
```

### Model dosyaları nerede saklanıyor? { #where-are-model-files-stored }

GPUStack konteynerini çalıştırırken Docker birimi `--volume/-v` parametresi kullanılarak bağlanır. Varsayılan önbellek yolu Docker veri dizininin altındadır, özellikle de birimler alt dizinindedir ve varsayılan yol şöyledir:

```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

Aşağıdaki yöntemle kontrol edebilirsiniz:

```bash
docker volume ls
docker volume inspect gpustack-data
```

Özel bir yola değiştirmeniz gerekiyorsa, kapsayıcıyı çalıştırırken bağlama yapılandırmasını değiştirin.

Örneğin, `/data/model-cache` ana bilgisayar dizinini bağlamak için:

```diff
sudo docker run -d --name gpustack \
    ...
    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/model-cache:/var/lib/gpustack/cache \
    ...
    gpustack/gpustack
```

---

## Modelleri Yönetme { #managing-models }

### Hugging Face'ten modeli nasıl dağıtabilirim? { #how-can-i-deploy-the-model-from-hugging-face }

Modelleri Hugging Face'ten dağıtmak için, model örneklerinin planlandığı sunucu düğümü ve çalışan düğümlerin Hugging Face'e erişimi olmalıdır veya bir ayna kullanabilirsiniz.

Örneğin, `hf-mirror.com` aynasını yapılandırın:

```diff
sudo docker run -d --name gpustack \
+    -e HF_ENDPOINT=https://hf-mirror.com \
    ...
    gpustack/gpustack
```

---

### Modeli Yerel Yoldan nasıl dağıtabilirim? { #how-can-i-deploy-the-model-from-local-path }

Modelleri Yerel Yoldan dağıtırken, model yolunun hedef çalışanlarda erişilebilir olduğundan emin olun.

Model belirli bir çalışanda depolanıyorsa modeli o çalışana dağıtmak için çalışan seçiciyi kullanabilirsiniz.

Diğer bir seçenek de paylaşılan bir depolamayı birden fazla düğüme monte etmektir.

Model dosyaları da konteynere monte edilmeli ve ana bilgisayar dizini ile konteyner montaj yolu aynı olmalıdır.

Safetensors modellerini Yerel Yoldan dağıtırken yol, **`*.safetensors`, `config.json` ve diğer dosyaları içeren model dizininin mutlak yolunu göstermelidir**.

GGUF modellerini Yerel Yoldan dağıtırken, **yolu, `.gguf` dosyasının** mutlak yoluna işaret etmelidir. Parçalanmış model dosyaları için ilk `.gguf` dosyasının (00001) mutlak yolunu kullanın.

---

### Model `Pending` durumunda takılı kalırsa ne yapmalıyım? { #what-should-i-do-if-the-model-is-stuck-in-pending-state }

`Pending` şu anda hiçbir çalışanın modelin gereksinimlerini karşılamadığı anlamına gelir. Sebebini görüntülemek için `Pending` durumunun üzerine gelin.

Öncelikle `Workers` kısmını kontrol ederek çalışan durumunun Hazır olduğundan emin olun.

O halde her arka ucun kendi işleme mantığı vardır. Örneğin vLLM için:

vLLM, tüm GPU'ların varsayılan olarak belleklerinin %90'ından fazlasının kullanılabilir olmasını gerektirir (`--gpu-memory-utilization` parametresi tarafından kontrol edilir). %90'ı aşan yeterli miktarda ayrılabilir GPU belleği olduğundan emin olun. Diğer modeller `Error` veya `Downloading` durumunda olsa bile GPU belleğinin zaten ayrılmış olduğunu unutmayın.

Tüm GPU'larda %90'dan fazla kullanılabilir bellek varsa ancak hala `Pending` görünüyorsa, bu durum belleğin yetersiz olduğunu gösterir. `safetensors` modelleri için gerekli GPU belleği (GB) şu şekilde tahmin edilebilir:

```
GPU Memory (GB) = Model weight size (GB) * 1.2 + 2
```

Ayrılabilir GPU belleği %90'ın altındaysa ancak modelin daha düşük bir ayırmayla çalışabileceğinden eminseniz `--gpu-memory-utilization` parametresini ayarlayabilirsiniz. Örneğin, GPU belleğinin %50'sini ayırmak için `Edit Model` → `Advanced` → `Backend Parameters`'e `--gpu-memory-utilization=0.5` ekleyin.

**Not**: Model çalıştırıldıktan sonra bir hatayla karşılaşırsa ve günlükler `CUDA: out of memory` gösteriyorsa, ayrılan GPU belleğinin yetersiz olduğu anlamına gelir. `--gpu-memory-utilization`'de daha fazla ayarlama yapmanız, daha fazla kaynak eklemeniz veya daha küçük bir model dağıtmanız gerekecek.

Modelin bağlam boyutu aynı zamanda gerekli GPU belleğini de etkiler. Daha küçük bir bağlam ayarlamak için `--max-model-len` parametresini ayarlayabilirsiniz. GPUStack'te bu parametre ayarlanmazsa varsayılan değeri 8192'dir. Arka uç parametrelerinde belirtilirse gerçek ayar geçerli olur.

Gerektiğinde daha küçük bir bağlama göre ayarlayabilirsiniz, örneğin `--max-model-len=2048`. Ancak her çıkarım isteği için maksimum belirteçlerin `--max-model-len` değerini aşamayacağını unutmayın. Bu nedenle bağlamın çok küçük ayarlanması çıkarımların kesilmesine neden olabilir.

`--enforce-eager` parametresi aynı zamanda GPU bellek kullanımının azaltılmasına da yardımcı olur. Bununla birlikte, vLLM'deki bu parametre, modeli istekli yürütme modunda yürütmeye zorlar; bu, grafik tabanlı yürütmede (tembel yürütmede olduğu gibi) optimizasyon için ertelenmek yerine, işlemlerin çağrıldığı anda hemen yürütüldüğü anlamına gelir. Bu, yürütmeyi yavaşlatabilir ancak hata ayıklamayı kolaylaştırabilir. Ancak grafik yürütmenin sağladığı optimizasyon eksikliği nedeniyle performansı da düşürebilir.

**SGLang** ve **MindIE** benzer bir süreci takip eder, yalnızca parametreleri farklılık gösterir. Daha fazla bilgi için [Yerleşik Çıkarım Arka Uçları](user-guide/built-in-inference-backends.md) bölümüne bakın.

---

### Model `Scheduled` durumunda takılı kalırsa ne yapmalıyım? { #what-should-i-do-if-the-model-is-stuck-in-scheduled-state }

Modelin planlandığı GPUStack kapsayıcısını yeniden başlatmayı deneyin. Sorun devam ederse nedenini analiz etmek için [buradan](troubleshooting.md#view-gpustack-logs) çalışan günlüklerini kontrol edin.

### Model `Error` durumunda takılı kalırsa ne yapmalıyım? { #what-should-i-do-if-the-model-is-stuck-in-error-state }

Sebebini görüntülemek için `Error` durumunun üzerine gelin. `View More` düğmesi varsa model günlüklerindeki hata mesajlarını kontrol etmek ve hatanın nedenini analiz etmek için buna tıklayın.

### `PYPI_PACKAGES_INSTALL` ile resmi vLLM görüntüsünü temel alan özel bir arka uç sürümü kullanıldığında model neden başlatılamıyor? { #why-does-the-model-fail-to-start-when-using-a-custom-backend-version-based-on-the-official-vllm-image-with-pypi_packages_install }

Resmi vLLM görüntüsünü temel alan özel bir vLLM arka uç sürümünü kullanarak bir model dağıtırken, ek Python paketlerini yüklemek için `PYPI_PACKAGES_INSTALL` kullanılırsa model başlatılamayabilir.

Şuna benzer bir hata görebilirsiniz:

```bash
/gpustack-command-xxxxxxxx: 40: /path/to/your_model: Permission denied
```

Bunun nedeni, kabın görüntünün varsayılan giriş noktası yerine özel bir komutla başlamasıdır.

`Inference Backends` → `vLLM` → `Edit` → `Versions Config` bölümünde, ilgili sürümü düzenleyin ve Görüntü Giriş Noktasını Geçersiz Kıl'ı şu şekilde ayarlayın:

```bash
vllm serve
```

### Bir modeli silmek neden disk alanında yer açmıyor? { #why-doesnt-deleting-a-model-free-up-disk-space }

Bu, yeniden konuşlandırma sırasında modelin yeniden indirilmesini önlemek içindir. `Model Files`'da manuel olarak temizlemeniz gerekir.

---

## Modelleri Kullanma { #using-models }

### Bir istekte en fazla 1 resim sağlanabilir hatasını nasıl çözebilirim? { #how-can-i-resolve-the-error-at-most-1-images-may-be-provided-in-one-request }

Bu vLLM'nin bir sınırlamasıdır. `Edit Model → Advanced → Backend Parameters`'deki `--limit-mm-per-prompt` parametresini gerektiği gibi ayarlayabilirsiniz. Örneğin, `--limit-mm-per-prompt={"image": 4}`, çıkarım isteği başına en fazla 4 görüntüyü desteklediği anlamına gelir; ayrıntılara bakın: [buradan](https://docs.vllm.ai/en/latest/configuration/engine_args/#-limit-mm-per-prompt).

---

## GPUStack'i Yönetme { #managing-gpustack }

### GPUStack'i bir proxy arkasında nasıl kullanırım? { #how-do-i-use-gpustack-behind-a-proxy }

GPUStack'i çalıştırırken standart proxy ortam değişkenlerini aktarmanızı öneririz.

Aşağıdaki örnekte GPUStack'in, NO_PROXY ortam değişkeninde belirtilen adreslere yapılan istekler dışında tüm istekleri hedef proxy'ye iletecek şekilde nasıl yapılandırılacağı gösterilmektedir.

```bash
docker run -d --name gpustack \
    -e HTTPS_PROXY="http://proxy-server:port" \
    -e HTTP_PROXY="http://proxy-server:port" \
    -e NO_PROXY="127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local" \
    ...
```

!!! note

    - IP adresini/proxy adresini uygun şekilde değiştirin.
    - Proxy'niz kimlik doğrulama gerektiriyorsa `http://username:password@proxy-server:port` biçimini kullanın. Şifrelerdeki özel karakterlerin URL kodlamasına ihtiyaç duyabileceğini unutmayın.
