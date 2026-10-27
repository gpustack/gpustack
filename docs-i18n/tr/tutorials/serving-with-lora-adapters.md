---
translation_status: translated
---

# LoRA Adaptörlü Modeller Sunma { #serving-models-with-lora-adapters }

LoRA (Düşük Sıralı Uyarlama), tüm ağırlıkları yeniden eğitmek yerine küçük bağdaştırıcı dosyaları yükleyerek temel modeli belirli bir alana uyarlayan, parametre açısından verimli bir ince ayar yöntemidir.

GPUStack, birden fazla LoRA bağdaştırıcısını konuşlandırılmış bir temel LLM'ye bağlamanıza olanak tanır ve bağdaştırıcı başına otomatik olarak bir Model Rotası oluşturur; arayanlar, OpenAI uyumlu `model` alanını ilgili rota adına ayarlayarak, her ince ayar için ayrı bir GPU ayırmadan, istek başına adaptörleri değiştirir.

Bu eğitimde, birden fazla LoRA bağdaştırıcısının vLLM, SGLang veya Ascend MindIE arka ucundaki tek bir temel modele nasıl ekleneceği ve OpenAI uyumlu API aracılığıyla belirli bir bağdaştırıcının nasıl çağrılacağı gösterilir.

!!! note

    LoRA şu anda yalnızca `vLLM`, `SGLang` ve `Ascend MindIE` arka uçlarında desteklenmektedir. `lora_list` yapılandırması diğer arka uçlarda göz ardı edilir.

## Önkoşullar { #prerequisites }

Devam etmeden önce aşağıdakilerden emin olun:

- GPUStack kurulu ve çalışıyor.
- GPU'lu bir Linux çalışan düğümü mevcuttur. Burada kullanılan temel model `Qwen3-8B`'dir, dolayısıyla GPU'nun 8B modeline hizmet vermek için yeterli belleğe ihtiyacı vardır (örneğin, 24 GB veya daha fazlası). İki bağdaştırıcı aynı örneği paylaşır ve yalnızca küçük miktarda ekstra bellek ekler.
- Model ve adaptör dosyalarını indirmek için Hugging Face'e erişim.

Bu eğitimde **iki** LoRA adaptörleri tek bir temel modele monte edilir, böylece arayanların istek başına ince ayarlar arasında nasıl geçiş yaptığını görebilirsiniz. Aşağıdaki yapılandırmayı kullanır:

| Ürün | Bu eğitimde kullanılan değer |
| --- | --- |
| Model kaynağı | `Hugging Face` |
| Arka uç | `vLLM` |
| Temel model | `Qwen/Qwen3-8B` |
| LoRA adaptörü 1 | `AMaslovskyi/qwen-devops-foundation-lora` (adı `devops`) |
| LoRA adaptörü 2 | `XinyuanWang/qwen3-8b-medical-lora` (adı `medical`) |

Her iki adaptör de `Qwen/Qwen3-8B`'ı temel model olarak bildirir. Bu kesin bir gerekliliktir: Bir dağıtıma takılan her bağdaştırıcının aynı temel modeli paylaşması gerekir.

!!! tip

    Eğitimi bu değerlerle tamamladıktan sonra kaynak olarak `ModelScope` veya `Local Path`'i, arka uç olarak `SGLang` veya `Ascend MindIE`'i ve kendi temel modelinizi ve adaptörlerinizi değiştirebilirsiniz. Arka uç başına farklılıklar için [Arka Uç Uyumluluğu ve Sınırları](#backend-compatibility-limits) konusuna bakın.

## 1. Adım: Temel Modeli LoRA Adaptörleriyle Dağıtın { #step-1-deploy-the-base-model-with-lora-adapters }

1. `Deployments` sayfasına gidin ve `Deploy Model`'e tıklayın, ardından kaynak olarak `Hugging Face`'yi seçin.
2. Arama kutusuna `Qwen/Qwen3-8B` yazın ve sonuçlardan seçin.
3. Arka ucu `vLLM` olarak ayarlayın.
4. `Advanced` bölümünü genişletin ve `LoRA Adapters`'i bulun. **ilk** adaptörünü ekleyin:

      - Bağdaştırıcı açılır menüsüne `AMaslovskyi/qwen-devops-foundation-lora` yazın ve seçin.
      - `LoRA name` alanına `devops` girin.

5. **ikinci** adaptörünü eklemek için tıklayın:

      - Bağdaştırıcı açılır menüsüne `XinyuanWang/qwen3-8b-medical-lora` yazın ve seçin.
      - `LoRA name` alanına `medical` girin.

6. Dağıtımı tamamlamak için `Save` öğesine tıklayın.

![Ekran görüntüsü: Dağıtım formunda her iki bağdaştırıcının da eklendiği LoRA Bağdaştırıcıları bölümü](../assets/tutorials/serving-with-lora-adapters/1-deployment-lora-form.png)

!!! note

    Temel model öneki olmadan yalnızca çıplak adaptör adını (`devops`, `medical`) girin. GPUStack, aynı zamanda her bağdaştırıcıyı çağırmak için kullandığınız Model Rota adı olan tam tanımlayıcıyı (`qwen3-8b:devops`, `qwen3-8b:medical`) otomatik olarak türetir (bkz. [Adım 2](#step-2-invoke-a-lora-adapter)).

!!! tip

    Her bağdaştırıcının ayrı bir `LoRA name` olması gerekir — daha fazlasını monte etmek için ekleme adımını tekrarlayın. İstediğiniz adaptör açılır listede görünmüyorsa öncelikle Hugging Face veya ModelScope'ta modelin gerçekten bir LoRA adaptörü gönderdiğini doğrulayın, ardından tam depo kimliğini seçmek için arama kutusuna yapıştırın.

## Adım 2: Bir LoRA Adaptörünü Çağırın { #step-2-invoke-a-lora-adapter }

Dağıtımdan sonra GPUStack, her LoRA bağdaştırıcısı için otomatik olarak `<base-model-name>:<adapter-name>` adlı bir Model Rota oluşturur. Bu eğitimde üç rota elde ediliyor: `qwen3-8b` tabanı, artı `qwen3-8b:devops` ve `qwen3-8b:medical`. Her LoRA rotası, `Model Routes` sayfasında temel modelin kendi Model Rotasının yanında görünür ve kolay tanımlama için `LoRA` rozetiyle etiketlenir.

![Ekran görüntüsü: Temel rotayı ve her iki LoRA adaptör rotasını listeleyen Model Rotalar sayfası](../assets/tutorials/serving-with-lora-adapters/3-model-routes-list.png)

`Playground`'da bir model seçtiğinizde açılır listede temel model ve tüm LoRA adaptörleri listelenir; seçeneğin değiştirilmesi, isteği ilgili adaptöre yönlendirir. Aynı durum OpenAI uyumlu API için de geçerlidir; `model` alanını temel model adına veya herhangi bir adaptör rota adına ayarlayın. Temelde tüm adaptörler aynı GPU örneğini paylaşır, dolayısıyla bunlar arasında geçiş yapmak, modelleri değiştirmekten çok daha ucuzdur.

![Ekran görüntüsü: Temel modeli ve her iki LoRA adaptörünü listeleyen oyun alanı modeli açılır menüsü](../assets/tutorials/serving-with-lora-adapters/2-playground-model-list.png)

Örneğin, aynı istem yalnızca `model` alanını değiştirerek iki ince ayardan birine yönlendirilebilir. DevOps bağdaştırıcısına bir Kubernetes sorusu gönderin:

```bash
curl http://<your-gpustack-server>/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b:devops",
    "messages": [{"role": "user", "content": "How do I debug a CrashLoopBackOff pod?"}]
  }'
```

Ardından yalnızca `model` değerini değiştirerek klinik soruyu tıbbi adaptöre yönlendirin:

```bash
curl http://<your-gpustack-server>/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b:medical",
    "messages": [{"role": "user", "content": "What are the first-line treatments for type 2 diabetes?"}]
  }'
```

`<your-gpustack-server>`'i sunucu adresinizle ve `<your-api-key>`'i `API Keys` sayfasında oluşturulan API anahtarıyla değiştirin. `"model": "qwen3-8b"` gönderilmesi bunun yerine adaptör uygulanmayan temel modeli hedefler.

## 3. Adım: LoRA Adaptörlerini Yönetin { #step-3-manage-lora-adapters }

Dağıtım oluşturulduktan sonra LoRA yapılandırmasını ayarlayabilirsiniz:

- **Bağdaştırıcı ekleyin**: Dağıtılan modeli düzenleyin ve kaydedin. Yeni eklenen bağdaştırıcının takılıp sunulmasından önce zaten `Running` durumunda olan örneklerin yeniden başlatılması gerekir.
- **Bir bağdaştırıcıyı kaldırın**: Dağıtılan modelden ilgili girişi silin. GPUStack, hiçbir `Running` örneği hala bağdaştırıcıyı kullanmadığında eşleşen Model Rotasını otomatik olarak kaldırır.
- **Ad çakışmaları**: İki farklı temel model, adları aynı Model Rotasına çözümlenen bağdaştırıcılar eklerse gönderim bir çakışma hatasıyla reddedilir. Kurtarmak için bağdaştırıcılardan birini yeniden adlandırın.


## Arka Uç Uyumluluğu ve Sınırları { #backend-compatibility-limits }

LoRA desteği, çıkarım arka uçlarına göre değişiklik gösterir:

| Arka uç | LoRA başlangıç argümanları / konfigürasyonu | Temel sınırlamalar |
| --- | --- | --- |
| `vLLM` | `--enable-lora`, `--max-loras`, `--lora-modules` | `--max-loras` `lora_list` uzunluğundan itibaren otomatik olarak ayarlanır. Temel model başladıktan sonra bağdaştırıcı sayısı sabitlenir; değişiklikler örneğin yeniden başlatılmasını gerektirir. |
| `SGLang` | `--enable-lora`, `--lora-paths name=path`, `--max-loras-per-batch` | Bir grupta aynı anda etkinleştirilen LoRA'ların sayısı `--max-loras-per-batch` ile sınırlıdır. |
| `Ascend MindIE` | `maxLoras` / `maxLoraRank` / `LoraModules` yapılandırma dosyasında | `maxLoraRank` varsayılan olarak 64'tür; Daha yüksek dereceli adaptörler başlatılamıyor. `maxLoras=0` otomatik olarak takılan adaptör sayısına eşittir. |

GPUStack bu parametreleri otomatik olarak enjekte eder; genellikle bunları `Backend Parameters` altında tekrarlamanıza gerek yoktur. Yalnızca `Advanced` → `Backend Parameters` altına bayrak ekleyerek varsayılanları geçersiz kılın (örneğin, `maxLoraRank`'i yükseltmek).

Tüm ayrıntılar ve sınırlamalar için her bir arka ucun resmi belgelerine bakın:

- vLLM: [LoRA Adaptörleri](https://docs.vllm.ai/en/stable/features/lora/)
- SGLang: [LoRA Sunumu](https://docs.sglang.ai/backend/lora.html)
- Ascend MindIE: [Multi-LoRA (yalnızca Çince)](https://www.hiascend.com/document/detail/zh/mindie/300/mindiellm/llmdev/user_guide/feature/multi_lora.md)
