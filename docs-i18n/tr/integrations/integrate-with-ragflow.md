---
translation_status: translated
---

# RAGFlow ile entegrasyon { #integrate-with-ragflow }

RAGFlow, yerel olarak dağıtılan LLM'lerden, görsel modellerden, gömme modellerinden, yeniden sıralamadan, Konuşmadan Metne ve Metinden Konuşmaya yeteneklerinden yararlanmak için GPUStack ile entegre olabilir. RAGFlow, GPUStack'te çalışan modelleri kendisi bulur; bir kez bağlanıp eklenecek modelleri seçmeniz yeterlidir.

!!! note

    Bu kılavuz, GPUStack kurulum kartını ekleyen ve v1.0.0-rc3'ten sonra gelen bir RAGFlow sürümü gerektirir.

## Modelleri Dağıtma { #deploying-models }

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin ve ihtiyacınız olan modelleri dağıtmak için `Deploy Model` düğmesine tıklayın. RAGFlow yalnızca çalışan ve rotası olan modelleri bulduğundan `Enable Model Route` seçeneğini açık bırakın. Örnek modeller:

    | Model | RAGFlow'daki türü |
    | --- | --- |
    | `qwen3-4b` | Chat |
    | `qwen2.5-vl-3b-instruct` | Chat + Vision |
    | `bge-m3` | Embedding |
    | `bge-reranker-v2-m3` | Rerank |
    | `faster-whisper-small` | Speech to text |
    | `cosyvoice2-0.5b` | Text to speech |

2. Her modelin replikaları hazır olana kadar bekleyin:

    ![GPUStack dağıtım listesi](../assets/integrations/integration-ragflow-gpustack-deployments.png)

## API Anahtarı Oluşturma { #create-an-api-key }

1. `Access Control` > `API Keys` sayfasına gidin, ardından `Add API Key` düğmesine tıklayın.

2. Adı girin ve `Access Permissions` altında `Model Access` seçili kalacak şekilde `All models` seçeneğini koruyun. Varsayılan `Expiration` değeri `1 month` olduğundan, RAGFlow bu anahtarı kullanmaya devam edecekse daha uzun bir süre seçin. Ardından `Save` düğmesine tıklayın.

    ![GPUStack API anahtarı](../assets/integrations/integration-ragflow-gpustack-api-key.png)

3. API anahtarını kopyalayın ve daha sonra kullanmak üzere saklayın. Bundan sonra `API Keys` sayfası yalnızca maskelenmiş değeri gösterir.

## GPUStack'i RAGFlow'a Entegre Etme { #integrating-gpustack-into-ragflow }

1. RAGFlow arayüzünü açın, `Model providers` sayfasını açmak için sağ üst köşedeki avatara tıklayın, ardından `Available models` altından `GPUStack` seçeneğini seçin.

2. Aşağıdaki alanları doldurun, ardından `Connect` düğmesine tıklayın:

    - Instance name: Bu GPUStack sunucusunu RAGFlow'da tanımlayan herhangi bir ad, örneğin `gpustack`.
    - GPUStack server URL: `/v1` yolu olmadan `http://your-gpustack-url`. Özel bir bağlantı noktası kullanıyorsanız ekleyin. `localhost` RAGFlow kapsayıcısını gösterdiği için kullanmayın. GPUStack, RAGFlow ile aynı ana makinede çalışıyorsa `http://host.docker.internal:<port>` kullanabilirsiniz.
    - API key: Önceki adımlarda kopyaladığınız API anahtarı.

3. RAGFlow çalışan modelleri türe göre gruplandırarak bağlam uzunluklarıyla birlikte listeler. Eklemek istemediğiniz modellerin onay kutusunu temizleyin. GPUStack bir modelin görüntü kabul edip etmediğini bildirmediğinden, `qwen2.5-vl-3b-instruct` gibi görsel modelleri `Chat + Vision` olarak ayarlayın:

    ![RAGFlow'da GPUStack bağlantısı](../assets/integrations/integration-ragflow-connect-gpustack.png)

    Bağlantı başarısız olursa RAGFlow nedeni formun altında gösterir:

    - `GPUStack rejected this API key.`: API anahtarını `Access Control` > `API Keys` altında kontrol edin.
    - `Cannot reach this server from RAGFlow.`: URL ve bağlantı noktasını RAGFlow kapsayıcısının içinden kontrol edin. Örneğin, RAGFlow `docker` dizininde aşağıdaki komutu çalıştırın; `"ok"` döndürmelidir:

        ```bash
        docker compose exec ragflow-cpu curl http://your-gpustack-url/healthz
        ```

4. Listenin altındaki `Add 6 models` düğmesine tıklayın (sayı, seçiminize göre değişir). Eklenen modeller örnekte görünür:

    ![RAGFlow'da model ekleme](../assets/integrations/integration-ragflow-add-model.png)

    Daha sonra GPUStack'te başka modeller dağıtırsanız, bunları bulup eklemek için örnekteki `Refresh` düğmesine tıklayın.

5. `Set default models` düğmesine tıklayın ve her tür için bir model seçin. Her seçim hemen kaydedilir:

    ![RAGFlow'da varsayılan modeller](../assets/integrations/integration-ragflow-set-default-models.png)

## Modelleri Kullanma { #using-the-models }

Artık modelleri `Dataset` ve `Chat` içinde kullanabilirsiniz. Basit bir örnek:

1. `Dataset` bölümüne gidin ve bir veri kümesi oluşturun. Varsayılan gömme modeli zaten seçilidir:

    ![RAGFlow'da veri kümesi oluşturma](../assets/integrations/integration-ragflow-create-dataset.png)

2. `Add file` > `Upload file` düğmesine tıklayın, `Parse on creation` seçeneğini açın, dosyanızı seçin ve `Save` düğmesine tıklayın. Ayrıştırma bittiğinde dosya, parça sayısını gösterir:

    ![RAGFlow'da dosya yükleme](../assets/integrations/integration-ragflow-upload-file.png)

3. `Retrieval testing` bölümüne gidin, `Rerank model` değerini `bge-reranker-v2-m3` olarak ayarlayın, bir soru girin ve `Run` düğmesine tıklayın:

    ![RAGFlow'da yeniden sıralama modeli](../assets/integrations/integration-ragflow-set-rerank-model.png)

4. `Chat` içinde bir sohbet oluşturup açın. Sağ üst köşedeki ayarlar simgesine tıklayın, `Datasets` altından veri kümesini seçin, `Model` altındaki sohbet modelini kontrol edin ve `Save` düğmesine tıklayın:

    ![RAGFlow'da sohbet ayarları](../assets/integrations/integration-ragflow-create-assistant.png)

    `Retrieval testing` bölümünde seçilen yeniden sıralama modeli sohbetlere uygulanmaz. Bir sohbetin aramasını yeniden sıralamak için aynı ayarlarda `Advanced settings` > `Naive` bölümünü genişletin, `Rerank model` değerini `bge-reranker-v2-m3` olarak ayarlayın ve `Save` düğmesine tıklayın.

5. Bir mesaj gönderin. Artık modelle etkileşime geçebilir ve veri kümesini sorgulayabilirsiniz:

    ![RAGFlow'da modelle sohbet](../assets/integrations/integration-ragflow-chat-with-model.png)

6. Yanıtların sesli okunması için `Advanced settings` altında `Text to speech` seçeneğini açın ve `Save` düğmesine tıklayın. Bir yanıtın üzerine gelin ve hoparlör simgesine tıklayın; ses çalarken simge duraklatma simgesine dönüşür:

    ![RAGFlow'da yanıtı sesli okuma](../assets/integrations/integration-ragflow-read-aloud.png)

7. Sesli giriş için mesaj kutusundaki mikrofon simgesine tıklayın, konuşun ve durdurmak için simgeye yeniden tıklayın. RAGFlow kaydı varsayılan Speech to text modeliyle metne dönüştürür ve mesajınız olarak gönderir.

8. Bir görüntü hakkında sohbet etmek için sohbet ayarlarında `Model` değerini `qwen2.5-vl-3b-instruct` olarak değiştirin ve yanıtlara ilgisiz belgelerden alıntılar karışmaması için `Datasets` alanını temizleyin. `Save` düğmesine tıklayın, yeni bir konuşma başlatın, ataş simgesiyle bir görüntü ekleyin ve sorunuzu gönderin:

    ![RAGFlow'da görsel modelle sohbet](../assets/integrations/integration-ragflow-chat-with-vlm.png)
