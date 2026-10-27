---
translation_status: translated
---

# RAGFlow ile entegrasyon { #integrate-with-ragflow }

RAGFlow, yerel olarak dağıtılan LLM'lerden, gömme modellerinden, yeniden sıralamadan, Konuşmadan Metne ve Metinden Konuşmaya yeteneklerinden yararlanmak için GPUStack ile entegre olabilir.

## Modelleri Dağıtma { #deploying-models }

1. GPUStack kullanıcı arayüzünde, `Deployments` sayfasına gidin ve ihtiyacınız olan modelleri dağıtmak için `Deploy Model`'e tıklayın. İşte bazı örnek modeller:

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![gpustack-modelleri](../assets/integrations/integration-gpustack-models.png)

2. Modelin İşlemlerinde, bu modele nasıl entegre edileceğini görmek için `API Access Info`'ı açın.

![gpustack-api-erişim-bilgisi](../assets/integrations/integration-gpustack-api-access-info.png)

## API Anahtarı Oluşturun { #create-an-api-key }

1. `Access Control` > `API Keys` sayfasına gidin ve ardından `New API Key`'ye tıklayın.

2. Adı girin ve ardından `Save`'a tıklayın.

3. API anahtarını kopyalayın ve daha sonra kullanmak üzere kaydedin.

## GPUStack'i RAGFlow'a entegre etme { #integrating-gpustack-into-ragflow }

1. RAGFlow kullanıcı arayüzüne erişin, sağ üst köşeye gidin ve avatara tıklayın, `Model Providers > GPUStack`'yi seçin, ardından `Add the model`'i seçin ve doldurun:

- Model tipi: Modele göre model tipini seçin.

- Model adı: Ad, GPUStack'te dağıtılan model adıyla eşleşmelidir.

- Temel URL: `http://your-gpustack-url/v1`, URL, yolu içermemeli ve konteynerin dahili ağına atıfta bulunduğundan `localhost` kullanmamalıdır. Özel bir bağlantı noktası kullanıyorsanız onu eklediğinizden emin olun. Ayrıca URL'ye RAGFlow kapsayıcısının içinden erişilebildiğinden emin olun (bunu `curl` ile test edebilirsiniz).

- API-Key: Önceki adımlardan kopyaladığınız API anahtarını girin.

- Maksimum Belirteçler: Geçerli model yapılandırması tarafından desteklenen maksimum belirteçleri girin.

Modeli eklemek için `OK`'e tıklayın:

![ragflow-ekleme modeli](../assets/integrations/integration-ragflow-add-model.png)

2. Gerektiğinde diğer modelleri ekleyin, ardından `Set default models`'da eklenen modelleri seçin ve kaydedin:

![ragflow-set-varsayılan-modeller](../assets/integrations/integration-ragflow-set-default-models.png)

Artık `Chat` ve `Knowledge Base`'deki modelleri kullanabilirsiniz, işte basit bir durum:

1. Yeni bir bilgi tabanı oluşturmak ve dosyanızı eklemek için `Knowledge base`'a gidin:

![ragflow-oluşturma-bilgi-tabanı](../assets/integrations/integration-ragflow-create-knowledge-base.png)

2. `Retrieval testing`'e gidin ve yeniden sıralama modelini `bge-reranker-v2-m3` olarak ayarlayın:

![ragflow-set-yeniden sıralama modeli](../assets/integrations/integration-ragflow-set-rerank-model.png)

3. `Chat`'de bir asistan oluşturun, önceden oluşturulmuş bilgi tabanını bağlayın ve bir sohbet modeli seçin:

![ragflow-oluşturma asistanı](../assets/integrations/integration-ragflow-create-assistant.png)

4. Bir sohbet oturumu oluşturun — artık modelle etkileşime girebilir ve bilgi tabanını sorgulayabilirsiniz:

![modelle-ragflow-sohbet](../assets/integrations/integration-ragflow-chat-with-model.png)

5. Asistanı düzenleyin ve modeli `qwen2.5-vl-3b-instruct` olarak değiştirin. Kaydettikten sonra yeni bir sohbet oluşturun ve çok modlu girişi etkinleştirmek için bir resim yükleyin:

![vlm ile ragflow-sohbet](../assets/integrations/integration-ragflow-chat-with-vlm.png)
