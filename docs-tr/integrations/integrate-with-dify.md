---
translation_status: translated
---

# Dify ile entegrasyon { #integrate-with-dify }

Dify, yerel olarak dağıtılan LLM'lerden, gömme modellerinden, yeniden sıralamadan, görüntü oluşturmadan, Konuşmadan Metne ve Metinden Konuşmaya yeteneklerinden yararlanmak için GPUStack ile entegre olabilir.

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

## GPUStack'i Dify'a entegre etme { #integrating-gpustack-into-dify }

1. Dify kullanıcı arayüzüne erişin, sağ üst köşeye gidin ve `PLUGINS`'e tıklayın, `Install from Marketplace`'i seçin, GPUStack eklentisini arayın ve yüklemeyi seçin.

![dify-install-gpustack-eklentisi](../assets/integrations/integration-dify-install-gpustack-plugin.png)

2. Kurulduktan sonra `Settings > Model Provider > GPUStack`'e gidin, ardından `Add Model`'i seçin ve doldurun:

- Model Tipi: Modele göre model tipini seçin.

- Model Adı: Ad, GPUStack'te dağıtılan model adıyla eşleşmelidir.

- Sunucu URL'si: `http://your-gpustack-url`, konteynerin dahili ağını ifade ettiği için `localhost` kullanmayın. Özel bir bağlantı noktası kullanıyorsanız onu eklediğinizden emin olun. Ayrıca URL'ye Dify konteynerinin içinden erişilebildiğinden emin olun (bunu `curl` ile test edebilirsiniz).

- API Anahtarı: Önceki adımlardan kopyaladığınız API anahtarını girin.

Modeli eklemek için `Save`'e tıklayın:

![dify-ekleme-modeli](../assets/integrations/integration-dify-add-model.png)

Gerektiğinde diğer modelleri ekleyin, ardından `System Model Settings`'da eklenen modelleri seçin ve kaydedin:

![dify-sistem-model-ayarları](../assets/integrations/integration-dify-system-model-settings.png)

Artık `Studio` ve `Knowledge`'deki modelleri kullanabilirsiniz, işte basit bir durum:

1. Bilgi oluşturmak ve belgelerinizi yüklemek için `Knowledge`'a gidin:

![dify-oluşturma-bilgisi](../assets/integrations/integration-dify-create-knowledge.png)

2. Parça Ayarlarını ve Alma Ayarlarını yapılandırın. Belge gömmeleri oluşturmak için gömme modelini, alma sıralamasını gerçekleştirmek için ise yeniden sıralama modelini kullanın.

![dify-set-yerleştirme-ve-yeniden-sıralama-modeli](../assets/integrations/integration-dify-set-embedding-and-rerank-model.png)

3. Belgeleri başarıyla içe aktardıktan sonra `Studio`'da bir uygulama oluşturun, önceden oluşturulan bilgiyi ekleyin, sohbet modelini seçin ve onunla etkileşim kurun:

![modelle-sohbet-dify](../assets/integrations/integration-dify-chat-with-model.png)

4. Modeli `qwen2.5-vl-3b-instruct` olarak değiştirin, önceden eklenen bilgi tabanını kaldırın, `Vision`'i etkinleştirin ve çok modlu girişi etkinleştirmek için sohbete bir resim yükleyin:

![vlm ile-sohbet-dify](../assets/integrations/integration-dify-chat-with-vlm.png)
