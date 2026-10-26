---
translation_status: translated
---

# Cherry Studio ile entegrasyon { #integrate-with-cherry-studio }

Cherry Studio, yerel olarak barındırılan LLM'lerden, gömme modellerinden ve yeniden sıralama özelliklerinden yararlanmak için GPUStack ile entegre olur.

## Modelleri Dağıtma { #deploying-models }

1. GPUStack kullanıcı arayüzünde, `Deployments` sayfasına gidin ve ihtiyacınız olan modelleri dağıtmak için `Deploy Model`'e tıklayın. İşte bazı örnek modeller:

    - qwen3-instruct-2507
    - qwen2.5-vl-7b
    - bge-m3
    - bge-reranker-v2-m3

   ![GPUStack modelleri](../assets/integrations/integration-gpustack-models2.png)

2. Modelin Operasyonlarında, bu modelle nasıl entegre edileceğini görmek için `API Access Info`'ı açın:

   ![API Erişim Bilgisi](../assets/integrations/integration-gpustack-api-access-info2.png)

## API Anahtarı Oluşturun { #create-an-api-key }

1. `Access Control` > `API Keys` sayfasına gidin ve ardından `New API Key`'ye tıklayın.

2. Adı girin ve ardından `Save`'a tıklayın.

3. API anahtarını kopyalayın ve daha sonra kullanmak üzere kaydedin.

## GPUStack'i Cherry Studio'ya entegre etme { #integrating-gpustack-into-cherry-studio }

1. Cherry Studio'yu açın, `Settings` → `Model Provider`'e gidin, GPUStack'i bulun, etkinleştirin ve gösterildiği gibi yapılandırın:

    - `API Key`: Önceki adımlardan kopyaladığınız API anahtarını girin.

    - `API Host`: `API Access Info` panelinde `Access URL`.

   ![Cherry Studio GPUStack kurulumu](../assets/integrations/integration-cherrystudio-01.png)

2. GPUStack sağlayıcı yapılandırmasında "Yönet"e tıklayın ve ihtiyacınız olan modelleri etkinleştirin:

   ![Modelleri yönet](../assets/integrations/integration-cherrystudio-02.png)

   ![Modelleri etkinleştir](../assets/integrations/integration-cherrystudio-03.png)

3. (İsteğe bağlı) API'yi test edin:

   ![API'yi test edin](../assets/integrations/integration-cherrystudio-04.png)

Yapılandırma sonrasında Cherry Studio ana sayfasına dönün ve modellerinizi kullanmaya başlayın.

## LLM'leri Kullanmak { #using-llms }

![LLM'leri Kullan](../assets/integrations/integration-cherrystudio-05.png)

## Multimodal Modellerin Kullanımı { #using-multimodal-models }

1. Çok modlu bir model seçin:

   ![Çok modlu modeli seçin](../assets/integrations/integration-cherrystudio-13.png)

2. Çok modlu sorular sorun:

   ![Çok modlu Soru-Cevap](../assets/integrations/integration-cherrystudio-14.png)

   ![Çok modlu Soru-Cevap](../assets/integrations/integration-cherrystudio-15.png)

## Bilgi Bankası Soru-Cevaplarını Geliştirmek için Yerleştirmeleri ve Yeniden Sıralamayı Kullanın { #use-embeddings-and-reranking-to-improve-knowledge-base-qa }

1. Bilgi Bankası yapılandırma sayfasını açın:

   ![KB ayarları](../assets/integrations/integration-cherrystudio-06.png)

2. Bilgi tabanı ekleyin:

   ![KB](../assets/integrations/integration-cherrystudio-08.png) Ekle

3. Bilgi tabanına içerik ekleyin (örnek olarak “Notlar”ı kullanarak):

   ![KB içeriği ekle](../assets/integrations/integration-cherrystudio-09.png)

   ![KB içerik ayrıntıları](../assets/integrations/integration-cherrystudio-10.png)

4. Ana sayfaya dönün ve bilgi tabanını kullanın Soru-Cevap:

   ![KB Soru-Cevap](../assets/integrations/integration-cherrystudio-11.png)

   ![KB Soru-Cevap örneği](../assets/integrations/integration-cherrystudio-12.png)
