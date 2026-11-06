---
translation_status: translated
---

# MaxKB ile entegrasyon { #integrate-with-maxkb }

MaxKB, bilgiye dayalı yapay zeka asistanları oluşturmak için yerel olarak dağıtılan **LLM'lerden, gömme modellerinden ve yeniden sıralama modellerinden** yararlanmak için GPUStack ile entegre olabilir.

## Modelleri Dağıtma { #deploying-models }

1. GPUStack kullanıcı arayüzünde, `Deployments` sayfasına gidin ve ihtiyacınız olan modelleri dağıtmak için `Deploy Model`'e tıklayın. İşte bazı örnek modeller:

* `qwen3.5-35b-a3b`

  ![](../assets/integrations/maxkb/01.png)
  ![](../assets/integrations/maxkb/02.png)

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/03.png)
  ![](../assets/integrations/maxkb/04.png)

* `qwen3-reranker-4b`

  ![](../assets/integrations/maxkb/05.png)
  ![](../assets/integrations/maxkb/06.png)

2. Dağıtımdan sonra modeli **Playground**'de test edebilirsiniz.

![](../assets/integrations/maxkb/07.png)
![](../assets/integrations/maxkb/09.png)
![](../assets/integrations/maxkb/08.png)

## Model Erişim Bilgilerini Alın { #obtain-model-access-information }

1. GPUStack kenar çubuğunda **Routes** sayfasını açın.

2. Rotanın yanındaki **Diğer eylemler menüsüne** tıklayın ve **API Erişim Bilgisi**'yi seçin.

![](../assets/integrations/maxkb/10.png)
![](../assets/integrations/maxkb/11.png)

Aşağıdaki bilgileri kaydedin:

```
Base URL
Model Name
API Key
```

Örnek:

```
Base URL: http://your-gpustack-url/v1

Model Name:
qwen3.5-35b-a3b
qwen3-embedding-4b
qwen3-reranker-4b

API Key:
gpustack_xxxxxxxxxxxxx
```

!!! note

    Kullanıcı arayüzündeki talimatları izleyerek API Anahtarı oluşturabilirsiniz.

## MaxKB'yi dağıtın { #deploy-maxkb }

MaxKB, Docker kullanılarak dağıtılabilir:

```bash
docker run -d \
  --name maxkb \
  --restart always \
  -p 8080:8080 \
  -v ~/.maxkb:/opt/maxkb \
  1panel/maxkb
```

Varsayılan kimlik bilgileri:

```
admin / MaxKB@123..
```

![](../assets/integrations/maxkb/50.png)

İlk kez giriş yaptıktan sonra şifreyi değiştirmek için yönlendirmeyi takip edin.

## GPUStack'i MaxKB'ye entegre etme { #integrating-gpustack-into-maxkb }

1. MaxKB kullanıcı arayüzünde, üst gezinme çubuğunda **Model** seçeneğine gidin.

![](../assets/integrations/maxkb/51.png)

2. **Model Ekle** seçeneğine tıklayın ve modeli yapılandırın.

![](../assets/integrations/maxkb/52.png)
![](../assets/integrations/maxkb/53.png)
![](../assets/integrations/maxkb/54.png)

Modeli yapılandırırken:

* **Temel Model**: GPUStack'te dağıtılan model adıyla eşleşmelidir.
* **API URL'si**: `http://your-gpustack-url/v1`
* **API Key**: GPUStack'ta oluşturulan API anahtarı.

!!! note

`API URL` ve `API Key` alanları, **Base Model değerini girip Enter tuşuna bastıktan sonra** görünür.

3. Aynı yöntemi kullanarak gömme ve yeniden sıralama modellerini ekleyin:

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/56.png)

* `qwen3-reranker-4b`

  **qwen3-reranker-4b** için **Generic Proxy**'i etkinleştirin.

  ![](../assets/integrations/maxkb/55.png)

  MaxKB aşağıdaki uç noktayı kullandığından bu gereklidir:

  ```
  /v2/rerank
  ```

  ![](../assets/integrations/maxkb/57.png)

Yapılandırma sonrasında modeller model listesinde görünmelidir.

![](../assets/integrations/maxkb/58.png)

## Bilgi Tabanı Oluşturun { #create-a-knowledge-base }

1. **Bilgi** sayfasına gidin.

2. **Oluştur**'e tıklayın ve **Web Bilgisi**'yi seçin.

![](../assets/integrations/maxkb/59.png)

3. Bir dokümantasyon URL'si veya başka bir veri kaynağı girin. MaxKB, içeriği otomatik olarak tarayacak ve ayrıştıracaktır.

![](../assets/integrations/maxkb/60.png)

Tarama tamamlandıktan sonra:

![](../assets/integrations/maxkb/61.png)

## Bir AI Aracısı Oluşturun { #create-an-ai-agent }

1. **Acente** sayfasına gidin.

2. Yeni bir temsilci oluşturmak için **Oluştur**'e tıklayın.

![](../assets/integrations/maxkb/62.png)
![](../assets/integrations/maxkb/63.png)

3. Aracıyı şununla yapılandırın:

* Sohbet modeli
* Bilgi tabanı
* Alma ayarları

4. Aracıyı etkinleştirmek için **Yayınla**'e tıklayın.

![](../assets/integrations/maxkb/64.png)

## Bilgi Bankasıyla Sohbet Edin { #chat-with-the-knowledge-base }

Asistanla etkileşime geçmek için sohbet arayüzünü açın.

![](../assets/integrations/maxkb/65.png)
![](../assets/integrations/maxkb/66.png)

Asistan artık bağlı bilgi tabanına ve GPUStack'te dağıtılan modellere dayalı olarak soruları yanıtlayabilir.
