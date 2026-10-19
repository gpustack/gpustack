---
translation_status: translated
---

# NewAPI ile entegrasyon { #integrate-with-newapi }

NewAPI, yerel olarak dağıtılan LLM'leri, gömme modellerini, yeniden sıralamayı, Konuşmadan Metne ve Metinden Konuşmaya özelliklerini kurumsal çalışanlar için birleşik bir OpenAI uyumlu API ağ geçidinde toplamak için GPUStack ile entegre olabilir.

## Modelleri GPUStack'ta Dağıtma { #deploying-models-in-gpustack }

1. GPUStack kullanıcı arayüzünde, `Deployments` sayfasına gidin ve ihtiyacınız olan modelleri dağıtmak için `Deploy Model`'e tıklayın. İşte bazı örnek modeller:

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![gpustack-modelleri](../assets/integrations/integration-gpustack-models.png)

2. Modelin İşlemlerinde, bu modele nasıl entegre edileceğini görmek için `API Access Info`'ı açın.

![gpustack-api-erişim-bilgisi](../assets/integrations/integration-gpustack-api-access-info.png)

## GPUStack'te bir API Anahtarı oluşturun { #create-an-api-key-in-gpustack }

1. GPUStack'te `Access Control` > `API Keys` sayfasına gidin, ardından `New API Key`'ye tıklayın.

2. Adı girin ve ardından `Save`'a tıklayın.

3. API anahtarını kopyalayın ve daha sonra kullanmak üzere kaydedin.

## GPUStack'i NewAPI'ye entegre etme { #integrating-gpustack-into-newapi }

1. NewAPI Yönetim Konsolu'na erişin, sol kenar çubuğunda `Channels` seçeneğine gidin ve `Add Channel` seçeneğine tıklayın

2. Kanal parametrelerini aşağıdaki gibi yapılandırın:

- **Type**: `OpenAI` veya `Custom` seçeneğini seçin (GPUStack, standart OpenAI uyumlu ve Anthropic uç noktalar sağlar).

- **Ad**: Genellikle model kimliğiyle aynı olan bir Kanal adı girin; örneğin, `qwen3-8b`.

- **Temel URL (Temel URL)**: `http://your-gpustack-url`, URL, GPUStack'in erişim URL'sine veya hem OpenAI uyumlu hem de Anthropic alan adına işaret etmelidir (newapi otomatik olarak ekleyeceği için v1'i kaldırın). NewAPI bir Docker konteynerinde çalışıyorsa, konteynerin dahili ağını ifade ettiğinden `localhost` kullanmayın. gpustack sunucusu tls ile başlıyorsa https'yi de kullanabilir

- **Key (密钥)**: Önceki adımda kopyaladığınız GPUStack API Anahtarını girin.

- **Models (模型)**: GPUStack'te dağıtılan model adlarını tam olarak girin veya seçin (ör. `qwen3-8b`, `bge-m3`, `bge-reranker-v2-m3`). Model adlarının GPUStack'in dağıtım adlarıyla eşleştiğinden emin olun.

![newapi-ekleme-kanalı](../assets/integrations/newapi1.png)
![newapi-ekleme-kanalı](../assets/integrations/newapi2.png)
![newapi-ekleme-kanalı](../assets/integrations/newapi3.png)
![newapi-ekleme-kanalı](../assets/integrations/newapi4.png)

3. Model fiyatlandırmasını yapılandırın.

![newapi-ekleme-kanalı](../assets/integrations/newapi5.png)
![newapi-ekleme-kanalı](../assets/integrations/newapi6.png)
![newapi-ekleme-kanalı](../assets/integrations/newapi7.png)

4. NewAPI ile GPUStack arasındaki bağlantıyı doğrulamak için kanal listesinde `Test` tuşuna tıklayabilirsiniz.

![newapi-ekleme-kanalı](../assets/integrations/newapi8.png)

---

## NewAPI'deki Entegre Modelleri Doğrulama ve Kullanma { #verifying-and-using-the-integrated-models-in-newapi }

1. NewAPI'de `api keys` sayfasına gidin, `create api key`'e tıklayın

* not: API anahtarı, kanal yapılandırma grubuyla aynı grubu seçmelidir, aksi takdirde modele erişemezsiniz

2. Yeni oluşturulan NewAPI anahtarını kopyalayın (`sk-` ile başlar).

3. Artık tüm GPUStack destekli modellere NewAPI'nin birleştirilmiş uç noktası aracılığıyla erişebilirsiniz:

- **Birleşik Temel URL**: `http://your-newapi-url/v1`
- **Birleşik API Anahtarı**: `sk-xxxxxx` (Yeni API anahtarınız)

`curl` kullanan örnek istek:

```bash
curl http://your-newapi-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-newapi-token" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
