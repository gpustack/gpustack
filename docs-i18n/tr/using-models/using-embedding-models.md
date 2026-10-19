---
translation_status: translated
---

# Gömme Modellerini Kullanma { #using-embedding-models }

**Metin gömmeleri**, metnin anlamsal içeriğini yansıtan sayısal temsillerdir; makinelerin farklı metinler arasındaki ilişkileri ve benzerlikleri anlamasını sağlar. Metni sürekli bir uzaydaki vektörlere dönüştürür; benzer anlamlı metinler bu uzayda birbirine daha yakın yer alır. Metin gömmeleri doğal dil işleme, bilgi erişimi ve öneri sistemlerinde yaygın olarak kullanılır.

Bu kılavuzda GPUStack'te gömme modellerini dağıtmayı ve bu modellerle metin gömmeleri oluşturmayı göstereceğiz.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- GPUStack kurulu ve çalışıyor. Değilse, [Hızlı Başlangıç ​​Kılavuzu](../quickstart.md)'na bakın.
- Model dosyalarını indirmek için Hugging Face'e erişim.

## Adım 1: Modeli Dağıtın { #step-1-deploy-the-model }

Modeli Katalogdan dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Catalog` sayfasına gidin.
2. Model listesi sayfasında, `Embedding` ile filtrelemek için açılır menüyü kullanın.
3. Model açıklamasını, maksimum bağlam uzunluğunu ve desteklenen boyutları inceleyin.

![Modeli Dağıt](../assets/using-models/using-embedding-models/deploy-model.png)

Dağıtımdan sonra model dağıtımının durumunu `Deployments` sayfasından izleyebilirsiniz.

![Model Listesi](../assets/using-models/using-embedding-models/model-list.png)

## Adım 2: API Anahtarı Oluşturun { #step-2-generate-an-api-key }

Metin gömmeleri oluşturmak için GPUStack API'sini kullanacağız; bunun için bir API anahtarı gereklidir:

1. `Access Control` > `API Keys` sayfasına gidin.
2. `New API Key` düğmesine tıklayın.
3. API anahtarı için bir ad girin ve `Save` düğmesine tıklayın.
4. Oluşturulan API anahtarını kopyalayın. API anahtarını yalnızca bir kez görüntüleyebilirsiniz, bu nedenle güvenli bir şekilde kaydettiğinizden emin olun.

## Adım 3: Metin Gömmeleri Oluşturun { #step-3-generate-text-embeddings }

Modeli dağıtıp bir API anahtarı aldıktan sonra GPUStack API'si aracılığıyla metin gömmeleri oluşturabilirsiniz. `curl` kullanan bir örnek:

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/embeddings \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${GPUSTACK_API_KEY}" \
-d '{
  "model": "qwen3-embedding-4b",
  "input": "The food was delicious and the waiter...",
  "encoding_format": "float"
}'
```

`<your-server-url>`'i GPUStack sunucunuzun URL'siyle ve `<your-api-key>`'i önceki adımda oluşturduğunuz API anahtarıyla değiştirin.

Örnek yanıt:

```json
{
  "data": [
    {
      "embedding": [
        -0.012189436703920364, 0.016934078186750412, 0.003965042531490326,
        -0.03453584015369415, -0.07623119652271271, -0.007116147316992283,
        0.11278388649225235, 0.019714849069714546, 0.010370955802500248,
        -0.04219457507133484, -0.029902394860982895, 0.01122555136680603,
        0.022912170737981796, 0.031186765059828758, 0.006303929258137941,
        # ... additional values
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "qwen3-embedding-4b",
  "object": "list",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```
