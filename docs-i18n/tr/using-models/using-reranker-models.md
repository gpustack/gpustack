---
translation_status: translated
---

# Yeniden Sıralama Modellerini Kullanma { #using-reranker-models }

**Yeniden Sıralama Modelleri**, belirli bir sorguyla alaka düzeyine göre bir öğe listesinin sıralamasını iyileştirmek için tasarlanmış özel modellerdir. Bilgi alma ve arama sistemlerinde, kullanıcının amacını karşılama olasılığı daha yüksek olan öğelere öncelik vererek ilk arama sonuçlarını iyileştirmek için yaygın olarak kullanılırlar. Yeniden sıralama modelleri, arama motorları, öneri sistemleri ve soru yanıtlama görevleri gibi uygulamalarda hassasiyeti artırmak için ilk belge listesini alır ve öğeleri yeniden sıralar.

Bu kılavuzda GPUStack'te yeniden sıralama modellerinin nasıl dağıtılacağını ve kullanılacağını göstereceğiz.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- GPUStack kurulu ve çalışıyor. Değilse, [Hızlı Başlangıç ​​Kılavuzu](../quickstart.md)'na bakın.
- Model dosyalarını indirmek için Hugging Face'e erişim.

## Adım 1: Modeli Dağıtın { #step-1-deploy-the-model }

Modeli Katalogdan dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Catalog` sayfasına gidin.
2. Model listesi sayfasında, `Reranker` ile filtrelemek için açılır menüyü kullanın.
3. Model açıklamasını, maksimum bağlam uzunluğunu ve desteklenen boyutları inceleyin.

![Modeli Dağıt](../assets/using-models/using-reranker-models/deploy-model.png)

Dağıtımdan sonra model dağıtımının durumunu `Deployments` sayfasından izleyebilirsiniz.

![Model Listesi](../assets/using-models/using-reranker-models/model-list.png)

## Adım 2: API Anahtarı Oluşturun { #step-2-generate-an-api-key }

Modelle etkileşim kurmak için GPUStack API'sini kullanacağız. Bunu yapmak için bir API anahtarı oluşturmanız gerekir:

1. `Access Control` > `API Keys` sayfasına gidin.
2. `New API Key` düğmesine tıklayın.
3. API anahtarı için bir ad girin ve `Save` düğmesine tıklayın.
4. Oluşturulan API anahtarını kopyalayın. API anahtarını yalnızca bir kez görüntüleyebilirsiniz, bu nedenle güvenli bir şekilde kaydettiğinizden emin olun.

## 3. Adım: Yeniden Sıralama { #step-3-reranking }

Dağıtılan model ve bir API anahtarıyla, GPUStack API aracılığıyla bir belge listesini yeniden sıralayabilirsiniz. İşte `curl` kullanan örnek bir komut dosyası:

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-reranker-4b",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

`<your-server-url>`'i GPUStack sunucunuzun URL'siyle ve `<your-api-key>`'i önceki adımda oluşturduğunuz API anahtarıyla değiştirin.

Örnek yanıt:

```json
{
  "model": "qwen3-reranker-4b",
  "object": "list",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "hi",
        "multi_modal": null
      },
      "relevance_score": 0.9996911287307739
    },
    {
      "index": 2,
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "multi_modal": null
      },
      "relevance_score": 0.8206241726875305
    },
    {
      "index": 1,
      "document": {
        "text": "it is a bear",
        "multi_modal": null
      },
      "relevance_score": 0.7244728803634644
    }
  ],
  "usage": {
    "total_tokens": 51
  }
}
```
