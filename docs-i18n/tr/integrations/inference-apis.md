---
translation_status: translated
---

# Çıkarım API'leri { #inference-apis }

## OpenAI Uyumlu API'ler { #openai-compatible-apis }

GPUStack, `/v1` uç noktasında [OpenAI uyumlu API'ler](https://platform.openai.com/docs/api-reference) sağlar.

GPUStack'te dağıtılan modelleri, OpenAI uyumlu API'yi destekleyen herhangi bir uygulama veya çerçeveyle, yalnızca GPUStack'in OpenAI uyumlu uç noktasına işaret ederek entegre edebilir ve kullanabilirsiniz.

### Desteklenen Uç Noktalar { #supported-endpoints }

Aşağıdaki API uç noktaları desteklenir:

- [x] [Modelleri Listele](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Tamamlama Oluştur](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Sohbet Tamamlama Oluştur](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Yanıt Oluştur](https://platform.openai.com/docs/api-reference/responses/create)
- [x] [Gömmeler Oluştur](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Resim Oluştur](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Resim Düzenleme Oluştur](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Konuşma Oluştur](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Transkripsiyon Oluştur](https://platform.openai.com/docs/api-reference/audio/createTranscription)

### Kullanım { #usage }

#### cURL Örneği { #curl-example }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## Anthropic Uyumlu API'ler { #anthropic-compatible-apis }

GPUStack, Anthropic uyumlu [`/v1/messages` API](https://platform.claude.com/docs/en/api/messages/create)'yi sağlar.

### Kullanım { #usage_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "max_tokens": 1024
  }'
```

## Jina Uyumlu Yeniden Sıralama API'si { #jina-compatible-rerank-api }

Erişimle Artırılmış Üretim (RAG) bağlamında yeniden sıralama, alınan belgelerden veya bilgi kaynaklarından en uygun bilgilerin kullanıcıya sunulmadan veya yanıt oluşturmak için kullanılmadan önce seçilmesi sürecini ifade eder.

OpenAI uyumlu API'lerin **bir `rerank` uç noktası sağlamadığını** unutmayın. Bu boşluğu doldurmak için GPUStack, `/v1/rerank` yolunda [Jina uyumlu bir Rerank API](https://jina.ai/reranker/) sağlar.

### Kullanım { #usage_2 }

#### cURL Örneği { #curl-example_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "bge-reranker-v2-m3",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

Örnek çıktı:

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
      },
      "index": 2,
      "relevance_score": 1.951932668685913
    },
    {
      "document": {
        "text": "it is a bear"
      },
      "index": 1,
      "relevance_score": -3.7347371578216553
    },
    {
      "document": {
        "text": "hi"
      },
      "index": 0,
      "relevance_score": -6.157620906829834
    }
  ],
  "usage": {
    "prompt_tokens": 69,
    "total_tokens": 69
  }
}
```

## System One Karar API'si { #system-one-decision-api }

Karar modelleri — `decision` kategorisindeki dağıtımlar — metin üretmek yerine System One (Jev) karar sözleşmesine göre yanıt verir: bir state ve bir dizi soru verildiğinde tamamlama yerine tiplendirilmiş yanıtlar döndürür. GPUStack bu sözleşmeyi `/v1/systemone` yolunda sunar.

`model` alanı zorunludur ve `/v1/chat/completions` ile tam olarak aynı şekilde model rotası adını alır. GPUStack modeli çözer ve istek gövdesinin geri kalanını ve tüm yanıtı karar motoruna dokunulmadan iletir.

### Kullanım { #usage_3 }

#### cURL Örneği { #curl-example_2 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/systemone \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
      "model": "qwen3-decision-8b",
      "state": {
          "messages": [
              { "role": "user", "content": "Write a 200-word poem in Hungarian." }
          ]
      },
      "questions": {
          "model_selection": {
              "type": "choice",
              "instructions": "Pick the model best suited.",
              "criteria": {
                  "qwen3-8b": "Strong multilingual generation.",
                  "llama3-8b": "Faster, weaker multilingual."
              }
          }
      }
  }' | jq
```

Örnek çıktı:

```json
{
  "model": "qwen3-decision-8b",
  "usage": {
    "input_tokens": 189,
    "output_tokens": 12
  },
  "answers": {
    "model_selection": {
      "type": "choice",
      "choice": "qwen3-8b",
      "confidence": 0.93,
      "probabilities": {
        "qwen3-8b": 0.93,
        "llama3-8b": 0.07
      }
    }
  }
}
```

`state`, kararın verildiği durumdur — düz metin veya sohbet mesajları gibi yapılandırılmış JSON. `questions` altındaki her giriş bir soru kimliğini soru tanımına eşler ve üç tür gelenekseldir:

| Tür | Anlamı | `criteria` | Yanıt alanı |
| --- | --- | --- | --- |
| `choice` | Bir seçenek seçin | Seçenek adı → açıklama | `choice` — seçilen seçeneğin adı |
| `score` | Durumu sıralı düzeylere göre puanlayın | Düzey açıklamalarının sıralı listesi | `score` — düzey dizini |
| `noul` | İkili (evet/hayır) soru | `true` / `false` taraflarının isteğe bağlı açıklamaları | `noul` — bir boole değeri |

Bir sorunun yanıtı `answers` içinde aynı kimlikle döner ve sonuçla birlikte `confidence` ve `probabilities` taşır. Sözleşmedeki soru seti açıktır — motorlar başka alanlar kabul edebilir — ve GPUStack yalnızca `model` alanını doğrular, geri kalanı olduğu gibi iletir. Karar modellerinin dağıtımı için [Karar Modellerini Kullanma](../using-models/using-decision-models.md) bölümüne bakın.

## Diğer API'ler { #other-apis }

Diğer API türleri için GPUStack, bir modeli dağıtırken **Generic Proxy** özelliğini etkinleştirmenize olanak tanır.

Etkinleştirildiğinde hedef modele ulaşmanın iki yolu vardır:

**Yol tabanlı form (önerilir)** — model rota kimliğini doğrudan URL'de `/model/proxy/<model_route_id>/<upstream-path>` olarak kodlayın. Ekstra başlığa gerek yoktur:

```bash
export GPUSTACK_API_KEY=your_api_key
# Assume the model route id is 42.
curl http://your_gpustack_server_url/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

**Başlık tabanlı form (kullanımdan kaldırıldı)** — `/model/proxy/<upstream-path>`'ye gönderin ve `X-GPUStack-Model` başlığı veya JSON gövdesindeki `model` alanı aracılığıyla hedef modeli belirtin:

```bash
curl http://your_gpustack_server_url/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

Başlığa dayalı form, geriye dönük uyumluluk amacıyla tutuldu ve gelecekteki bir sürümde kaldırılacaktır; yeni entegrasyonlar yol tabanlı formu kullanmalıdır.

Daha fazla ayrıntı için bkz. [Genel Proxy'yi Etkinleştirme](../user-guide/model-deployment-management.md#enable-generic-proxy).
