---
translation_status: translated
---

# Karar Modellerini Kullanma { #using-decision-models }

**Karar modelleri**, metin üretmek yerine yapılandırılmış, tiplendirilmiş soruları yanıtlar. Bir *state* (kararın verildiği durum) ve bir dizi *question* verildiğinde karar modeli her soru için tiplendirilmiş bir sonuç döndürür: seçilen seçenek, bir düzey puanı veya evet/hayır yanıtı; genellikle güven değerleriyle birlikte. Tam bir üretimin israf olduğu durumlarda, örneğin model seçimi ve yönlendirme, sınıflandırma ve hafif değerlendirme görevlerinde kullanılır.

GPUStack'te karar modelleri, `/v1/systemone` uç noktasındaki System One (Jev) karar sözleşmesiyle sunulur. Bu kılavuzda GPUStack'te karar modellerinin nasıl dağıtılacağı ve kullanılacağı gösterilmektedir.

## Önkoşullar { #prerequisites }

Başlamadan önce şunlara sahip olduğunuzdan emin olun:

- GPUStack kurulu ve çalışıyor. Değilse [Hızlı Başlangıç Kılavuzu](../quickstart.md)'na bakın.
- Model dosyalarını indirmek için Hugging Face erişimi.

## Adım 1: Modeli Dağıtın { #step-1-deploy-the-model }

GPUStack şu anda karar API'sini sunan iki tür dağıtımı desteklemektedir:

### vLLM: Metin Üretim Modelini Yeniden Kullanma { #vllm-repurpose-a-text-generation-model }

vLLM `0.32.0` ve sonrası, System One karar sözleşmesini normal servis portlarında sunar. Herhangi bir metin üretim modelini her zamanki gibi `vLLM` arka ucuyla dağıtın, ardından modelin kategorisini `Decision` olarak değiştirin — GPUStack'e bu dağıtımın bir karar hizmeti olduğunu bildiren şey kategoridir.

1. `vLLM` arka ucuyla (`0.32.0` veya üzeri sürüm) bir metin üretim modeli dağıtın. [Model Dağıtım Yönetimi](../user-guide/model-deployment-management.md)'ne bakın.
2. Modeli düzenleyin ve model ayarlarında `Category` değerini `Decision` yapın.

![Model Kategorisi](../assets/using-models/using-decision-models/model-category.png)

### llama.cpp: Laya Karar GGUF'ini Sunma { #llamacpp-serve-a-laya-decision-gguf }

Laya karar GGUF dosyaları, genel bir omurga (modern-bert, qwen, llama) üzerinde bir karar meta verisi anahtarı taşır. GGUF'ü diğer GGUF modelleri gibi dağıtın; llama.cpp tarafından sunulur. GPUStack dağıtım sırasında karar meta verisini algılar ve modeli otomatik olarak `decision` kategorisine yerleştirir — bu anahtar olmasaydı aynı dosya yeniden sıralama veya gömme modeli olarak yanlış okunacaktı.

![Model Listesi](../assets/using-models/using-decision-models/model-list.png)

Hangi yolu seçerseniz seçin, dağıtım çalışmaya başladığında `Models` sayfasındaki `Category` sütunu `Decision` gösterir. Kategori elle de düzeltilebilir; kategori değerleri doğrulanır, bu yüzden hatalı yazılmış bir kategori sessizce kaydedilmek yerine reddedilir.

## Adım 2: Karar İsteyin { #step-2-ask-a-decision }

Karar istemek için UI'yı ya da API'yi kullanabilirsiniz.

### UI Üzerinden { #via-the-ui }

1. `Playground` > `Decision` sayfasına gidin veya `Models` sayfasındaki model satırında `Open in Playground` öğesine tıklayın.
2. Model seçiciden bir karar modeli seçin — yalnızca `decision` kategorisindeki modeller listelenir ve model, kararın tek parametresidir.
3. `State` düzenleyicisini kararın verildiği durumla doldurun (sohbet mesajlarından bir örnek ile başlatılır; düz metin veya yapılandırılmış JSON kabul eder) ve `Questions` düzenleyicisini sorularla doldurun (bir `choice` sorusuyla başlatılır; `+ choice`, `+ score` ve `+ noul` düğmeleriyle genişletilir).
4. `Run` öğesine tıklayın (veya `Ctrl`/`Cmd` + `Enter` tuşlarına basın) ve tiplendirilmiş yanıtları okuyun.

Düzenleyiciler ve yanıt kartları hakkında ayrıntılar için [Karar Playground](../user-guide/playground/decision.md) sayfasına bakın.

### API Üzerinden { #via-the-api }

Modele bir state ve soruları `POST /v1/systemone` ile gönderin (`API Keys` sayfasında oluşturulan bir API anahtarıyla kimlik doğrulaması yapın). `model` alanı zorunludur ve OpenAI uyumlu uç noktalarda olduğu gibi dağıtılan modelin adını (veya model rotası adını) taşır.

`curl` kullanan örnek bir betik:
```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/systemone \
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

`<your-server-url>` değerini GPUStack sunucunuzun URL'si ile, `<your-api-key>` değerini `API Keys` sayfasında oluşturulan bir API anahtarı ile değiştirin.

Örnek yanıt:

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

`questions` altındaki her giriş, bir soru kimliğini soru tanımına eşler ve üç geleneksel türden birini kullanır; bir sorunun yanıtı `answers` içinde aynı kimlikle geri döner:

| Tür | Anlamı | `criteria` | Yanıt alanı |
| --- | --- | --- | --- |
| `choice` | Bir seçenek seçin | Seçenek adı → açıklama | `choice` — seçilen seçeneğin adı |
| `score` | Durumu sıralı düzeylere göre puanlayın | Düzey açıklamalarının sıralı listesi | `score` — düzey dizini |
| `noul` | İkili (evet/hayır) soru | `true` / `false` taraflarının isteğe bağlı açıklamaları | `noul` — bir boole değeri |

GPUStack `model` alanını çözer ve istek gövdesinin geri kalanını ve tüm yanıtı, karar motoruna dokunulmadan olduğu gibi iletir — sözleşmedeki soru seti açıktır ve motorlar başka alanlar kabul edebilir.

## Sonraki Adımlar { #next-steps }

- Dağıtılmış bir karar modeli, bir rotanın [Karar Hizmeti Yönlendirme](../user-guide/model-route-management.md#decision-service-routing) politikasının karar hizmeti olarak görev yapabilir ve istek başına rota hedeflerini puanlayabilir.
- Uç noktaların tam başvurusu için [Çıkarım API'leri](../integrations/inference-apis.md) bölümüne bakın.
