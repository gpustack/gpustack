---
translation_status: translated
---

# Araç Çağırma ile Çıkarım { #inference-with-tool-calling }

Araç çağırma, modelleri harici araçlara ve sistemlere bağlamanıza olanak tanır. Bu, AI asistanlarını yeteneklerle güçlendirmek veya uygulamalarınız ile modeller arasında derin entegrasyonlar oluşturmak gibi birçok şey için faydalıdır.

Bu eğitimde, yapay zekanızın yeteneklerini genişletmek için GPUStack içinde araç çağırmayı nasıl ayarlayacağınızı ve kullanacağınızı öğreneceksiniz.

!!! note

    1. Araç çağırma, [vLLM](../user-guide/built-in-inference-backends.md#vllm) ve [SGLang](../user-guide/built-in-inference-backends.md#sglang) çıkarım arka uçlarında desteklenir.
    2. Araç çağırma temelde istem mühendisliğiyle sağlanır; bu yeteneği kazanabilmeleri için modellerin uygun şablonlarla eğitilmiş olması gerekir. Bu nedenle tüm LLM'ler araç çağırmayı desteklemez.

## Önkoşullar { #prerequisites }

Devam etmeden önce aşağıdakilerden emin olun:

- GPUStack kurulu ve çalışıyor.
- GPU'lu bir Linux çalışan düğümü mevcuttur. Bu eğitim için model olarak [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) kullanacağız. Model, en az 18 GB VRAM'e sahip bir GPU gerektirir.
- Model dosyalarını indirmek için Hugging Face'e erişim.

## Adım 1: Modeli Dağıtın { #step-1-deploy-the-model }

### Katalogdan Dağıt { #deploy-from-catalog }

Araç çağırmayı destekleyen LLM'ler katalogda `tools` özelliğiyle işaretlenmiştir. Katalogdan böyle bir model seçtiğinizde, araç çağırma varsayılan olarak etkinleştirilir.

### Llama-box Kullanarak Özel Dağıtım Örneği { #example-of-custom-deployment-using-llama-box }

Llama-box kullanarak GGUF modellerini dağıttığınızda, onu destekleyen modeller için araç çağırma varsayılan olarak etkindir.

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin ve `Deploy Model` düğmesine tıklayın. Açılır menüde modelinizin kaynağı olarak `Hugging Face`'yi seçin.
2. Modelleri GGUF formatına göre filtrelemek için `GGUF` onay kutusunu etkinleştirin.
3. `Qwen/Qwen2.5-7B-Instruct-GGUF` modelini bulmak için arama çubuğunu kullanın.
4. Modeli dağıtmak için `Save` düğmesine tıklayın.

![GGUF Modelini Dağıt](../assets/tutorials/inference-with-tool-calling/deploy-model-gguf.png)

### vLLM Kullanarak Özel Dağıtım Örneği { #example-of-custom-deployment-using-vllm }

Modelleri vLLM kullanarak dağıttığınızda, ek parametrelerle araç çağırmayı etkinleştirmeniz gerekir.

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin ve `Deploy Model` düğmesine tıklayın. Açılır menüde modelinizin kaynağı olarak `Hugging Face`'yi seçin.
2. `Qwen/Qwen2.5-7B-Instruct` modelini bulmak için arama çubuğunu kullanın.
3. Yapılandırmalarda `Advanced` bölümünü genişletin ve `Backend Parameters` bölümüne doğru aşağı kaydırın.
4. `Add Parameter` düğmesine tıklayın ve aşağıdaki parametreleri ekleyin:

- `--enable-auto-tool-choice`
- `--tool-call-parser=hermes`

5. Modeli dağıtmak için `Save` düğmesine tıklayın.

![Modeli Dağıt](../assets/tutorials/inference-with-tool-calling/deploy-model.png)

Dağıtımdan sonra modelin durumunu `Deployments` sayfasından izleyebilirsiniz.

## Adım 2: API Anahtarı Oluşturun { #step-2-generate-an-api-key }

Modelle etkileşim kurmak için GPUStack API'sini kullanacağız. Bunu yapmak için bir API anahtarı oluşturmanız gerekir:

1. `Access Control` > `API Keys` sayfasına gidin.
2. `New API Key` düğmesine tıklayın.
3. API anahtarı için bir ad girin ve `Save` düğmesine tıklayın.
4. Oluşturulan API anahtarını daha sonra kullanmak üzere kopyalayın.

## Adım 3: Çıkarım Yapın { #step-3-do-inference }

Dağıtılan model ve bir API anahtarı ile modeli GPUStack API aracılığıyla çağırabilirsiniz. Burada `curl` kullanan örnek bir komut dosyası verilmiştir (`<your-server-url>`'i GPUStack sunucu URL'nizle ve `<your-api-key>`'yi önceki adımda oluşturulan API anahtarıyla değiştirin):

```bash
export GPUSTACK_SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $GPUSTACK_SERVER_URL/v1-openai/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GPUSTACK_API_KEY" \
-d '{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'
```

Örnek yanıt:

```json
{
  "model": "qwen2.5-7b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "chatcmpl-tool-b99d32848b324eaea4bac5a5830d00b8",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 212,
    "total_tokens": 242,
    "completion_tokens": 30
  }
}
```
