---
translation_status: translated
---

# Model Sağlayıcı Yönetimi { #model-provider-management }

GPUStack, Genel MaaS entegrasyonu sağlamak için Higress özelliklerinden yararlanır. `Model` - `Provider` sayfasında genel model hizmetlerinin entegrasyonunu yönetebilirsiniz.

Şu anda desteklenen sağlayıcılar şunlardır:

- AI360
- Microsoft Azure
- Baichuan Intelligent Technology
- Baidu AI Cloud
- Amazon Bedrock
- Anthropic Claude
- Cloudflare AI
- Cohere
- Coze
- DeepL
- DeepSeek
- Dify
- Doubao
- Fireworks AI
- Galadriel
- Google Gemini
- Generic Provider
- GitHub Copilot
- xAI Grok
- Groq
- Tencent Hunyuan
- LongCat AI
- MiniMax
- Mistral AI
- Moonshot AI
- Ollama
- OpenAI
- OpenRouter
- Alibaba Qwen
- iFLYTEK SparkDesk
- StepFun
- Together AI
- NVIDIA Triton Inference Server
- 01 AI
- Zhipu AI

## Sağlayıcı Oluştur { #create-provider }

1. `Providers` Sayfasına gidin.
2. `Add Provider`'a tıklayın.
3. `Name`, `Type`, `API Key` gibi gerekli seçenekleri doldurun.
4. Bu sağlayıcı için en az bir model yapılandırmak için `Add Model`'e tıklayın.
5. `Save` düğmesine tıklayın.

## Anthropic Uyumlu Uç Nokta Kullanma { #use-an-anthropic-compatible-endpoint }

`Anthropic Claude` sağlayıcısı varsayılan olarak `api.anthropic.com` adresine bağlanır. Aynı API'yi sunan kendi barındırdığınız veya vekil sunucu üzerinden erişilen bir hizmeti kullanmak için `claudeCustomUrl` alanına portu ve varsa yol önekini içeren mutlak bir `http(s)` URL'si girin; örneğin `http://192.168.50.14:8080` veya `https://gateway.example.com/anthropic`.

API'nin tam taban yolunu **değil**, hizmetin adresini ve altında sunulduğu yol önekini belirtin. Anthropic yolları bu değere eklenir; bu nedenle çoğu uyumlu hizmetin belgelerindeki taban URL'si bu eki içerse bile `claudeCustomUrl` `/v1` ile bitmemelidir. `http://192.168.50.14:8080/v1` ayarı GPUStack'in `/v1/v1/models` ve `/v1/v1/messages` yollarını aramasına neden olur. Genellikle çıkarımın erişebildiği bir uç noktada `Test Model` başarısız olur. Bu, `/v1` içeren tam API tabanı olan `openaiCustomUrl` ayarının tersidir. URL'de kimlik bilgisi, sorgu veya parça kabul edilmez; API anahtarı sağlayıcının ayrı anahtar alanına yazılmalıdır.

Böyle bir sağlayıcı varsayılan olarak yine OpenAI API üzerinden sunulur: uç noktaya giderken istekler Anthropic protokolüne, dönen yanıtlar da ters yönde dönüştürülür. İstekleri değiştirmeden iletmek için sağlayıcı yapılandırmasına `protocol: original` ekleyin. Doğrudan iletim, OpenAI şemasında karşılığı olmayan istem önbellekleme, düşünme blokları ve araç kullanım bloklarını korur. Ancak bu durumda sağlayıcı **yalnızca Anthropic protokolünü** sunar: `/v1/chat/completions` yoluna gelen OpenAI biçimli istekler artık dönüştürülmez. Bunun yerine Anthropic yollarını (`/v1/messages`) kullanın.

## Sağlayıcı için Rota Ekle { #add-route-for-provider }

1. `Providers` Sayfasına gidin.
2. Rota oluşturmak istediğiniz Sağlayıcıyı bulun.
3. Bu sağlayıcının `Operations`'inde `Add Route`'a tıklayın.
4. Sağlayıcı için `Name` ve `Route Targets` rotasını gerektiği gibi değiştirin.
5. `Save` düğmesine tıklayın.

## Sağlayıcıyı Düzenle { #edit-provider }

1. `Providers` Sayfasına gidin.
2. Düzenlemek istediğiniz Sağlayıcıyı bulun.
3. Gerektiğinde adı, türü ve API anahtarı özelliklerini değiştirin. Gerektiğinde modelleri model listesine ekleyin/kaldırın.
4. `Save` düğmesine tıklayın.

## Sağlayıcıyı Sil { #delete-provider }

1. `Providers` Sayfasına gidin.
2. Silmek istediğiniz Sağlayıcıyı bulun.
3. `Operations` sütunundaki `Delete` düğmesine tıklayın.
4. Silme işlemini onaylayın.
