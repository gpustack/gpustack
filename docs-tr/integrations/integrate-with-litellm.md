---
translation_status: translated
---

# LiteLLM ile entegrasyon { #integrate-with-litellm }

LiteLLM, yerel olarak dağıtılan LLM'leri, gömme modellerini, yeniden sıralamayı, Konuşmadan Metne ve Metinden Konuşmaya yeteneklerini kurumsal çalışanlar için birleşik bir OpenAI uyumlu/Anthropic API ağ geçidinde toplamak için GPUStack ile entegre olabilir.

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

## GPUStack'i LiteLLM'ye entegre etme { #integrating-gpustack-into-litellm }

1. LiteLLM yönetim kullanıcı arayüzünü açın

```
http://your-litellm-url:4000/ui
```

![litellm-kontrol paneli](../assets/integrations/litellm-1.png)

2. modeller+uç noktalar menüsünü aç -> model ekle

![litellm-add-model1](../assets/integrations/litellm-2.png)
![litellm-add-model2](../assets/integrations/litellm-3.png)

3. litellm'de test ->oyun alanı

![litellm-sohbet1](../assets/integrations/litellm-4.png)
