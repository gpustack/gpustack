---
translation_status: translated
---

# Model Kataloğu { #model-catalog }

Model Kataloğu, GPUStack ayarlı modellerin bir indeksidir.

## Modellere Göz Atın { #browse-models }

`Catalog` sayfasına giderek Model Kataloğuna göz atabilirsiniz. Modelleri ada ve kategoriye göre filtreleyebilirsiniz. Aşağıdaki ekran görüntüsü Model Kataloğu sayfasını göstermektedir:

![Model Kataloğu](../assets/model-catalog.png)

## Katalogdan Bir Model Dağıtın { #deploy-a-model-from-the-catalog }

Model kartına tıklayarak Model Kataloğundan bir modeli dağıtabilirsiniz. Bir model dağıtım yapılandırma sayfası görünecektir. Dağıtım yapılandırmasını inceleyip özelleştirebilir ve modeli dağıtmak için `Save` düğmesine tıklayabilirsiniz.

## Model Kataloğunu Özelleştir { #customize-model-catalog }

`--model-catalog-file` işaretini kullanarak GPUStack sunucu yapılandırması aracılığıyla bir YAML dosyası sağlayarak Model Kataloğunu özelleştirebilirsiniz. Yerel bir dosya yolunu veya bir URL'yi kabul eder. Şema için yerleşik model kataloğu dosyasına [buradan](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml) başvurabilirsiniz.

Sunucuyu yeniden başlatmadan kataloğu özelleştirmek için bunun yerine bir katalog kaynağı yapılandırın; belgeniz paketlenmiş kataloğun yerini alır ve çekmece, başlamanız için size o kataloğu verir. Bkz. [OTA İçerik Güncellemeleri](ota-content-updates.md).

Aşağıda özel bir model kataloğu YAML dosyası örneği verilmiştir:

```yaml
draft_models:
- name: Qwen3-8B-EAGLE3
  algorithm: eagle3
  source: huggingface
  huggingface_repo_id: Tengyunw/qwen3_8b_eagle3
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Model Kataloğunun Hava Boşluklu Ortamlarda Kullanımı { #using-model-catalog-in-air-gapped-environments }

Yerleşik model kataloğu, modelleri Hugging Face veya ModelScope'tan alır. GPUStack'i internet erişimi olmayan, hava boşluklu bir ortamda kullanıyorsanız, model kataloğunu yerel yol model kaynağı kullanacak şekilde özelleştirebilirsiniz. İşte bir örnek:

```yaml
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: local_path
      local_path: /path/to/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: local_path
      # assuming you have /path/to/DeepSeek-R1-0528 directory
      local_path: /path/to/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Model Katalog Şeması { #model-catalog-schema }

Model Kataloğu YAML dosyası iki ana bölüm içerir: `draft_models` ve `model_sets`.

- `draft_models`: Spekülatif kod çözmeye yönelik taslak modellerin listesi.
- `model_sets`: Test edilen ve optimize edilen model setlerinin listesi.

Her taslak modelde aşağıdaki alanlar bulunur:

| Alan | Tür | Açıklama |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | Taslak modelin adı.                                                                    |
| algorithm           | string        | Modelin spekülatif kod çözme algoritması. Şu anda yalnızca Eagle3 desteklenmektedir.                                                    |
| source              | string        | Modelin kaynağı (ör. huggingface, model_scope).                                               |
| huggingface_repo_id | string        | Modelin Hugging Face veri deposu kimliği (kaynak huggingface ise).                            |
| model_scope_model_id  | string        | Modelin ModelScope veri havuzu kimliği (kaynak model_scope ise).                              |

Her model seti aşağıdaki alanlara sahiptir:

| Alan | Tür | Açıklama |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | Modelin adı.                                                                      |
| description         | string        | Modelin kısa açıklaması.                                                           |
| home                | string        | Modelin ana sayfa URL'si.                                                              |
| icon                | string        | Modelin simge URL'si.                                                                  |
| categories          | list of str   | Modelin ait olduğu kategorilerin listesi.                                                  |
| capabilities        | list of str   | Modelin yeteneklerinin listesi.                                                       |
| size                | int           | Modelin milyarlarca parametre cinsinden boyutu.                                                                 |
| licenses            | list of str   | Modelin lisanslarının listesi.                                                              |
| release_date        | string        | Modelin YYYY-AA-GG formatında çıkış tarihi.                                         |
| specs               | list of spec  | Model için dağıtım spesifikasyonlarının listesi. |


Her dağıtım özelliği aşağıdaki alanlara sahiptir:

| Alan | Tür | Açıklama |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| mode                | string        | GPUStack, üretim, gecikme ve standart senaryolar dahil olmak üzere farklı kullanım durumları için hem geleneksel hem de optimize edilmiş modlar sağlar. Kullanıcılar ayrıca gerektiği gibi özel modları da tanımlayabilir. |
| quantization        | string        | Niceleme türü (örn. FP16, FP8, INT8).                                               |
| gpu_filters         | dict          | Uyumlu GPU'ları belirtmek için GPU filtreleri.                                                  |

Dağıtım spesifikasyonundaki diğer alanlar, model API alanlarına benzer. Daha fazla ayrıntı için [API Referansı](../api-reference.md) belgelerine bakın.
