---
translation_status: translated
---

# Özel Çıkarım Arka Uçlarını Kullanma { #using-custom-inference-backends }

Bu kılavuz, doğrulanmış topluluk yapılandırmalarını kullanma ve kendi yapılandırmanızı sıfırdan oluşturma dahil olmak üzere GPUStack'e özel çıkarım arka uçlarının nasıl ekleneceğini açıklamaktadır.

Parametre açıklamaları için [Kullanıcı Kılavuzu](../user-guide/inference-backend-management.md)'e bakın.

## Arka Uç Türleri { #backend-types }

GPUStack üç tür çıkarım arka ucunu destekler:

- **Yerleşik**: GPUStack tarafından bakımı yapılan, farklı donanımlar için otomatik olarak optimize edilen önceden yapılandırılmış arka uçlar (vLLM, MindIE, VoxBox, SGLang...).
- **Topluluk**: Önceden doğrulanmış özel arka uç yapılandırmaları. Bunlar aslında manuel kurulumu basitleştirmek için "topluluk" etiketli CustomBackend'lerdir.
- **Custom**: Özel Docker imajları ve komutlarıyla kendinizi yapılandırdığınız arka uçlar.

## Topluluk Arka Uçlarını Kullanma { #using-community-backends }

Topluluk arka uçları, popüler çıkarım motorlarını eklemenin en hızlı yolunu sağlar.

**Adımlar:**

1. Çıkarım Arka Uç sayfasına gidin → "Arka Uç Ekle"yi tıklayın
2. "Topluluk" seçeneğini seçin
3. "Topluluk Arka Uç Pazarı"na göz atın ve ihtiyacınız olan arka uçları etkinleştirin

## Özel Arka Uçlar Oluşturma { #creating-custom-backends }

### Temel Adımlar { #core-steps }
1. Gerekli çıkarım arka ucu için Docker imajını hazırlayın
2. Başlatma komutunu belirlemek için imajnün ENTRYPOINT veya CMD'sini anlayın
3. Çıkarım Arka Uç sayfasına yapılandırma ekleme
4. Modelleri dağıtın ve yeni eklenen arka ucu seçin

### Örnek: TensorRT-LLM { #example-tensorrt-llm }
Aşağıda, çıkarım arka ucunun nasıl ekleneceği ve kullanılacağını göstermek için örnek olarak TensorRT-LLM kullanılmaktadır.
> Bu örnekler performans açısından optimize edilmiş yapılandırmalar değil, işlevsel gösterimlerdir. Daha iyi performans için her bir arka ucun ayarlamaya ilişkin resmi belgelerine bakın.

1. TensorRT-LLM belgelerinde bağlantısı verilen [sürüm sayfasında](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release) gerekli imajyü bulun.
2. TensorRT-LLM imajları çıkarım hizmetini `trtllm-serve` kullanarak başlatmalıdır; aksi takdirde etkileşimli bir kabuk oturumu başlatırlar. `run_command`, dağıtım bir çalışan için planlandığında otomatik olarak gerçek değerlerle değiştirilen `{{model_path}}` ve `{{port}}` (ve isteğe bağlı olarak `{{model_name}}`, `{{worker_ip}}`, `{{gpu_count}}`, `{{gpu_ids}}`) gibi yer tutucuları destekler. `{{gpu_count}}` ve `{{gpu_ids}}`, zamanlanan çalışana atanan GPU'ları yansıtır (özel arka uçlar tek bir çalışan üzerinde çalışır).
3. Çıkarım Arka Uç sayfasına yapılandırma ekleyin; YAML içe aktarımı desteklenmektedir. Örnek:
```yaml
backend_name: TensorRT-LLM-custom
default_version: 1.2.0rc0
version_configs:
  1.2.0rc0:
    image_name: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0
    run_command: 'trtllm-serve {{model_path}} --host 0.0.0.0 --port {{port}}'
    custom_framework: cuda
```

4. Dağıtımlar sayfasında yeni eklenen arka ucu seçin ve modeli dağıtın.
![image.png](../assets/tutorials/using-custom-backend/deploy-by-custom-backend.png)

**Sonuç**

Çıkarım arka ucu hizmeti başladıktan sonra model_instance durumu RUNNING olur.
![image.png](../assets/tutorials/using-custom-backend/custom-backend-running.png)
Oyun Alanında sohbetlere katılabilirsiniz.
![image.png](../assets/tutorials/using-custom-backend/use-custom-backend-in-playground.png)

## Gelişmiş Yapılandırma { #advanced-configuration }

### Ortam Değişkenlerini Kullanma { #using-environment-variables }

Ortam değişkenleri, komutlarda değerleri sabit kodlamadan esnek yapılandırma sağlar:

```yaml
backend_name: advanced-backend-custom
default_env:
  CACHE_DIR: /models/cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --cache {{CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}'
    env:
      LOG_LEVEL: debug  # Override for this version
```

Bu örnekte:
- `CACHE_DIR` ve `LOG_LEVEL` arka uç düzeyinde tanımlanır
- `v1` sürümü, `LOG_LEVEL` değerini `debug` olarak geçersiz kılar
- Her iki değişkene de komutta `{{VAR_NAME}}` sözdizimi kullanılarak başvurulur

### Özel Giriş Noktası { #custom-entrypoint }

Görüntü özel başlatma gerektirdiğinde kabın varsayılan giriş noktasını geçersiz kılın. Giriş noktalarını hem arka uç hem de sürüm düzeyinde ayarlayabilirsiniz:

```yaml
backend_name: custom-entry-backend-custom
default_entrypoint: /usr/local/bin/default-init
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --port {{port}}'
  v2:
    image_name: my-backend:v2
    custom_framework: cuda
    entrypoint: /usr/local/bin/v2-init  # Version-specific entrypoint overrides default
    run_command: 'serve {{model_path}} --port {{port}}'
```
