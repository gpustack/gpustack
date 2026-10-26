---
translation_status: translated
---

# Çıkarım Arka Uç Yönetimi { #inference-backend-management }

GPUStack, yöneticilerin çıkarım arka uçlarını ve arka uç sürümlerini yapılandırmasına olanak tanır.

Bu makale, Çıkarım Arka Uç sayfası için operasyonel bir kılavuz görevi görmektedir. Desteklenen yerleşik arka uçlar ve bunların yetenekleri için bkz. [Yerleşik Çıkarım Arka Uçları](built-in-inference-backends.md).

Özel arka uçların yapılandırılmasına ilişkin yönergeler ve çalıştığı doğrulanan özel arka uç örnekleri için bkz. [Özel Çıkarım Arka Uçları](../tutorials/using-custom-backends.md).

## Arka Uç Kaynakları { #backend-sources }

GPUStack üç tür çıkarım arka ucunu destekler:

- **Yerleşik**: GPUStack tarafından sağlanan önceden yapılandırılmış arka uçlar (ör. vLLM, MindIE, VoxBox). Bunlar silinemez.
- **Topluluk**: Topluluk Arka Uç Pazarı tarafından paylaşılan arka uçlar. Gerektiğinde bunları etkinleştirebilirsiniz.
- **Custom**: Kendi konfigürasyonlarınızla oluşturduğunuz arka uçlar. Bunlar serbestçe eklenebilir, düzenlenebilir ve silinebilir.

Yerleşik ve Topluluk listeleri GPUStack'i yükseltmeden kablosuz olarak güncellenir. Ayrıca her iki listeyi de kendi belgenize yönlendirebilirsiniz. Bkz. [OTA İçerik Güncellemeleri](ota-content-updates.md).

## Parametre Açıklaması { #parameter-description }

| Parametre Adı | Açıklama | Gerekli |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Ad | Çıkarım arka uç adı | Evet |
| Durum Denetimi Yolu | Arka ucun çalışır durumda olduğunu ve yanıt verdiğini doğrulamak için kullanılan durum denetimi yolu. Varsayılan: /v1/models (OpenAI uyumlu).                                                                                                                                                                                                                                                                                                                                                                                      | Hayır |
| Varsayılan Yürütme Komutu | Konteyner başlatma komutu/args. Örneğin (vLLM): `vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}} --host {{worker_ip}}`. `{{model_path}}`, `{{model_name}}`, `{{port}}`, `{{worker_ip}}`, `{{gpu_count}}`, `{{gpu_ids}}` ve `{{VAR_NAME}}` (ortam değişkenleri için) yer tutucuları, dağıtım bir çalışan için planlandığında otomatik olarak değiştirilir; yer tutucu değişikliğinden sonra argümanlar POSIX stili kullanılarak bölünür. `{{gpu_count}}` ve `{{gpu_ids}}`, zamanlanmış tek çalışana atanan GPU'ları yansıtır (özel arka uçlar, çoklu çalışan çıkarımını desteklemez). Değerleri boşluklarla alıntılayın ve kabuk operatörlerinden kaçının. | Hayır |
| Varsayılan Giriş Noktası | Konteyner giriş noktası geçersiz kılma. Ayarlanırsa bu arka uç için görüntü giriş noktasının yerini alır. Argümanlar POSIX stili kullanılarak bölünür.                                                                                                                                                                                                                                                                                                                                                                 | Hayır |
| Varsayılan Ortam Değişkenleri | Bu arka ucun tüm sürümleri için ayarlanacak ortam değişkenleri. `{{VAR_NAME}}` söz dizimi kullanılarak komutlarda başvurulabilir. Sürüme özgü ortam değişkenleri önceliklidir.                                                                                                                                                                                                                                                                                                                | Hayır |
| Varsayılan Arka Uç Parametreleri | Dağıtım sırasında Gelişmiş Arka Uç Parametreleri bölümünü önceden doldurun; başlatmadan önce bunları ayarlayabilirsiniz | Hayır |
| Bayrak Formatı | Arka ucun başlatma komutunu birleştirirken her `--key value` çiftinin nasıl oluşturulacağını kontrol eder: `Space Separated (--key value)` veya `Equal Sign (--key=value)`. Normalleştirmeyi devre dışı bırakmak için boş bırakın (parametreler tam olarak girildiği gibi tutulur). Çoklu değer seçenekleri her zaman boşlukla ayrılmış halde kalır. Bkz. [Bayrak Formatı](#flag-format).                                                                                                                                                                       | Hayır |
| Ortak Arka Uç Parametreleri | Bu arka uç için yaygın olarak kullanılan parametrelerin listesi, hızlı giriş için dağıtım sırasında `Backend Parameters` girişinde öneri olarak gösterilir. `Default Backend Parameters`'den farklı olarak bunlar yalnızca hızlı seçim için sunulur ve dağıtım formu açıldığında önceden doldurulmaz. Çalışma zamanı davranışını etkilemez.                                                                                                                                                                                                                                                     | Hayır |
| Açıklama | Açıklama | Hayır |
| Sürüm Yapılandırmaları | Bu arka ucun mevcut sürümlerini yapılandırın | Evet |
| Varsayılan Sürüm | Dağıtım sırasında önceden seçilir. Sürüm seçmezseniz görseli kullanılır | Hayır |

## Ortam Değişkenleri { #environment-variables }

Ortam değişkenlerini iki düzeyde tanımlayabilirsiniz:

- **Arka uç seviyesi** (`Default Environment Variables`): Arka ucun tüm sürümlerine uygulanır
- **Sürüm düzeyi** (`Environment Variables`): Bir sürüme özeldir, arka uç düzeyi değişkenlerini geçersiz kılar

Ortam değişkenlerine, `{{VAR_NAME}}` sözdizimi kullanılarak komutlarda başvurulabilir.

**Örnek:**

```yaml
backend_name: my-backend-custom
default_env:
  MODEL_CACHE_DIR: /cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-image:v1
    custom_framework: cuda
    run_command: "serve {{model_path}} --cache {{MODEL_CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}"
    env:
      MODEL_CACHE_DIR: /custom-cache # Overrides backend-level value
```

Bu örnekte:

- `MODEL_CACHE_DIR` arka uç seviyesinde `/cache` olarak ayarlandı
- Sürüm `v1`, onu `/custom-cache` olarak geçersiz kılar
- `LOG_LEVEL` tüm sürümler için `info` olarak kalır
- Her iki değişkene de komutta `{{VAR_NAME}}` sözdizimi kullanılarak başvurulur

Sürüm Yapılandırmaları parametre açıklaması:

| Parametre Adı | Açıklama | Gerekli |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Sürüm | Dağıtım sırasında Arka Uç Sürüm açılır menüsünde gösterilen sürüm adı | Evet |
| Resim Adı | Arka uç için kapsayıcı resim adı (ör. `ghcr.io/org/image:tag`) | Evet |
| Framework<br/>(custom_framework) | Arka uç çerçevesi (dahili tanımlayıcı: `custom_framework`). Dağıtım ve planlama, desteklenen çerçevelere göre filtrelenir | Evet |
| Ortam Değişkenleri | Bu sürüme özel ortam değişkenleri. Arka uç düzeyini `Default Environment Variables` geçersiz kılar. `{{VAR_NAME}}` söz dizimi kullanılarak `Execution Command`'de başvurulabilir. | Hayır |
| Giriş noktası | Sürüme özel konteyner giriş noktası geçersiz kılma. Atlanırsa `Default Entrypoint` kullanılır. Argümanlar POSIX stili kullanılarak bölünür.                                               | Hayır |
| Yürütme Komutu | Sürüme özel başlatma komutu. Atlanırsa Varsayılan Yürütme Komutu kullanılır. Ayrıştırma ve bölme kuralları `Default Execution Command` ile aynıdır.                 | Hayır |

## Bayrak Formatı { #flag-format }

`Flag Format`, arka ucun başlatma komutunu birleştirirken GPUStack'in her `--key value` çiftini nasıl oluşturacağını kontrol eder. Arka uç seviyesinde yapılandırılır, bu arka ucu kullanan tüm model dağıtımları için geçerlidir ve çalışan konteyner başlatma komutunu birleştirdiğinde etkili olur.

Mevcut değerler:

- `Space Separated (--key value)`: boşlukla ayrılır, `--key value` olarak oluşturulur.
- `Equal Sign (--key=value)`: `=` ile birleştirilir, `--key=value` olarak oluşturulur.
- Boş (varsayılan): normalleştirme yok; parametreler tam olarak `Backend Parameters`'da girildiği gibi tutulur.

Bir çıkarım motorunun komut satırı ayrıştırıcısı, her parametreyi manuel olarak yeniden yazmaya gerek kalmadan iki formdan yalnızca birini kabul ettiğinde tutarlı bir parametre stili uygulamak için bu alanı kullanın.

**Örnek:**

Arka ucun `Flag Format`'si `Equal Sign (--key=value)` olarak ayarlandığında, bu arka ucu aşağıdaki `Backend Parameters` ile kullanan bir dağıtım:

```bash
--tensor-parallel-size 2 --max-model-len 8192 --lora-modules a=/path1 b=/path2
```

başlatma komutunda şu şekilde normalleştirilir:

```bash
vllm serve Qwen/Qwen3.5-0.8B --tensor-parallel-size=2 --max-model-len=8192 --lora-modules a=/path1 b=/path2
```

`--tensor-parallel-size` ve `--max-model-len` eşittir işareti formuna dönüştürülürken, `--lora-modules` birden fazla değer taşıdığı için boşlukla ayrılmış formu korur.

!!! note

    - **Çok değerli seçenekler** (ör. `--lora-modules`) seçilen formattan bağımsız olarak her zaman boşlukla ayrılmış halde kalır. Bunları `--key=value` biçimine dönüştürmek argparse anlambilimini değiştirir, çünkü sonraki değerler genellikle öncekilerin üzerine yazılır.
    - **Tek çizgi kısa seçenekleri** (örneğin, llama.cpp'nin `-n`, `-ngl` ve `-m`) kelimesi kelimesine korunur, `--n` gibi çift çizgi formlarına zorlanmaz.

## Özel Çıkarım Arka Ucu Ekle { #add-custom-inference-backend }

1. Sağ üst köşedeki "Arka Uç Ekle" butonuna tıklayın.
2. Formu doldurarak veya YAML tanımını yapıştırarak özel bir çıkarım arka ucu ekleyebilirsiniz. Alan anlamları için yukarıdaki parametre açıklamalarına bakın.
3. Arka uç adı oluşturulduktan sonra değiştirilemez. Özel arka uç adları "-custom" (formda önceden doldurulmuş) ile bitmelidir.
4. Göndermek için "Kaydet"e tıklayın.

Özel çıkarım arka ucu eklemenin iki yolu vardır:

- Kullanıcı arayüzü formu aracılığıyla: **Model Hizmeti > Çıkarım Arka Uçları** sayfasına gidin ve **Özel Çıkarım Arka Uç Ekle** düğmesine tıklayın.
- YAML yapılandırması aracılığıyla: Arka uç yapılandırmasını içeren bir YAML dosyasını içe aktarın.

### Topluluk Çıkarımı Arka Uçunu Etkinleştir { #enable-community-inference-backend }

Bunlar aslında "topluluk" kaynak etiketine sahip özel arka uçlardır ve manuel yapılandırma olmadan hızlı bir şekilde özel arka uçlar oluşturmanıza olanak tanır.

1. Çıkarım Arka Uç sayfasında sağ üst köşedeki "Arka Uç Ekle" düğmesine tıklayın.
2. Piyasadaki mevcut topluluk arka uçlarına göz atmak için "Topluluk" seçeneğini seçin.
3. Kullanmak istediğiniz arka ucu bulun ve kartın işlem menüsünden "Etkinleştir"e tıklayın.
4. Etkinleştirildiğinde arka uç, model dağıtımları için kullanılabilir hale gelir. Devre dışı bırakmak için Çıkarım Arka Uç sayfasından arka ucu silin.

## Çıkarım Arka Ucunu Düzenle veya Özel Sürüm Ekle { #edit-inference-backend-or-add-custom-version }

1. Çıkarım Arka Uç sayfasında hedef arka ucu bulun. Kartın sağ üstteki açılır menüsünden "Düzenle"yi seçin.
2. Arka uç özelliklerini değiştirin (ad değiştirilemez) veya yeni bir sürüm ekleyin.
3. Yerleşik arka uçlar için özel sürümlerin "-custom" (formda önceden doldurulmuş) ile bitmesi gerekir.
4. Göndermek için "Kaydet"e tıklayın.

### Örnek: Yerleşik vLLM Çıkarımı Arka Ucuna Özel Bir Sürüm Ekleme { #example-add-a-custom-version-to-the-built-in-vllm-inference-backend }

1. Çıkarım Arka Uç sayfasında vLLM çıkarım arka ucunu bulun. Kartın sağ üstteki açılır menüsünden "Düzenle"yi seçin.
2. Sürüm Yapılandırmaları bölümünde "Sürüm Ekle"ye tıklayın.
3. Alanları aşağıdaki gibi doldurun:

- Versiyon: `0.19.0`
- Resim Adı: `vllm/vllm-openai:v0.19.0`
- Çerçeve: `cuda`
- Görüntü Giriş Noktasını Geçersiz Kıl: `vllm serve`
- Yürütme Komutu: `{{model_path}} --host {{worker_ip}} --port {{port}} --served-model-name {{model_name}}`

4. Göndermek için "Kaydet"e tıklayın.

!!! note

    vLLM, v0.11.1'den bu yana Docker görüntüsünün giriş noktasını değiştirdi. Bu nedenle vLLM v0.11.1 veya üzeri için özel bir sürüm eklerken `Override Image Entrypoint` ve `Execution Command` alanını belirtmeniz gerekir; aksi takdirde model başlatılamaz. `gpustack/runner` görsellerinin daha yeni sürümlerini kullanıyorsanız `Execution Command` alanını ayarlamanıza gerek yoktur.

    Yürütülebilir dosyayı `Override Image Entrypoint` aracılığıyla sağlayan arka uçlar için (vLLM ve SGLang gibi), `Execution Command` yalnızca argümanlar içermelidir. Yürütülebilir dosyayı (örneğin `vllm serve` veya `python -m vllm ...`) `Execution Command`'e koymayın, aksi takdirde giriş noktasından sonra eklenir ve `vllm serve python -m vllm ...` gibi kopyalanmış bir önek üretir.

### Örnek: Yerleşik SGLang Çıkarımı Arka Ucuna Özel Bir Sürüm Ekleme { #example-add-a-custom-version-to-the-built-in-sglang-inference-backend }

1. Çıkarım Arka Uç sayfasında SGLang çıkarım arka ucunu bulun. Kartın sağ üstteki açılır menüsünden "Düzenle"yi seçin.
2. Sürüm Yapılandırmaları bölümünde "Sürüm Ekle"ye tıklayın.
3. Alanları aşağıdaki gibi doldurun:

- Versiyon: `0.5.10`
- Resim Adı: `lmsysorg/sglang:v0.5.10`
- Çerçeve: `cuda`
- Görüntü Giriş Noktasını Geçersiz Kıl: `sglang serve`
- Yürütme Komutu: `--model-path {{model_path}} --host {{worker_ip}} --port {{port}}`

4. Göndermek için "Kaydet"e tıklayın.

## Özel Çıkarım Arka Ucunu Sil { #delete-custom-inference-backend }

1. Çıkarım Arka Uç sayfasında hedef arka ucu bulun ve kartın sağ üst açılır menüsünden "Sil"i seçin.
2. Yerleşik arka uçlar silinemez.
3. Onay iletişim kutusunda "Sil"e tıklayın.

## Çıkarım Arka Uç Sürümlerini Listeleme { #list-versions-of-inference-backend }

Çıkarım Arka Uç sayfasında, tüm yerleşik ve özel olarak eklenen sürümlere göz atabileceğiniz bir modal açmak için arka uç kartında herhangi bir yere tıklayın (eylem düğmeleri hariç).

## Esnek Test Dağıtımı { #flexible-testing-deployment }

Arka uç tanımını düzenlemeden görüntüyü ve başlatma komutunu hızlı bir şekilde doğrulamak veya ayarlamak için bu modu kullanın.

1. Dağıtımlar sayfasına gidin, "Modeli Dağıt" düğmesini tıklayın ve herhangi bir model kaynağını seçin.
2. Temel sekmesinde, "Arka uç" açılır menüsünü açın ve "Yerleşik" bölümü altında "Özel"i seçin.
3. İki alan görünür: `image_name` ve `run_command`. Bunlar yalnızca bu dağıtım için arka uç yapılandırmasını geçersiz kılar.
4. Kalan gerekli ayarları inceleyin ve dağıtımı gönderin.

## Özel Çıkarım Arka Uçlarının Sınırlamaları { #limitations-of-custom-inference-backends }

Özel çıkarım arka uçları, birden fazla çalışan arasında dağıtılmış çıkarımı desteklemez.
