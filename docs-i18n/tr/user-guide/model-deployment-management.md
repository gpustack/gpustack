---
translation_status: translated
---

# Model Dağıtım Yönetimi { #model-deployment-management }

`Models - Deployments` sayfasına giderek GPUStack'teki model dağıtımlarını yönetebilirsiniz. GPUStack'teki bir model dağıtımı, model örneklerinin bir veya daha fazla kopyasını içerir. Dağıtım sırasında GPUStack, model meta verilerinden model örnekleri için kaynak gereksinimlerini otomatik olarak hesaplar ve bunları uygun şekilde mevcut çalışanlara planlar.

## Modeli Dağıt { #deploy-model }

Şu anda [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn) modelleri ve yerel yollar desteklenmektedir.

### Hugging Face Modelini Uygulama { #deploying-a-hugging-face-model }

1. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Hugging Face`'i seçin.

2. Sol üstteki arama çubuğunu kullanarak modeli `Hugging Face`'den isme göre arayın. Örneğin, `Qwen/Qwen3-0.6B`.

3. `Name`, `Cluster`, `Backend`, `Backend Version` ve `Replicas`'yi gerektiği gibi ayarlayın.

4. Gerekirse performans yapılandırmaları için `Performance` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Performansla İlgili Yapılandırma](#performance-related-configuration) bölümüne bakın.

5. Gerekirse yapılandırmaları planlamak için `Scheduling` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Zamanlama Yapılandırması](#scheduling-configuration) bölümüne bakın.

6. Gerekirse gelişmiş yapılandırmalar için `Advanced` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Gelişmiş Yapılandırma](#advanced-configuration) bölümüne bakın.

7. `Save` düğmesine tıklayın.

### ModelScope Modelini Dağıtma { #deploying-a-modelscope-model }

1. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `ModelScope`'i seçin.

2. Sol üstteki arama çubuğunu kullanarak modeli `ModelScope`'den isme göre arayın. Örneğin, `Qwen/Qwen3-0.6B`.

3. `Name`, `Cluster`, `Backend`, `Backend Version` ve `Replicas`'yi gerektiği gibi ayarlayın.

4. Gerekirse performans yapılandırmaları için `Performance` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Performansla İlgili Yapılandırma](#performance-related-configuration) bölümüne bakın.

5. Gerekirse yapılandırmaları planlamak için `Scheduling` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Zamanlama Yapılandırması](#scheduling-configuration) bölümüne bakın.

6. Gerekirse gelişmiş yapılandırmalar için `Advanced` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Gelişmiş Yapılandırma](#advanced-configuration) bölümüne bakın.

7. `Save` düğmesine tıklayın.

### Yerel Yol Modelini Dağıtma { #deploying-a-local-path-model }

Bir modeli yerel yoldan dağıtabilirsiniz. Model yolu, çalışanlarda bulunan bir dizin (örneğin indirilmiş bir Hugging Face model dizini) veya bir dosya (örneğin bir GGUF model dosyası) olabilir. Bu, hava boşluğu olan bir ortamda çalışırken kullanışlıdır.

!!! note

    1. GPUStack, kaynak gereksinimlerini tahmin etmek için model dosyalarını kullanır. Model yoluna sunucuda erişilemiyorsa GPUStack, ona çalışanlardan erişmeye çalışacaktır.
    2. GPUStack model dosyalarını otomatik olarak senkronize etmez. Model yolunun hedef çalışanlar üzerinde erişilebilir olduğundan emin olmalısınız (ör. NFS, rsync vb. kullanarak). Modeli belirli çalışanlara dağıtmak için çalışan seçici yapılandırmasını da kullanabilirsiniz.

Yerel yol modelini dağıtmak için:

1. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Local Path`'i seçin.

2. Dağıtımın `Name` alanını doldurun.

3. `Model Path`'ı doldurun.

4. `Cluster`, `Backend`, `Backend Version` ve `Replicas`'yi gerektiği gibi ayarlayın.

5. Gerekirse performans yapılandırmaları için `Performance` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Performansla İlgili Yapılandırma](#performance-related-configuration) bölümüne bakın.

6. Gerekirse yapılandırmaları planlamak için `Scheduling` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Zamanlama Yapılandırması](#scheduling-configuration) bölümüne bakın.

7. Gerekirse gelişmiş yapılandırmalar için `Advanced` bölümünü genişletin. Daha fazla ayrıntı için lütfen [Gelişmiş Yapılandırma](#advanced-configuration) bölümüne bakın.

8. `Save` düğmesine tıklayın.

### Arka uç { #backend }

Şu anda GPUStack bazı yerleşik arka uçları desteklemektedir: vLLM, SGLang, MindIE ve VoxBox.

Daha fazla ayrıntı için lütfen [Çıkarım Arka Uçları](built-in-inference-backends.md) bölümüne bakın.

### Arka Uç Sürümü { #backend-version }

Bir arka uç sürümü seçin. Sürümün kullanılabilirliği seçilen arka uca bağlıdır. Bu seçenek, uyumluluğu sağlamak veya belirli arka uç sürümlerinde sunulan özelliklerden yararlanmak için kullanışlıdır.

Yerleşik bir vLLM veya SGLang arka ucu için, çalıştırıcı kataloğunda olmayan bir kapsayıcı görüntüsünü kullanmak üzere dağıtım yapılandırmasında `image_name` değerini ayarlayın. `backend_version` ayarlanmadan bırakın: özel bir resim ve arka uç sürümü birlikte seçilemez. Bir vLLM dağıtımı özel bir görüntü kullandığında, dağıtılmış yürütme varsayılan olarak `mp` olur. Görüntü Ray içeriyorsa ve onu kullanmak istiyorsanız, `backend_parameters`'de `--distributed-executor-backend=ray`'i ayarlayın.

Yerleşik bir vLLM veya SGLang arka ucu için, `run_command` yürütülebilir dosya olmadan ilk başlangıç argümanlarını sağlar; GPUStack, hizmet veren ana bilgisayar ve bağlantı noktası da dahil olmak üzere yönetilen bağımsız değişkenlerini eklemeye devam ediyor. Özel bir arka uç için `run_command`, yürütülebilir dosya da dahil olmak üzere tam komutu belirtir. Örnek için [Özel Çıkarım Arka Uçlarını Kullanma](../tutorials/using-custom-backends.md) konusuna bakın.

## Model Dağıtımını Düzenle { #edit-model-deployment }

1. Dağıtım listesi sayfasında düzenlemek istediğiniz model dağıtımını bulun.
2. `Operations` sütunundaki `Edit` düğmesine tıklayın.
3. Nitelikleri gerektiği gibi güncelleyin. Örneğin, ölçeği büyütmek veya küçültmek için `Replicas`'ı değiştirin.
4. `Save` düğmesine tıklayın.

!!! note

    Model dağıtımını düzenledikten sonra yapılandırma mevcut model örneklerine uygulanmayacaktır. Mevcut model örneklerini silmeniz gerekir. GPUStack, güncellenen model yapılandırmasına göre yeni örnekleri yeniden oluşturacaktır.

## Model Dağıtımını Durdurun { #stop-model-deployment }

Bir model dağıtımını durdurmak, kopya hedefini sıfıra ayarlar, örneklerini siler ve kaynakları serbest bırakır. Zamanlanmış ölçeklendirmenin etkin olduğu dağıtımlar için, aynı zamanda zamanlama yürütmesini askıya almak üzere `scaling_schedule.paused` değerini de ayarlar. Etkinleştirilen ayar, temel ve kurallar korunur. Zamanlamayı etkin tutan düzenlemeler ve geri almalar, duraklatma durumunu korur; sıradan dağıtımlar, herhangi bir duraklatma durumu olmadan replika ölçeklendirmeyi kullanır.

1. Dağıtım listesi sayfasında durdurmak istediğiniz model dağıtımını bulun.
2. `Operations` sütunundaki üç nokta düğmesini tıklayın, ardından `Stop`'i seçin.
3. İşlemi onaylayın.

## Model Dağıtımını Başlat { #start-model-deployment }

Zamanlanmış ölçeklendirme etkinleştirildiğinde, başlangıç `scaling_schedule.paused`'i temizler ve GPUStack, kaçırılan pencereleri yeniden oynatmadan geçerli zamandaki programı değerlendirir. Temel sıfır olduğunda çoğaltma hedefi etkin pencerelerin dışında sıfır kalabilir; dağıtım duraklatılmak yerine zamanlamasını bekliyor. Bu dağıtımlar için eylem `Resume Schedule` olarak etiketlenmiştir.

Etkin bir zamanlama olmadan başlatma, çoğaltma hedefini bire ayarlar. Zamanlanmış ölçeklendirmenin devre dışı bırakılması ayrı bir yapılandırma seçeneğidir. Kopya sayısını doğrudan listede düzenlemek, onaylandıktan sonra manuel ölçeklendirmeye geçer.

1. Dağıtım listesi sayfasında başlatmak istediğiniz model dağıtımını bulun.
2. `Operations` sütunundaki üç nokta düğmesini tıklayın, ardından `Start`'i seçin.

## Model Dağıtımını Sil { #delete-model-deployment }

1. Dağıtım listesi sayfasında silmek istediğiniz model dağıtımını bulun.
2. `Operations` sütunundaki üç nokta düğmesini tıklayın, ardından `Delete`'i seçin.
3. Silme işlemini onaylayın.

## Dağıtımları Dışa ve İçe Aktarma { #export-and-import-deployments }

Dağıtımları, tam yapılandırmalarını içeren bir YAML dosyasına aktarın ve bunları yeniden oluşturmak için bu dosyayı içe aktarın. Tipik kullanımlar: Yükseltme veya geçişten önce yedekleme yapın, GPUStack'i yeniden yükledikten sonra geri yükleyin, dağıtım yapılandırmasını sürüm kontrolü altında tutun, bir dağıtım şablonunu ekibinizle paylaşın veya aynı dağıtımları başka bir kümede yeniden oluşturun.

### Dağıtımları Dışa Aktarma { #export-deployments }

1. Dağıtım listesi sayfasında dışa aktarmak istediğiniz model dağıtımını bulun.
2. `Operations` sütunundaki üç nokta düğmesini tıklayın, ardından `Export YAML`'i seçin.
3. Birden fazla dağıtımı aynı anda dışa aktarmak için önce bunları seçin, ardından listenin üzerindeki toplu işlemler açılır menüsünden `Export YAML` seçeneğini seçin.

Tarayıcı, tek bir dağıtım için `<deployment-name>.yaml` veya birkaç dağıtım için `gpustack-deployments-<timestamp>.yaml` adlı bir YAML dosyası indirir.

Dosya, YAML belgesi başına `---` ile ayrılmış bir dağıtım içerir. Her biri, dağıtımı oluşturan istek gövdesi artı ait olduğu kümeden sunucunun oluşturduğu alanlar çıkarılarak elde edilir:

```yaml
# Exported from GPUStack v2.3.0 at 2026-09-07T10:00:00Z
name: qwen3-8b
cluster_name: production
source: huggingface
huggingface_repo_id: Qwen/Qwen3-8B
replicas: 2
placement_strategy: binpack
worker_selector:
  zone: a
gpu_selector:
  gpu_ids:
  - worker-1:cuda:0
  gpus_per_replica: 1
backend: vLLM
backend_version: 0.11.0
backend_parameters:
- --max-model-len=32768
env:
  HF_TOKEN: hf_xxx
enable_model_route: true
---
name: bge-m3
cluster_name: staging
source: huggingface
huggingface_repo_id: BAAI/bge-m3
replicas: 1
```

Her dağıtım sol kenar boşluğunda başladığından, dosyadan bir tane kopyalayabilir veya yeniden girinti yapmadan elle yazabilirsiniz. İçe aktarma aynı zamanda tek bir YAML listesi olarak yazılan bir dosyayı da kabul eder; GPUStack'in daha önceki sürümleri de bu şekilde dışa aktarılır.

#### Saha Referansı { #field-reference }

Dosyadaki her alan, dağıtım formunda ayarladığınız bir şeydir. Aşağıda, dışa aktarılan bir dosyanın bunları yazma sırasına göre her iki yerde de her birine ne denildiği verilmiştir. Bir alanın ne anlama geldiği, sağdaki sütundan bağlantı verilerek bu ayarın nerede olduğu belgelenir.

| Alan | Dağıtım formunda |
| --- | --- |
| `name` | `Name` |
| `cluster_name` | `Cluster`, kümenin kimliği yerine adı olarak yazılır |
| `source` | `Source` |
| `huggingface_repo_id`, `huggingface_filename` | `Repo ID` ve `File Name`, Hugging Face modeli için |
| `model_scope_model_id`, `model_scope_file_path` | `Repo ID` ve `File Name`, ModelScope modeli için |
| `local_path` | `Model Path`, yerel yol modeli için |
| `replicas` | `Replicas` |
| `categories` | [`Model Category`](#model-category) |
| `placement_strategy` | `Scheduling` → [`Auto`](#auto), yerleştirme stratejisi |
| `cpu_offloading` | [`Allow CPU Offloading`](#allow-cpu-offloading) |
| `distributed_inference_across_workers` | [`Allow Distributed Inference Across Workers`](#allow-distributed-inference-across-workers) |
| `worker_selector` | `Scheduling` → [`Manual`](#manual) → `Selector` |
| `gpu_selector` | `Scheduling` → `Specify GPU` |
| `gpu_type_selector` | `Scheduling` → `Specify GPU Type` |
| `backend` | [`Backend`](#backend) |
| `backend_version` | [`Backend Version`](#backend-version) |
| `backend_parameters` | [`Backend Parameters`](#backend-parameters) |
| `image_name` | Yerleşik veya özel bir arka uç için kapsayıcı görüntüsü; yerleşik arka uçlar için `backend_version` ile birbirini dışlayan |
| `run_command` | Yerleşik vLLM veya SGLang için başlangıç argümanları veya özel bir arka uç için tam komut |
| `native_anthropic_api` | `Native Anthropic API` |
| `env` | [`Environment Variables`](#environment-variables) |
| `restart_on_error` | [`Auto-Restart On Error`](#auto-restart-on-error) |
| `extended_kv_cache` | [`Extended KV Cache`](#extended-kv-cache) |
| `speculative_config` | `Performance` → spekülatif kod çözme |
| `scaling_schedule` | `Scheduled Scaling` |
| `generic_proxy` | [`Enable Generic Proxy`](#enable-generic-proxy) |
| `lora_list` | [`LoRA Adapters`](#lora-adapters) |
| `enable_model_route` | `Enable Model Route`, formun yalnızca dağıtım oluştururken sunduğu |

Son alan grubu — `gpu_selector`, `gpu_type_selector`, `extended_kv_cache`, `speculative_config`, `scaling_schedule` ve `lora_list` — tek bir değer yerine bir blok tutar ve bu bloğun şekli burada yazılmaz. Bir dağıtımda ayarı yapılandırın, dışa aktarın ve bloğu dosyadan kopyalayın: bu, hem onu yazıya dökmekten daha hızlıdır hem de bundan emin olmanın tek yoludur.

`description` ve `distributable` dosya içerisinde taşınır ancak form üzerinde herhangi bir kontrolü yoktur.

Dosyanın dışarıda bıraktığı şeyler: Kimlikler, zaman damgaları, hazır kopyalar ve duraklatma durumu gibi çalışma zamanı durumu, zamanlayıcı tarafından türetilen meta veriler, sahibi kuruluş, erişim politikası ve LoRA çalışma zamanı yolları. Zamanlanmış ölçeklendirme etkinleştirildiğinde, dışa aktarılan kopya sayısı geçerli pencere hedefi yerine yapılandırılmış temeldir. Sunucu, içe aktarma sırasında çalışma zamanı durumunu yeniden oluşturur; Etkinleştirilmiş duraklatılmış zamanlamanın mevcut bir dağıtımın üzerine yazılması, duraklatma durumunu korur.

Birkaç kümeden dağıtımları aynı anda seçmek iyidir: her giriş kendi `cluster_name`'ini kaydeder, böylece bir dosya hepsini açıklar ve bütün olarak geri aktarılır. Küme, kimlik yerine ada göre yazılır, böylece bir dosya, yeniden yüklemede olduğu gibi, kümelerini farklı şekilde numaralandıran bir GPUStack'e doğru bir şekilde geri yüklenir.

Bir dosya, ne kadar çok kümeye sahip olursa olsun, bir kuruluşu kapsar. Dağıtım adları, bir kuruluş genelinde değil, bir kuruluş içinde benzersizdir, bu nedenle, yalnızca `All`'i görüntüleyen bir platform yöneticisinin yapabileceği, kuruluşları kapsayan bir seçim reddedilir; her kuruluşu kendi başına ihraç edin.

Bir dağıtımın çalıştırıldığı yer hiçbir zaman dışa aktarılmaz. Otomatik olarak planlanan bir dağıtımın dosyasında hiçbir `gpu_selector` yoktur, bu nedenle onu içe aktarmak, hedef kümede mevcut olan her şeye göre yeniden planlar. Kendi seçtiğiniz GPU'lar bir planlama sonucundan ziyade amacınızdır, dolayısıyla yazıldığı gibi dışa aktarılırlar; aşağıdaki farklı bir kümeye içe aktarma notuna bakın.

!!! warning

    Ortam değişkenleri tam olarak girdiğiniz şekilde dışa aktarılır ve `HF_TOKEN` gibi kimlik bilgileri içerebilir. Dışa aktarılan bir dosyayı halka açık bir depoya teslim etmeyin; Paylaşmadan önce hassas değerleri kaldırın veya değiştirin.

### Dağıtımları İçe Aktarma { #import-deployments }

1. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `YAML File`'i seçin.
2. Bir `.yaml` dosyası seçin. Dosya başlangıç noktasıdır; önce bir kümeden hiçbir şey okunmaz ve dosyanın adlandırdığı dağıtımlar, farkın salt okunur tarafını dolduracak şekilde aranır. İkinci bir dosya seçildiğinde ilk dosyanın, düzenlemelerin ve tümünün yerine geçilir.
3. Dağıtımların nereye gideceğine karar verin. `Follow the file`, çekmece böyle başlar, kümedeki her girişi kendi `cluster_name` adlarıyla yerleştirir; bu, tek bir içe aktarma işleminde birkaç kümeyi kapsayan bir yedeği geri yükleyen şeydir. Hiçbir kümeyi adlandırmayan bir girdi, dağıtımının halihazırda içinde bulunduğu kümeye geri döner. Bunun yerine belirli bir kümeyi seçin ve dosya ne derse desin, her girdi oraya gider.
4. Farkı inceleyin. Sol taraf GPUStack'in tuttuğu dağıtımdır, sağ taraf ise içe aktarılacak olandır. Yanlarındaki liste belgenin tamamıyla başlar ve ardından belgedeki her dağıtımı `Create`, `Update` veya `Unchanged` olarak işaretlenmiş olarak, `Update`'in değiştireceği alanların sayısıyla birlikte indeksler; söz konusu dağıtımı tek başına düzenlemek için bir girişi veya hepsini aynı anda düzenlemek için ilk satırı seçin.
5. Listede kırmızı olarak işaretlenen her şeyi düzeltin. Bunun nedeni farkın üzerinde gösterilir ve siz belgeyi düzenlerken belge yeniden kontrol edilir; yazmayı bıraktıktan kısa bir süre sonra veya düzenleyiciden çıktığınızda hemen.
6. `Import`'a tıklayın. Plan mevcut herhangi bir dağıtımın yerini alıyorsa bunu bir kez onaylayın; iletişim kutusu değiştirilecek her dağıtımın adını verir. Her şey bir anda yazılır; herhangi bir giriş başarısız olursa dosyadan hiçbir şey yazılmaz.

`Import` belge hiçbir şey yazmazken devre dışı kalır. El değmemiş bir dışa aktarımı yeniden içe aktarmak tam olarak budur: her giriş `Unchanged` olarak okunur ve yalnızca ondan değiştirdiğiniz şey bir içe aktarmadır.

Dosyanın kendisi hiçbir zaman yüklenmez: içe aktarma isteğiyle birlikte yalnızca düzenleyicideki metin taşınır.

İçe aktarma sırasında aşağıdaki kurallar geçerlidir:

- **Üzerine yazma kasıtlı ve dar kapsamlıdır.** Adı mevcut bir dağıtımla eşleşen bir giriş, yalnızca içe aktarırken değiştirmeyi onaylamanız durumunda ve yalnızca bu dağıtım **durduruldu** ise onun yerine geçer. Durduruldu, hem kopyaların sıfıra ölçeklendiği hem de çalışan örnek kalmadığı anlamına gelir; ölçek küçültme hemen geri döner, ancak örneklerin kapatılması biraz zaman alır, bu nedenle hemen ardından yapılan bir içe aktarma sizden beklemenizi isteyebilir. Dağıtım adları kuruluşunuzun tamamında benzersiz olduğundan, bir girişteki ad, hangi kümede çalışırsa çalışsın her zaman tam olarak bir dağıtımı tanımlar.
- **Küme, girişin sıradan bir alanıdır.** Düzenleyicide `cluster_name`'yi değiştirin veya içe aktarma için farklı bir küme seçin; üzerine yazma, dağıtımı oraya taşır. Üzerine yazma yalnızca **durduruldu** dağıtıma ulaştığı için taşınacak örnek yoktur: sonrakiler o zamana kadar adlandırdığı kümeye yerleştirilir. Yeni kümeyle eşleşecek şekilde `gpu_selector` ve `worker_selector`'yi güncellemeyi veya bunları kaldırmayı unutmayın.
- **Üzerine yazma işlemi dağıtımın yerini alır, onunla birleşmez.** Dosya istenen durumdadır, dolayısıyla ondan sildiğiniz alan varsayılan durumuna geri döner. `gpu_selector` bloğunu kaldırmak, dağıtımı otomatik planlamaya nasıl döndürdüğünüzdür. Girişlerde plan işaretleri `Unchanged` hiç yazılmıyor.
- **Model rotaları `enable_model_route`'i takip eder.** Üzerine yazma işleminde bunu `false` olarak ayarlamak, dağıtımın oluşturduğu rotayı LoRA alt rotalarıyla birlikte siler. Bir LoRA bağdaştırıcısının dosyadan kaldırılması, bağdaştırıcının rotasını kendi başına siler, geri kalanı yalnız kalır. Her iki durumda da, silinen bir rota aynı zamanda başka bir dağıtıma da hizmet ediyorsa giriş reddedilir; önce diğer hedefleri ayırın.
- **Kopyalar dosyadan gelir.** Hedef ortam, dosyanın geldiği ortamdan farklı bir boyutta olduğunda düzenleyicide `replicas`'yi düzenleyin. Özenle seçilmiş GPU'larla, tıpkı dağıtımı düzenlerken yaptığınız gibi, `gpu_selector`'in hala yeni sayım için yeterli sayıda GPU'ya sahip olup olmadığını kontrol edin. Zamanlanmış ölçeklendirme etkinleştirildiğinde, zamanlama çoğaltma sayısına sahip olduğundan sayım `scaling_schedule.baseline_replicas` olarak ayarlanır.
- GPUStack'in tanımadığı bir alan sessizce bırakılmak yerine reddedilir. Bu genellikle dosyanın daha yeni bir GPUStack sürümünden geldiği anlamına gelir; hatada adı geçen alanı kaldırın ve yeniden deneyin.
- **Bu GPUStack'in sahip olmadığı bir küme, tahmin edilmez, adlandırılır.** Kümelerin başka bir şey olarak adlandırıldığı başka bir kurulumdan bir dosya içe aktarıldığında, `cluster_name`'si bilinmeyen her giriş işaretlenir. Ya burada var olan bir küme olarak yeniden adlandırın ya da her şeyi içine koymak için belirli bir küme seçin. Henüz var olmayan bir dağıtım için hiçbir kümeyi adlandırmayan bir giriş aynı şekilde işaretlenir; onu koyacak hiçbir yer yoktur.
- **Dosya hangi kuruluşa girdiğini söylüyor.** `All`'yi görüntüleyen bir platform yöneticisinin seçilmiş bir kuruluşu yoktur, dolayısıyla `Follow the file` kümelerden birini dosya adlarını okur. İki kuruluştan gelen kümeleri adlandıran bir dosya veya bu GPUStack'te bulunmayan bir dosya, herhangi bir yerde varsayılan olarak ayarlamak yerine reddedilir; içe aktarma için belirli bir küme seçin veya önce kuruluşa geçin.
- Bir giriş, dışa aktarıldığı kümeden farklı bir kümeye girdiğinde, `gpu_selector` ve `worker_selector` altındaki `gpu_ids` hala orijinal kümenin GPU'larına ve çalışanlarına atıfta bulunur. Bunları yeni kümedeki değerlerle değiştirin veya zamanlayıcının dağıtımı yapmasına izin vermek için bunları kaldırın; aksi halde GPU'lar bulunamadığı için içe aktarma başarısız olur.

## Dağıtım Revizyon Geçmişi { #deployment-revision-history }

GPUStack, bir model dağıtımı oluşturduğunuzda veya YAML'den içe aktarılan güncellemeler de dahil olmak üzere dağıtım ayarlarını değiştirdiğinizde bir yapılandırma revizyonunu kaydeder. Aynı yapılandırmayı kaydetmek, zamanlanmış ölçeklendirmeyi duraklatmak veya devam ettirmek, örnekleri yeniden başlatmak ve zamanlanmış ölçeklendirme gibi otomatik değişiklikler revizyon oluşturmaz. Mevcut dağıtımlar, geçmişleri ilk açıldığında veya yapılandırmaları ilk kez güncellendiğinde ilk anlık görüntüyü alır; önceki yapılandırmalar kurtarılamaz.

1. Model dağıtım listesinde dağıtımı bulun.
2. `Operations` menüsünü açın ve `Revision history`'i seçin.
3. Salt okunur YAML farklılığında önceki sürümüyle karşılaştırmak için bir düzeltme seçin. Örneğin, `v2` seçildiğinde solda `v1` ve sağda `v2` gösterilir. En son revizyon incelemeye hazır durumda kalır. Bir ilk revizyon veya öncülü silinmiş veya budanmış bir revizyon, bunun yerine tam konfigürasyonunu gösterir.
4. Eski bir revizyonun satırındaki `Roll back to this revision` eylemine tıklayın. Aynı çekmece, mevcut dağıtım yapılandırmasını bu revizyonla karşılaştıran `Roll back <name> to vN` başlıklı bir önizlemeye geçer. Geri alma incelenirken revizyon listesi gizlenir. Önceki seçiminize ve sayfanıza dönmek için `Back to history`'e tıklayın. Yapılandırmayı uygulamak için önizlemede doğrudan `Roll back` seçeneğine tıklayın; ek onay iletişim kutusu gösterilmez. En son revizyonun geri alma eylemi yoktur. Yapılandırmalar eşleştiğinde geri alma devre dışı bırakılır. Sunucu, yapılandırmayı normal düzenlemeyle aynı kuralları kullanarak doğrular. Yapılandırmayı değiştiren başarılı bir geri alma, yeni bir revizyon oluşturur.

Geri alma, arka uç yapılandırması, ortam değişkenleri, yerleştirme, kopyalar, zamanlanmış ölçeklendirme ve LoRA bağdaştırıcıları dahil olmak üzere dağıtım ayarlarını geri yükler. Dağıtımın adını, açıklamasını, organizasyonunu, kümesini, erişim ilkesini ve revizyon saklama ayarını korur. Bağımsız model rota ayarları geri yüklenmez; türetilmiş LoRA rotaları geri yüklenen bağdaştırıcıları takip eder.

Kaydedilen anlık görüntüde alan yoksa geri alma, o alanın geçerli değerini korur. Geri alma önizlemesi, gönderilecek yapılandırmayla eşleşecek şekilde korunan değeri içerir. Kaydedilen anlık görüntüdeki açık bir `null` alanı yine de temizler. Uyumsuz yapılandırma şeması değişiklikleri, geçmişin taşınmasını gerektirir.

Revizyon görünümleri, YAML içe ve dışa aktarmayla aynı yapılandırma formatını kullanır: ayarlanmamış model alanları atlanır ve alanlar şema sırasını takip eder. `false`, `0` gibi açık değerler ve boş geçersiz kılmalar görünür durumda kalır. Bir ayarın temizlenmesi, onun çizgisini farktan kaldırmak gibi görünür. Bu biçimlendirme yalnızca görüntülenen yapılandırmayı etkiler; geri alma, kaydedilen anlık görüntünün tamamını kullanır.

Önizlemede, altbilginin solunda, eylem düğmelerinin yanında varsayılan olarak seçilen bir `Restart on rollback` onay kutusu bulunur. Geri yüklenen yapılandırmayı kaydettikten sonra kullanıcı arayüzü, örneklerini yeniden oluşturmak için dağıtımın yeniden başlatılmasını ister. Yeniden başlatmadan kaydetmek için onay kutusunun işaretini kaldırın; Yapılandırma değişiklikleri örneklerin yeniden oluşturulmasını gerektirdiğinde sarı bir bildirim görünür. Geri alma ve istenen yeniden başlatma başarılı olduğunda çekmece kapanır ve dağıtım listesi yenilenir. Yapılandırma kaydedilirse ancak yeniden başlatma başarısız olursa çekmece açık kalır, her iki sonucu da rapor eder ve geri alma işlemini tekrarlamayan `Retry restart` sağlar. Başarılı bir yeniden deneme aynı zamanda çekmeceyi kapatır ve listeyi yeniler. Geri alma API'sinin doğrudan çağrılması yalnızca yapılandırmayı geri yükler.

En son revizyon, en yeni kaydedilen konfigürasyonu tanımlar; çalışan tüm örneklerin bunu kullanacağını garanti etmez. Yeniden başlatma eşzamansızdır ve sıfır kesinti süresine sahip bir geçiş sağlamaz. Geri alma işlemini gönderdiğinizde yapılandırmanın geçerliliği kontrol edilir, böylece başvurulan kaynaklar veya desteklenen ayarlar değiştiyse önizleme yine de uygulanamayabilir.

### Revizyon Saklama { #revision-retention }

Dağıtımlar, varsayılan olarak en son sürüme ek olarak **10** eski revizyonları da korur. Dağıtım formu bu ayarı göstermez. İleri düzey kullanıcılar API aracılığıyla `revision_history_limit` ayarını yapabilir. Yalnızca en son revizyonu saklamak için bunu **0** olarak ayarlayın. Limitin düşürülmesi, konfigürasyon revizyonu oluşturmadan en eski girişlerden başlayarak fazla geçmişi anında kaldırır.

Geçmiş çekmecesi bireysel revizyon silme olanağı sunmaz; otomatik temizleme, dağıtımın saklama sınırına uyar. API eski revizyonların silinmesine izin verir ancak en son revizyon silinemez. Dağıtımın silinmesi, tüm düzeltmelerini kaldırır. Kaldırılan revizyonlar kurtarılamaz.

Geçmiş, dağıtımın düzenlenmesiyle aynı organizasyona ve erişim kısıtlamalarına sahiptir ve çalışan ve küme SİSTEM sorumluları tarafından kullanılamaz. Ortam değişkenleri orijinal değerleriyle saklanır; dolayısıyla eski bir revizyon, söz konusu revizyon silinene veya otomatik olarak budanana kadar kimlik bilgilerini koruyabilir.

## Model Örneğini Görüntüle { #view-model-instance }

1. Dağıtım listesi sayfasında kontrol etmek istediğiniz model dağıtımını bulun.
2. Dağıtımın örnek listesini görüntülemek için `>` sembolüne tıklayın.

## Model Örneğini Sil { #delete-model-instance }

1. Dağıtım listesi sayfasında kontrol etmek istediğiniz model dağıtımını bulun.
2. Dağıtımın örnek listesini görüntülemek için `>` sembolüne tıklayın.
3. Silmek istediğiniz model örneğini bulun.
4. `Operations` sütunundaki model örneği için üç nokta düğmesini tıklayın, ardından `Delete`'i seçin.
5. Silme işlemini onaylayın.

!!! note

    Bir model örneği silindikten sonra GPUStack, gerekirse dağıtımın beklenen kopyalarını karşılamak için yeni bir örneği yeniden oluşturacaktır.

## Model Örnek Günlüklerini Görüntüle { #view-model-instance-logs }

1. Dağıtım listesi sayfasında kontrol etmek istediğiniz model dağıtımını bulun.
2. Dağıtımın örnek listesini görüntülemek için `>` sembolüne tıklayın.
3. Kontrol etmek istediğiniz model örneğini bulun.
4. `Operations` sütunundaki model örneği için `View Logs` düğmesine tıklayın.

## Performansla İlgili Yapılandırma { #performance-related-configuration }

GPUStack, model çıkarım performansını optimize etmek için aşağıdaki yapılandırma seçeneklerini sunar.

### Genişletilmiş KV Önbelleği { #extended-kv-cache }

KV önbelleğini CPU belleğine veya uzak depolama birimine boşaltmak için genişletilmiş KV önbelleğini etkinleştirebilirsiniz. Bu özellik, özellikle uzun içerik uzunlukları gerektiren sınırlı GPU belleğine sahip kurulumlar için kullanışlıdır. Temel olarak GPUStack, [LMCache](https://github.com/LMCache/LMCache) veya HiCache gibi seçilen arka uç ve çerçevenin desteklediği KV önbellek boşaltma işlemini yapılandırır.

Mevcut seçenekler:

- **RAM-VRAM Oranı**: KV önbellek için kullanılan sistem RAM'inin GPU VRAM'ına oranı. Örneğin 2.0, RAM'deki önbelleğin GPU VRAM'inden iki kat daha büyük olabileceği anlamına gelir.
- **Maksimum RAM Boyutu**: Sistem belleğinde (GiB) depolanan KV önbelleğinin maksimum boyutu. Ayarlanırsa bu değer `RAM-to-VRAM Ratio`'yi geçersiz kılar.
- **Önbellek Parçacıklarının Boyutu**: KV önbellek parçası başına token sayısı. Konektörün tüm KV bloklarını boşalttığı ve bunları motorun blok boyutunda yeniden kullandığı CANN'deki vLLM için geçerli değildir.

Bu özellik yalnızca belirli arka uçlar ve çerçeveler için çalışır.

#### Uyumluluk Matrisi { #compatibility-matrix }

| Arka uç | Çerçeve |
| ------- | ---------------- |
| vLLM | CUDA, ROCm, CANN |
| SGLang | CUDA, ROCm, CANN |

CANN'de vLLM ile genişletilmiş KV önbelleği, vLLM Ascend v0.21.0 veya üzerini gerektirir ve Atlas 300I Duo'da (310P) mevcut değildir.

## Planlama Yapılandırması { #scheduling-configuration }

### Zamanlama Modu { #schedule-mode }

#### Otomatik { #auto }

GPUStack, mevcut kaynak kullanılabilirliğine göre model örneklerini otomatik olarak uygun GPU'lara/Çalışanlara planlar.

- **Yerleşim Stratejisi**

Yayılma: Tüm kümenin kaynaklarının tüm çalışanlar arasında nispeten eşit şekilde dağıtılmasını sağlayın. Tek bir çalışan üzerinde daha fazla kaynak parçalanmasına neden olabilir.

Binpack: Çalışanlar/GPU'lar üzerindeki kaynak parçalanmasını azaltarak küme kaynaklarının genel kullanımına öncelik verin.

- **Çalışan Seçici**

Yapılandırıldığında zamanlayıcı, model örneğini belirtilen etiketleri içeren çalışana dağıtacaktır.

1. `Workers` sayfasına gidin ve istediğiniz çalışanı düzenleyin. Etiketler bölümüne ekleyerek çalışana özel etiketler atayın.

2. `Deployments` sayfasına gidin ve `Deploy Model` düğmesine tıklayın. `Scheduling` bölümünü genişletin ve önceden atanmış çalışan etiketlerini `Worker Selector` yapılandırmasına girin. Dağıtım sırasında Model Örneği, bu etiketlere göre ilgili çalışana tahsis edilecektir.

#### Manuel { #manual }

Bu zamanlama türü, kullanıcıların model örneğinin hangi GPU'ya dağıtılacağını belirtmesine olanak tanır.

- **GPU Seçici**

  Listeden bir veya daha fazla GPU seçin. Model örneği, kaynaklar izin verirse seçilen GPU'ya dağıtım yapmayı deneyecektir.

- **Çoğaltma başına GPU**

Otomatik: Sistem, varsayılan olarak ikinin katlarını kullanarak ve seçilen GPU'lar tarafından sınırlandırılarak kopya başına GPU sayısını otomatik olarak hesaplar.

Manuel: Açılır menüden her kopyanın kullanması gereken GPU sayısını seçin.

## Gelişmiş Yapılandırma { #advanced-configuration }

GPUStack, model dağıtımı için özel yapılandırmaları destekler.

### Model Kategorisi { #model-category }

Model kategorisi, modelleri düzenlemenize ve filtrelemenize yardımcı olur. Varsayılan olarak GPUStack, modelin meta verilerine göre model kategorisini otomatik olarak algılar. Kategoriyi açılır listeden seçerek de özelleştirebilirsiniz.

### Arka Uç Parametreleri { #backend-parameters }

Modeli çalıştırırken özelleştirmek istediğiniz arka uç için parametreleri girin. Desteklenen parametre formatları:

| Yöntem | Örnek | Açıklamalar |
|------------------|----------------------------------------------------|-----------------------------------------------------------------------------|
| Eşittir İşareti Bölme | `--hf-overrides={"architectures": ["NewModel"]}` | - |
| Boşluk Bölme | `--hf-overrides '{"architectures": ["NewModel"]}'` | `shell-like` stil bölmeyi destekler (ör. boşluk içeren değerler için). |
| Ayrı Alanlar | `--max-model-length`, `8192` | Parametre adını ve değerini iki ayrı öğe olarak girin.                       |

Desteklenen parametrelerin tam listesi için lütfen [Çıkarım Arka Uçları](built-in-inference-backends.md) bölümüne bakın.

### Ortam Değişkenleri { #environment-variables }

Modeli çalıştırırken kullanılan ortam değişkenleri. Bu değişkenler başlangıçta arka uç işlemine aktarılır.

### LoRA Adaptörleri { #lora-adapters }

LoRA (Düşük Sıralı Uyarlama), tüm ağırlıkları yeniden eğitmek yerine küçük bağdaştırıcı dosyaları yükleyerek temel modeli belirli bir alana uyarlayan, parametre açısından verimli bir ince ayar yöntemidir. GPUStack, birden fazla LoRA bağdaştırıcısını tek bir konuşlandırılmış temel LLM'ye bağlamanıza olanak tanır ve bağdaştırıcı başına otomatik olarak bir Model Rota oluşturur.

!!! note

    LoRA yalnızca `vLLM`, `SGLang` ve `Ascend MindIE` arka uçlarında desteklenir. `LoRA Adapters` yapılandırması diğer arka uçlarda göz ardı edilir.

Dağıtım formundaki `Advanced` bölümünü genişletin, `LoRA Adapters`'i bulun ve bağdaştırıcıları tek tek ekleyin:

- Açılır menüden bir adaptör seçin. Liste aramayı destekler. Bağdaştırıcınız gösterilmiyorsa, önce `Hugging Face` veya `ModelScope`'de modelin gerçekten bir model sağladığını doğrulayın, ardından seçmek için depo kimliğini arama kutusuna yapıştırın.
- `LoRA name`: Çıplak adaptör adını girin, örneğin `alpaca`.

Dağıtımdan sonra GPUStack, her LoRA bağdaştırıcısı için otomatik olarak `<base-model-name>:<adapter-name>` adlı bir Model Rota oluşturur. Bağdaştırıcıları değiştirmek için OpenAI uyumlu API'nin `model` alanını ilgili rota adına ayarlayın. Tüm bağdaştırıcılar aynı GPU örneğini paylaşır. GPUStack, `--enable-lora` gibi arka uç başlangıç argümanlarını otomatik olarak enjekte eder, dolayısıyla bunları genellikle `Backend Parameters` altında tekrarlamanıza gerek kalmaz.

!!! note

    Dağıtılmış bir modelin LoRA yapılandırmasını değiştirdikten sonra, yeni yapılandırmanın etkili olması için ilgili örneklerin yeniden başlatılması gerekir.

Çağırma örnekleri ve arka uç başına uyumluluk ayrıntıları da dahil olmak üzere uçtan uca iş akışının tamamı için [LoRA Adaptörleriyle Model Sunma](../tutorials/serving-with-lora-adapters.md) eğitimine bakın.

### CPU Boşaltmasına İzin Ver { #allow-cpu-offloading }

!!! note

    Yalnızca özel arka uçlar için kullanılabilir.

CPU boşaltma etkinleştirildiğinde, GPU kaynakları yetersizse GPUStack CPU belleğini tahsis edecektir. Çıkarım arka ucunu hibrit CPU+GPU veya tam CPU çıkarımını kullanacak şekilde doğru şekilde yapılandırmanız gerekir.

### Çalışanlar Genelinde Dağıtılmış Çıkarıma İzin Ver { #allow-distributed-inference-across-workers }

!!! note

    vLLM, SGLang ve MindIE arka uçları için kullanılabilir.

Birden fazla çalışan arasında dağıtılmış çıkarımı etkinleştirin. Birincil Model Örneği, bir veya daha fazla diğer çalışandaki arka uç örnekleriyle iletişim kurarak hesaplama görevlerini onlara aktarır.

### Hata Durumunda Otomatik Yeniden Başlatma { #auto-restart-on-error }

Bir hatayla karşılaşırsa model örneğinin otomatik olarak yeniden başlatılmasını etkinleştirin. Bu özellik, model örneğinin yüksek düzeyde kullanılabilirliğini ve güvenilirliğini sağlar. Bir hata oluşursa GPUStack, üstel bir geri çekilme stratejisi kullanarak model örneğini otomatik olarak yeniden başlatmayı deneyecektir. Yeniden başlatma girişimleri arasındaki gecikme, maksimum 5 dakikalık aralığa kadar katlanarak artar. Bu yaklaşım, kalıcı hatalar durumunda sistemin sık sık yeniden başlatılarak bunaltılmasını önler.

### Genel Proxy'yi Etkinleştir { #enable-generic-proxy }

OpenAI uyumlu API'lerle entegrasyon yaygın bir uygulama olsa da, kullanıcıların kullanım durumları için farklı gereksinimleri olabilir. GPUStack, OpenAI uyumlu olanlar dışındaki tüm çıkarım API'lerini destekler ve onu AI uygulama geliştirme için daha esnek hale getirir.

GPUStack, Genel Proxy etkinleştirildiğinde hedef modele yönelik iki yol sunar. **yol tabanlı form** önerilir; **başlık tabanlı form** geriye dönük uyumluluk nedeniyle tutuldu ve kullanımdan kaldırıldı.

#### Yol tabanlı form (önerilir) { #path-based-form-recommended }

Sayısal model rota kimliğini proxy önekine ekleyin — `/model/proxy/<model_route_id>/<upstream-path>` — ve GPUStack isteği o rotanın hedeflerine gönderir. Ekstra başlığa gerek yoktur.

```bash
# Assume the model route id is 42.
curl http://<server-url>/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

Ağ geçidi, iletmeden önce `/model/proxy/<id>` şeritler, böylece yukarı akış çıkarım sunucusu şunu görür:

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

Model rota kimliği, yeniden adlandırmalarda sabittir ve model detay sayfasında bulunabilir veya `GET /v2/model-routes` adresinden alınabilir.

#### Başlık tabanlı form (kullanımdan kaldırıldı) { #header-based-form-deprecated }

!!! warning "Kullanımdan kaldırıldı"
    `/model/proxy` + `X-GPUStack-Model` formu geriye dönük uyumluluk amacıyla tutulmaktadır ve gelecekteki bir sürümde kaldırılacaktır. Yukarıdaki yol tabanlı forma geçin.

```bash
curl http://<server-url>/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

Yönlendirmeden önce `/model/proxy` yol öneki çıkarılır. Ağ geçidinin hedef modeli çözebilmesi için JSON gövdesinde `X-GPUStack-Model` başlığını veya `model` özelliğini sağlamanız gerekir. Yukarı akış çıkarım sunucusunda istek şöyle görünür:

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```
