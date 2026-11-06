---
translation_status: translated
---

# Önbellek Hizmeti Yönetimi { #cache-service-management }

Önbellek hizmeti, model dağıtımlarının kendi KV önbelleklerini tutmak yerine bağlandığı paylaşılan bir KV önbelleğidir. Önbellek çıkarım motorunun dışında bulunur; böylece aynı çalışandaki dağıtımlar ve replikalar birbirlerinin öneklerini yeniden kullanır. L2 depolama arka ucu bu yeniden kullanımı yeniden başlatmalar ve düğümler arasında da sağlar.

Önbellek hizmetleri yalnızca yerleşik vLLM ve SGLang arka uçlarına eklenir ve bir dağıtım yalnızca kendi kümesindeki bir önbellek hizmetine bağlanabilir.

## Önbellek Hizmeti Oluşturun { #create-a-cache-service }

1. `Cache Service` sayfasına gidin.
2. `Add Cache Service` düğmesine tıklayın ve bir sağlayıcı seçin.
3. `Name` girin.
4. Bir `Version` seçin veya `Custom` seçip `Container Image` girin.
5. `RAM Size (GiB)` — önbellek sunucusunun belleğinde tutulan önbellek kapasitesi gibi sağlayıcı ayarlarını girin.
6. _(İsteğe bağlı)_ Hizmetin çalışacağı çalışanları sınırlamak için `Worker Label Selector` ayarlayın.
7. _(İsteğe bağlı)_ Önbelleği daha büyük bir katmana (yerel dizin, Redis veya Valkey veya S3 uyumlu nesne deposu) aktarmak için `L2 Storage Backend` ekleyin. Birkaç girişe sırayla öncelik verilir: okumalar ilkini tercih eder, yazmalar hepsine gider.
8. _(İsteğe bağlı)_ `Advanced`'in altına `Parameters` ve `Environment Variables` ekleyin. Birden fazla bileşeni çalıştıran bir hizmet, her biri farklı bir ikili dosya çalıştırdığından, bileşen başına ayrı bir parametre listesi tutar.
9. `Save` düğmesine tıklayın.

!!! tip

    Mevcut bir önbellek hizmetinde yapılan değişiklikler yalnızca örnekleri silinip yeniden oluşturulduktan sonra geçerli olur.

## Bir Dağıtım Ekleme { #attach-a-deployment }

1. Yerleşik vLLM veya SGLang arka ucunu kullanarak bir model dağıtımı oluşturun veya düzenleyin.
2. `Enable Extended KV Cache`'ı seçin.
3. `Cache Backend`'ı önbellek hizmetine ayarlayın. Yalnızca aynı kümedeki ve seçilen arka uçla uyumlu önbellek hizmetleri listelenir.
4. `Save` düğmesine tıklayın.

Her örnek bağlandığı önbellek hizmetini ve son isabet oranını bildirir. Örneğin, çalışanında kullanılabilir bir önbellek örneği bulunmadığı için önbelleksiz başlayan bir örnek hata vermek yerine paylaşılan KV önbelleğinin etkin olmadığını bildirir.

## LMCache ile Hibrit Dikkat Modellerini Çalıştırma { #running-hybrid-attention-models-with-lmcache }

Hibrit bir model, tekrarlayan katmanları (Mamba veya Gated-DeltaNet (GDN) doğrusal dikkat) tam dikkat katmanlarıyla birleştirir. Qwen3.5 ve Qwen3.6 serileri, Qwen3-Next, Kimi-Linear ve Kimi K3 bu türdendir.

Tekrarlayan katman, token başına bir anahtar/değer çifti yerine sabit boyutlu bir durum tutar; bu durumun anlık görüntüsü yalnızca blok sonunda alınabilir. Bu nedenle motor, boyutu modelden belirlenen ve normalde kullanılan birkaç token'dan çok daha büyük bloklarla çalışır. LMCache, ancak önbelleği bu blok sınırlarıyla uyumlu birimlerde saklarsa modele hizmet verebilir. İki taraf da bunu otomatik ayarlamaz: ayarlar olduğu gibi bırakılırsa motor ya başlamaz ya da önbelleğe bağlansa bile hiç isabet kaydetmez.

### Adım 1: Motorun seçtiği blok boyutunu bulun { #step-1-find-the-block-size-the-engine-chose }

vLLM, bir dikkat sayfası en az bir yinelenen durum sayfası kadar büyük olana kadar dikkat bloğu boyutunu yükseltir ve dağıtım başladığında sonucu günlüğe kaydeder:

```
Setting attention block size to 544 tokens to ensure that attention page size is >= mamba page size.
```

Bu sayı `N` değeridir. Modele, veri türüne ve paralelliğe bağlıdır; dolayısıyla varsaymak yerine kendi dağıtımınızın günlüğünden okuyun. Aşağıdaki değerler, modelin kendi hassasiyetiyle tek GPU'da yapılan çalıştırmalardan alınmıştır; başvuru tablosu olarak değil, gördüğünüz sonucu kontrol etmek için verilmiştir.

| Modeli | `N` |
| ----------------------------------------- | --- |
| `Qwen/Qwen3.5-0.8B`                       | 544 |
| `Qwen/Qwen3.6-27B`                        | 784 |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 944 |
| `moonshotai/Kimi-K3`                      | 768 |

### Adım 2: Önbellek hizmetini yapılandırın { #step-2-configure-the-cache-service }

- `Chunk Size`'ı `N`'e veya bunun katlarına ayarlayın. Önbellek sunucusunun yığın boyutu, motorun blok boyutunun katı olmalıdır, aksi takdirde önbellek kaydı başarısız olur.
- Önbellek sunucusunun `Parameters`'ine `--separate-object-groups` ekleyin, böylece her dikkat penceresi kendi nesne grubu olarak saklanır. Sunucu bunu varsayılan olarak kapalı tutar.

!!! note

    Parça boyutu dağıtıma değil, önbellek hizmetine aittir. Bir `N` değeri için ayarlanan hizmet, blok boyutu bu değeri tam bölen normal modellere de hizmet verir; ancak `N` değerleri farklı iki hibrit modelin her biri için ayrı bir önbellek hizmeti gerekir.

### 3. Adım: Modeli hizmete dağıtın { #step-3-deploy-the-model-against-the-service }

Modeli [Dağıtım Ekle](#attach-a-deployment)'deki gibi dağıtın ve 2. adımdaki önbellek hizmetini `Cache Backend` olarak seçin. Çalışan bir örnek, bağlı olduğu hizmeti ve gördüğü isabet oranını bildirir.

### Referans { #reference }

[Hibrit Dikkat Modelleri](https://docs.lmcache.ai/mp/hybrid_models.html) — aynı gereksinimin LMCache belgelerindeki açıklaması ve sunucunun bunun için kabul ettiği bayraklar.

## Önbellek Sağlayıcılarını Özelleştirin { #customize-cache-providers }

Seçilebilen sağlayıcılar bu sürümle gelenleri ve yüklü uzantıların eklediklerini kapsar. Platform yöneticileri farklı bir imajı sabitlemek, sürüm eklemek veya GPUStack ile gelmeyen bir sağlayıcı tanımlamak için bu kümenin tamamını kendi belgeleriyle değiştirebilir.

1. `Cache Service` sayfasına gidin.
2. `Manage Providers` düğmesine tıklayın.
3. `Provider Source` çekmecesinde üç kaynaktan birini seçin:

    - `Yaml File`: içeriği düzenleyiciye yapıştırın veya içine bir dosya yüklemek için `Import`'e tıklayın.
    - `URL`: `Source URL`'i doldurun. Sunucu, kaydettiğinizde ve `Update Now`'e tıkladığınızda onu getirir. `http(s)` olmalı, kimlik bilgileri taşımamalı, depo sayfası yerine dosyanın kendisine işaret etmeli, UTF-8 olmalı ve 4 MB'tan büyük olmamalıdır.
    - `Embedded`: bu sürümün taşıdığı sağlayıcılara geri dönün.

4. `Save` düğmesine tıklayın.

!!! warning

    **Boş bir düzenleyiciden değil, yerleşik dosyadan başlayın.** Kendinize ait bir belge, tüm setin doğrudan yerini alır - yüklü bir uzantının katkıda bulunduğu sağlayıcılar da dahil olmak üzere, beyan etmediği her sağlayıcı hizmet dışı kalır. Bu kurulumun şu anda sunduğu hizmetleri indirmek için düzenleyici başlığındaki `Built-in File` seçeneğine tıklayın ve bunu düzenleyin.

Belge kaydedilirken doğrulanır. Bu sürümün okuyamadığı bir bildirim, olduğu gibi gösterilecek bir yer tutucu veya var olmayan bir alan gibi hizmet veremeyecek herhangi bir içerik varsa belge tümüyle reddedilir. Özellikle sonuncusu önemlidir: bilinmeyen anahtarlar yok sayıldığından `run_command` yerine `run_cmd` yazılan bileşen aksi halde hiçbir komut olmadan başlatılırdı.

Kayıt işlemi, bir önbellek hizmetinin hâlâ kullandığı sağlayıcıyı veya sürümü kaldıracaksa da reddedilir. Mesajda etkilenen her hizmet belirtilir. Önce bu hizmetleri silin veya yeni belgede bulunan bir sağlayıcıya ve sürüme geçirin.

!!! note

    Çalışan örnekler başlatıldıkları bildirimi korur. Yeni bildirim örnek yeniden oluşturulduğunda geçerli olur; GPUStack yükseltmesinde sağlayıcının değişmesi de aynı şekilde uygulanır.

    Sağlayıcılar zamanlanmış olarak güncellenmez: URL kaynağı kaydedildiğinde ve `Update Now` seçildiğinde okunur. Kendi belgeniz yapılandırılmışken GPUStack yükseltmesi veya yeni uzantıyla gelen sağlayıcılar, siz bunları belgeye eklemedikçe görünmez. Eklenenleri görmek için yerleşik dosyayı yeniden indirin.
