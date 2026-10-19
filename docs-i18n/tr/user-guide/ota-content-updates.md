---
translation_status: translated
---

# OTA İçerik Güncellemeleri { #ota-content-updates }

GPUStack, model kataloğunu, Topluluk Arka Uç Pazaryerini ve yerleşik arka uç sürümlerini resmi bir OTA sunucusundan alır, böylece yeni modeller ve arka uç sürümleri, yükseltme veya yeniden başlatma gerekmeden çalışan bir GPUStack hizmetine ulaşır.

Güncellemeler varsayılan olarak açıktır ve içerik tek bir ana bilgisayardan gelir, `https://ota.gpustack.ai/latest`. GPUStack hizmetiniz kısıtlı bir ağda çalışıyorsa, bu tek ana bilgisayara izin vermek yeterlidir. Ayrıca, kendi bakımını yaptığınız ve o ağdan erişilebilen bir dosyaya da geçebilirsiniz. Üçüncü seçenek, OTA sunucusunun kendi kopyasını barındırmak ve GPUStack'i başlangıçta `--ota-server-url` ile ona yönlendirmektir. Her üç içerik türü de otomatik olarak güncellenmektedir; yalnızca kopyanızdan okunurlar. Bakınız [start](../cli-reference/start.md).

Kaynaklar içeriğin sunulduğu sayfada yapılandırılır. Katalog `Model Service > Catalog` sayfasındadır ve her iki tür arka uç içeriği de `Model Service > Inference Backends` sayfasındadır. Her ikisi de yalnızca platform yöneticilerinin görebileceği `Manage Sources` düğmesini kullanır.

## OTA ile Güncellenen İçerikler { #what-updates-over-the-air }

| İçerik | OTA sunucusunun yayınladığı şeyler |
| --- | --- |
| Model kataloğu | Yeni ve revize edilmiş katalog girişleri |
| Topluluk Arka Uç Pazaryeri | Yeni ve revize edilmiş topluluk arka uç tanımları |
| Yerleşik arka uç sürümleri | vLLM ve SGLang gibi yerleşik arka uçlar için yeni sürümler ve bunların görüntüleri |

Her tür aynı anda bir **kaynak** tarafından sunulur ve tüm bu içeriği tek bir kaynak dosya taşır. Kaydedildiğinde, bu türün kendisine eklemek yerine o anda hizmet ettiği şeyin yerini alır, böylece yeni dosyada eksik olan tüm girişler onunla birlikte hizmet dışı bırakılır.

| Kaynak | Sunuyor | Otomatik güncellemeler |
| --- | --- | --- |
| `URL` | Sunucunun alıp sakladığı bir dosya. | İsteğe bağlı |
| `Yaml File` | Yapıştırdığınız bir dosya. Yalnızca model kataloğu. | Hayır |
| `Embedded` | Yalnızca bu sürümün paketlendiği şey. Ağ erişimi yok. | Hayır |

Bir kaynağı boş bırakın ve bu tür kullanıcı arayüzünde `Official` etiketli OTA sunucusunu takip etmeye devam edin. Kendi URL'nizi veya Yaml dosyanızı yapılandırdığınızda, bu tür OTA sunucusunu takip etmeyi bırakır ve sağladığınız şeyi kullanır.

Kendinize ait bir `Source URL` şu koşulları karşılamalıdır:

- `http(s)` olmalı ve kimlik bilgisi taşımamalıdır.
- Bir depo sayfasına değil, dosyanın kendisine işaret etmelidir.
- Dosya 4 MB'tan büyük olmamalı ve UTF-8 olmalıdır.

Dosya sunucu tarafından alınır, dolayısıyla hiçbir çalışanın bu adrese erişmesine gerek yoktur.

## Katalog Kaynağını Yapılandırma { #configure-the-catalog-source }

1. `Model Service > Catalog` sayfasına gidin.
2. `Manage Sources` düğmesine tıklayın.
3. `Catalog Source` çekmecesinde iki kaynak türünden birini seçin:

    - `URL`: `Source URL`'i doldurun.

        ![Ekran görüntüsü: URL kaynağına sahip Katalog Kaynağı çekmecesi](../assets/ota-content-updates/1-catalog-source-url.png)

    - `Yaml File`: içeriği `Content`'e yapıştırın veya düzenleyiciye bir dosya yüklemek için `Import`'e tıklayın.

        ![Ekran görüntüsü: Yaml Dosya düzenleyicisine sahip Katalog Kaynağı çekmecesi](../assets/ota-content-updates/2-catalog-source-yaml.png)

4. `Save`'a tıklayın.

!!! note

    Boş bir dosyadan değil, zaten mevcut olan içerikten başlayın. `Yaml File`'de, OTA sunucusunun şu anda ne sunduğunu indirmek için `Official File`'e tıklayın. `URL`'de, `Source URL` kutusu halihazırda takip ettiği adresle zaten doldurulmuştur, dolayısıyla onu oradan kopyalayabilirsiniz.

## Arka Uç Kaynaklarını Yapılandırma { #configure-the-backend-sources }

İki tür arka uç içeriği bağımsız olarak güncellenir ve her birinin kendi kaynağı vardır. Bir çekmece her ikisini de panel olarak yapılandırır:

1. `Model Service > Inference Backends` sayfasına gidin.
2. `Manage Sources` düğmesine tıklayın.
3. `Backend Source` çekmecesinde ihtiyacınız olan paneli genişletin:
    - `Built-in backend source`: yerleşik arka uçların (vLLM, SGLang, MindIE, VoxBox) sürümleri ve görüntüleri.
    - `Community backend source`: Topluluk Arka Uç Pazarı.
4. `URL`'ı seçin ve `Source URL`'i doldurun. Her iki tür de yapıştırılan içeriği değil, yalnızca bir URL'yi destekler.
5. Bu paneldeki `Save` düğmesine tıklayın. Her panel kendi kendine kaydeder.

![Ekran görüntüsü: Her iki panelin de genişletilmiş olduğu Arka Uç Kaynağı çekmecesi](../assets/ota-content-updates/3-backend-source.png)

!!! warning

    Bir güncelleme, dağıtımın hâlâ kullandığı arka uç sürümünü kaldıracaksa kaydetme işlemi reddedilir ve mesajda modeller listelenir. Önce bu modelleri sıfıra ölçeklendirin veya yeni dosyanın hâlâ taşıdığı bir sürüme taşıyın ve ardından tekrar kaydedin.

!!! note

    Yeni dosyada bulunmayan bir topluluk arka ucu, etkinse veya elle eklediğiniz bir sürümü varsa özel arka uç olarak korunur; üzerindeki modeller çalışmaya devam eder. Kaynak onu yeniden yayımlarsa tekrar topluluk arka ucuna dönüşür, ancak yeniden etkinleştirmeniz gerekir. Devre dışı ve elle değiştirilmemiş olan ise kaldırılır; dolayısıyla düzenlediğiniz `default_env`, arka uç geri geldiğinde geri yüklenmez.

## Güncelleme Zamanlamasını Kontrol Edin { #control-the-update-schedule }

Her kaynağın kendi güncelleme ayarları vardır:

- `Auto-update`: güncellemelerin sabit aralıklarla otomatik olarak kontrol edilip edilmeyeceği. Bir tür OTA sunucusunu takip ederken varsayılan olarak açıktır. Kendi `URL`'inize geçtikten sonra varsayılan olarak kapalıdır, bu nedenle onu kendiniz seçin.
- `Update Interval (hours)`: `Auto-update` seçildiğinde görünür. Otomatik kontroller arasındaki aralıktır, varsayılan olarak 12 saattir ve bunu değiştirebilirsiniz.
- `Update Now`: Arayı beklemeden hemen bir kez kontrol edin.

`Source URL`'ı değiştirdikten sonra `Save`'e tıklayın. Yeni adresi saklar ve hemen getirir. İçerik değişmediğinde hiçbir şey tekrar indirilmez.

!!! note

    Geçici olarak erişilemeyen bir OTA sunucusu hizmeti etkilemez. Halihazırda getirilen içerik sunulmaya devam eder, hata kaydedilir ve bir saat içinde bir sonraki deneme yapılır. OTA sunucusuna hiç ulaşmamış bir küme, gömülü içeriği sunar. Bu içerik tamdır, yalnızca mevcut GPUStack sürümünüzden sonra yayınlanan hiçbir şeyi içermez.

## Güncellemeleri Durdurun ve Gömülü İçerik Sunun { #stop-updates-and-serve-embedded-content }

`Auto-update`'in temizlenmesi yalnızca kontrolleri durdurur ve zaten getirilmiş olan içerik sunulmaya devam eder. Getirilen içeriğin sunulmasını tamamen durdurmak için `Embedded`'i seçin ve kaydedin. Bu tür, yeni bir çevrimdışı yüklemeyle aynı şekilde, yalnızca sürümün paketlendiği şeye hizmet eder.

Örneğin bir katalog girişi kümenize uymadığında veya bazı arka uç sürümlerin görselleri ortamınıza çekilemediğinde `Embedded`'a geçin.

- **Hiçbir konfigürasyon kaybolmaz.** Girdiğiniz URL ve içerik saklanır, yalnızca hizmet dışı bırakılır. Bunları geri yüklemek için tekrar `URL` veya `Yaml File`'i seçin.
- **Ağa gerek yok.** Kullanmayı durdurmak istediğiniz dosyaya zaten ulaşılamıyor olsa bile `Embedded`'ye geçebilirsiniz.
- Bu tek tür içeriği etkiler, her sunucuda anında etkili olur ve yeniden başlatma gerektirmez.
- OTA sunucusunu takip etmeye geri dönmek için `Reset to Official Source`'e tıklayın. Bu, girdiğiniz şeyi temizler.

!!! warning

    `Embedded` seçeneğine geçmek kullanılabilir arka uç sürümlerini azalttığından geçiş aynı nedenle reddedilebilir. Geçmeden önce hiçbir dağıtımın yalnızca indirilen içerikte bulunan bir sürümü kullanmadığını doğrulayın.
