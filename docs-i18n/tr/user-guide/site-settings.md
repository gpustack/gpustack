---
translation_status: translated
---

# Site Ayarları { #site-settings }

Platform yöneticileri GPUStack dağıtımının marka görünümünü **Erişim Kontrolü → Site ayarları** bölümünden özelleştirebilir. Ayarlar tüm kullanıcılar ve kuruluşlar tarafından paylaşılır; sunucu veritabanında saklanır ve yeniden başlatmalar ile replikalar arasında korunur.

## Siteyi özelleştirme { #customize-the-site }

1. Platform yöneticisi olarak oturum açın ve **Site ayarları** bölümünü açın.
2. Site adını ve birincil rengi ayarlayın.
3. İsteğe bağlı olarak logo, kompakt kenar çubuğu logosu, koyu tema sürümleri ve site simgesi yükleyin.
4. İsteğe bağlı olarak alt bilgideki şirket adını ve bağlantısını ayarlayın.
5. **Değişiklikleri kaydet** düğmesine tıklayın. Mevcut tarayıcı ayarları hemen uygular. Diğer tarayıcılar sonraki yeniden yüklemede yeni ayarları kullanır.

Site adı, tarayıcı başlığında ve logonun alternatif metninde kullanılır. Özel logo belirtilmediğinde giriş sayfasında özel ad gösterilir. Koyu tema logosu yoksa ilgili açık tema görseli kullanılır. Bir görseli temizlemek yerleşik görseli geri getirir; **Varsayılana sıfırla** birincil rengi geri yükler. Boş alt bilgi alanlarında GPUStack varsayılanları kullanılır.

Görseller PNG, JPEG veya WebP biçiminde ve her biri en fazla 512 KiB olmalıdır. Görseller veritabanında saklanır; harici görsel URL’leri ve SVG yüklemeleri desteklenmez. Alt bilgi bağlantıları HTTP(S) URL’lerini veya `/about` gibi aynı kökene ait yolları kabul eder.

Marka bilgileri, oturum açılmadan önce de herkese açıktır. Bu alanlara gizli bilgiler eklemeyin.

Herkese açık okuma yanıtları ETag içerir ve önbelleğin yeniden doğrulanmasını gerektirir. Eşleşen `If-None-Match` isteklerine gövdesiz `304 Not Modified` yanıtı verilir; böylece değişmeyen görseller tekrar indirilmez. Başka bir sunucu replikasındaki değişikliklerin hemen görünmesi için her okuma yine veritabanını kontrol eder.

## API { #api }

| Yöntem | Uç nokta | Erişim |
| --- | --- | --- |
| GET | `/v2/site-settings/public` | Yalnızca herkese açık marka alanları; kimlik doğrulama gerekmez |
| GET | `/v2/site-settings` | Platform yöneticisi |
| PUT | `/v2/site-settings` | Yönetim kapsamına sahip platform yöneticisi |

`PUT` yapılandırmanın tamamını değiştirir. `site_name` (1–128 karakter) ve `primary_color` (`#007BFF` gibi altı basamaklı onaltılık renk) zorunludur. İsteğe bağlı alanlar `logo`, `logo_dark`, `mini_logo`, `mini_logo_dark`, `favicon`, `footer_name` (en fazla 128 karakter) ve `footer_url` (en fazla 2048 karakter) şeklindedir. Gönderilmeyen isteğe bağlı alanlar temizlenir. Görseller base64 veri URL’leri olarak iletilir. Yanıtlar, ilk içerik değişikliğine kadar null olan `updated_at` alanını içerir. Bilinmeyen giriş alanları reddedilir.

## Dağıtım uyumluluğu { #deployment-compatibility }

Düzenlemeyi etkinleştirmek için ilgili sunucu ve UI sürümlerini birlikte dağıtın. Sunucunun Alembic yükseltmesi, v2.3.1 geçiş paketinde `site_settings` tablosunu oluşturur. İlk kayıttan önce okumalar satır oluşturmadan GPUStack varsayılanlarını döndürür. Sürüm düşürme, tabloyu ve kaydedilmiş marka bilgilerini siler.

Eski bir sunucuya bağlanan veya herkese açık ayarları beş saniyede yükleyemeyen UI, GPUStack varsayılanlarıyla başlar. Yönetici ayarlarının okunamaması düzenlemeyi devre dışı bırakır ve yeniden deneme seçeneği sunar.

Marka eklentisi içeren derlemeler eklentinin başlatma sürecini ve düzenleyicisini korur. Yerleşik site ayarları sayfası marka eklentisi bulunmayan derlemelerde kullanılabilir.
