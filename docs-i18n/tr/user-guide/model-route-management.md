---
translation_status: translated
---

# Model Rotaları Yönetimi { #model-route-management }

GPUStack model rotası yönetimi sağlar. Model rotalarıyla model takma adları, trafik dağıtımı, olağanüstü durum kurtarma ve hem herkese açık hem de özel modeller için ortak bir giriş noktası oluşturabilirsiniz.

## Rota Oluştur { #create-route }

1. `Routes` sayfasına gidin.
2. `Add Route`'a tıklayın.
3. Rotanın `Name` alanına modelin sunulacağı adı girin.
4. Bir veya daha fazla `Route Targets` seçin.
5. `Save` düğmesine tıklayın.

## Rota Hedeflerini Yönet { #manage-route-targets }

1. `Routes` sayfasına gidin.
2. Hedefleri yönetmek istediğiniz model rotasını açın.
3. Tutmak istemediğiniz hedef için `Operations` sütunundaki `Delete` butonuna tıklayın.
4. `Operations` sütunundaki `Edit` düğmesine tıklayın. Rota düzenleme sayfasında, model rota için hedefleri ekleyin veya kaldırın ya da hedeflere ilişkin trafik ağırlığını ayarlayın.
5. Rota düzenleme sayfasında bu rotanın yedek hedefini seçin veya seçimini kaldırın.

## Yük Dengeleme { #load-balancing }

Bir model rotasının birden fazla hedefi varsa her isteği hangi hedefin karşılayacağını GPUStack ağ geçidindeki yük dengeleme (LB) eklentisi seçer. Davranış rota yapılandırmasından türetilir: hedef ağırlıkları ve etkin politika eklentileri (Oturum Yakınlığı, En Az İşlenmekte Olan İstek, Karar Hizmeti Yönlendirmesi) birlikte gösterilen `Route By` modunu belirler.

### Yönlendirme Yöntemi { #route-by }

Rota listesindeki `Route By` sütunu, isteklerin hedeflere nasıl dağıtıldığını gösterir. GPUStack bu değeri her rota değişikliğinde hedef ağırlıklarından ve etkin politika eklentilerinden hesaplar; değer istemci tarafından girilmez.

| Şuna Göre Yönlendir | Yapılandırma şekli | Yönlendirme davranışı |
| --- | --- | --- |
| `Target Weight` | Her hedefin ağırlığı 0'dan büyüktür | Trafiği hedeflere göre ağırlığa göre bölün; her hedefin ağırlığı 0'dan büyük olmalıdır |
| `Policy` | Ağırlık yoktur ve en az bir politika eklentisi (Oturum Yakınlığı, En Az İşlenmekte Olan İstek, Karar Hizmeti Yönlendirmesi) etkindir | Hedefler etkin politika eklentileriyle seçilir |
| *(dönüşümlü)* | Hiçbir hedefin ağırlığı yoktur ve hiçbir politika etkin değildir | Hedefler sırayla seçilir (round-robin) |

Notlar:

- Ağırlıklar ve politikalar birbirini dışlar: ya her hedefin ağırlığı olmalı ya da hiçbirinin olmamalıdır. Karma yapılandırma rotayı kullanılamaz yapar; tüm hedeflerin ağırlığını 0'dan büyük veya hepsini 0 olarak ayarlayın.
- Yedek hedefler hiçbir zaman aday sayılmaz; mod belirlenirken ağırlıkları yok sayılır.
- Mod, canlı örnek durumunu değil, yapılandırmanın şeklini açıklar; tüm ağırlıklı bir rota, örnekleri kapalıyken `Target Weight` kalır.

### Politika Eklentileri { #policy-plugins }

Politika eklentileri `Policy` yönlendirmesine katılır: her eklenti her hedef için bir puan döndürür ve ağ geçidi en iyi ağırlıklı toplama sahip hedefi seçer. Eklentinin `Weight` değeri (kesirli olabilen pozitif sayı) toplama katkısını belirler. Değer ayarlanmazsa eklentinin yerleşik varsayılanı kullanılır.

#### Oturum Yakınlığı { #session-affinity }

Aynı oturumun isteklerini aynı hedefe yönlendirerek istem önbelleklerinin yeniden kullanılmasını sağlar. Oturumlar sıralı bir anahtar zinciriyle tanımlanır.

- `Session Keys (Ordered, First Match Wins)` — sıralı anahtar kaynakları zinciridir; ilk değer döndüren kaynak kullanılır. Her giriş bir istek `Header` değeri (ör. `session-id`) veya `Body Key` (ör. `prompt_cache_key`) olabilir. Oturum yakınlığı etkinse en az bir oturum anahtarı gerekir.

#### En Az İşlenmekte Olan İstek { #least-inflight-request }

Hedefleri o anda işledikleri istek sayısına göre puanlar ve en az isteği işleyen hedefi tercih eder.

İki eklenti birlikte etkinse Oturum Yakınlığı oturumun bağlı olduğu hedefi, En Az İşlenmekte Olan İstek ise boş hedefi ödüllendirir. Oturum ancak ikinci eklentinin puan farkı yakınlığın avantajını aşarsa başka hedefe geçer. Varsayılan ağırlıklarda bu durum 2–3 hedefli rotalarda kolayca oluşur, 5 veya daha fazla hedefli rotalarda ise hiç oluşmaz. Dengeyi değiştirmek için eklentilerin `Weight` değerlerini ayarlayın; hesaplamalar ve örnekler için [LB Politikalarıyla Yönlendirme](../tutorials/routing-with-lb-policies.md) bölümüne bakın.

#### Karar Hizmeti Yönlendirme { #decision-service-routing }

Hedefleri görev zorluğuna göre puanlar: her istek kısaltılarak model seçimi sorusuyla birlikte Jev uyumlu bir karar hizmetine — `Model` - `Provider` sayfasındaki harici `TypeSafe Decision Service (Jev)` sağlayıcısına veya yerel olarak dağıtılmış bir karar modeline ([Karar Modellerini Kullanma](../using-models/using-decision-models.md) bölümüne bakın) — gönderilir ve yanıtı ağırlıklı oy olarak kullanılır. Karar hatası, zaman aşımı veya 2xx dışı yanıt durumunda sessizce diğer eklentilere dönülür.

Ayarlar:

- `Decision Service` — karar hizmeti kaynağı (gerekli): harici bir TypeSafe sağlayıcısı veya yerel olarak dağıtılmış bir karar modeli (`decision` kategorisindeki çalışır durumdaki dağıtımlar). İkisi birbirini dışlar — tam olarak birini seçin.
- `Decision Model` — sorulacak karar motoru, seçilen kaynaktan türetilir: harici sağlayıcı için seçilen motor (örn. `jev-latest`, sağlayıcıda önbelleğe alınan motorlardan okunur), yerel dağıtım için dağıtımın adı. Ayarlanmamışsa veya geçersizse karar yönlendirme atlanır ve istekler diğer eklentilere geri döner.
- `Instructions` — Karar servisine gönderilen model seçimi sorusu için ekstra rehberlik.
- `Model Criteria` — model adı → yetenek açıklaması: karar hizmetinin bir model seçmek için kullandığı temel. En az bir kriter gereklidir; girişler rotanın hedeflerinden oluşturulabilir.
- `Weight` — diğer politikalarla birlikte kullanıldığında ağırlıklı toplama katkısıdır; ayarlanmazsa yerleşik varsayılan (10) kullanılır.

Küçük ve büyük bir model arasında uçtan uca bir yönlendirme örneği için [LB Politikalarıyla Yönlendirme](../tutorials/routing-with-lb-policies.md) konusuna bakın.

Yedek hedefler (`fallback_status_codes`) ayrı bir yedekleme eklentisi tarafından işlenir ve LB adayı seçimine katılmaz. Bir rota, LB adayları arasındaki trafik dağıtımını olağanüstü durum kurtarma için yedek hedeflerle birleştirebilir.

## Rota Erişimini Yetkilendir { #authorize-route-access }

1. `Routes` sayfasına gidin.
2. Yetkilendirme ayarını değiştirmek istediğiniz rotayı bulun.
3. `Operations` sütunundaki `Access Settings` düğmesine tıklayın.
4. `Access Scope`'ı gerektiği gibi değiştirin.
5. `Allowed Users` kapsamı için bu rota için yetkilendirmek istediğiniz kullanıcıları seçin ve onaylamak için `>` tuşuna tıklayın.
6. `Save` düğmesine tıklayın.

## Rotayı Düzenle { #edit-route }

1. `Routes` sayfasına gidin.
2. Düzenlemek istediğiniz rotayı bulun.
3. `Operations` sütunundaki `Edit` düğmesine tıklayın.
4. Adı, model kategorisini, açıklamayı ve rota hedeflerini gerektiği gibi değiştirin.
5. `Save` düğmesine tıklayın.

## Rotayı Sil { #delete-route }

1. `Routes` sayfasına gidin.
2. Silmek istediğiniz rotayı bulun.
3. `Operations` sütunundaki `Delete` düğmesine tıklayın.
4. Silme işlemini onaylayın.
