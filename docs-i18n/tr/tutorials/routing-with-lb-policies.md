---
translation_status: translated
---

# LB Politikalarıyla Yönlendirme { #routing-with-lb-policies }

Bu eğitimde GPUStack'in rota düzeyinde yük dengeleme politikaları pratikte anlatılmaktadır: yükü dağıtırken oturumları sabit tutmak ve karar hizmetinin istek başına küçük ve büyük bir model arasında seçim yapmasına izin vermek. Yapılandırma referansı için bkz. [Model Rota Yönetimi](../user-guide/model-route-management.md#load-balancing).

## Önkoşullar { #prerequisites }

- Çalışan bir GPUStack kurulumu. Bu politikaları uygulayan AI ağ geçidi, varsayılan olarak GPUStack ile dağıtılır, dolayısıyla standart kurulumun ekstra kuruluma ihtiyacı yoktur.
- Rota hedefi olarak hizmet veren iki konuşlandırılmış model, ör. `qwen3-0.6b` ve `qwen3-32b` (aşağıdaki örneklerde bunlar kullanılmaktadır).
- Karar hizmeti kısmı için: bir karar hizmeti — `Model` - `Provider` sayfasında `TypeSafe Decision Service (Jev)` sağlayıcısı olarak kayıtlı bir Jev karar hizmeti (bkz. [Model Sağlayıcı Yönetimi](../user-guide/model-provider-management.md)) veya yerel olarak dağıtılmış bir karar modeli (bkz. [Karar Modellerini Kullanma](../using-models/using-decision-models.md)).

## Adım 1: Rotayı Oluşturun { #step-1-create-the-route }

1. `Routes` sayfasına gidin.
2. `Add Route`'e tıklayın, örneğin adını verin. `qwen3`.
3. Tüm ağırlıkları ayarlamadan bırakarak `qwen3-0.6b` ve `qwen3-32b`'i `Route Targets` olarak seçin.
4. `Save`'a tıklayın.

Ağırlık ve politika olmadan, rotanın `Route By`'si dönüşümlüdür — iki model arasında düz dönüşümlü.

## Adım 2: Session Affinity ve Least Inflight Request Etkileşimi { #step-2-understand-how-session-affinity-and-least-inflight-request-interact }

Politika eklentileri etkinleştirildiğinde her eklenti tüm aday hedefleri puanlar. Son seçim mekanizması en yüksek ağırlıklı toplam puana sahip hedefi seçer:

```
total(candidate) = Σ over plugins ( plugin weight × normalized score )
```

Session Affinity en yüksek puanı oturumun mevcut hedefine, Least Inflight Request ise en az yüklü hedefe verir. Oturum ancak Least Inflight Request puan farkı Session Affinity puan farkını aştığında mevcut hedeften ayrılır.

Her iki eklentinin ağırlığı varsayılan değerinde (1) olduğunda aşağıdaki tablo, oturumun mevcut hedefinde `x` işlenmekte olan istek biriktiği ve diğer tüm hedeflerin boşta olduğu durumun sonucunu gösterir. Bu, hedef değiştirmeye en elverişli durumdur:

| Hedefler (N) | x=0 | x=1 | x=2 | x=3 | x=4 | Hedef değişir mi? |
| ----------- | --- | --- | --- | --- | --- | ----------- |
| 2 | korunur | eşitlik ⚠️ | **DEĞİŞİR** | **DEĞİŞİR** | **DEĞİŞİR** | evet |
| 3 | korunur | korunur | eşitlik ⚠️ | **DEĞİŞİR** | **DEĞİŞİR** | evet |
| 4 | korunur | korunur | korunur | korunur | korunur | yalnızca x ≈ 5.3 olduğunda |
| ≥5 | korunur | korunur | korunur | korunur | korunur | **hayır** |

- **DEĞİŞİR**: oturum bir sonraki aday hedefe geçer.
- **eşitlik ⚠️**: puan farkları tam olarak eşittir. Son seçim rastgele yapılır; aynı istek başka hedefe geçebilir veya mevcut hedefte kalabilir.
- **N ≥ 5**: Least Inflight Request puan farkı `1/(N−1) ≤ 0.25` ile sınırlıdır; bu, Session Affinity farkından küçüktür. Yalnızca yük dengesizliği oturumun hedef bağlılığını bozamaz.

Daha büyük rotalarda yükün önemli olmasını sağlamak için ağırlıkları yeniden dengeleyin — ölçekli ağırlık oranları altındaki çevirme eşikleri için [Ağırlıklar Nasıl Çalışır](#how-weights-work) konusuna bakın.

## Adım 3: Session Affinity ve Least Inflight Request Özelliklerini Etkinleştirme { #step-3-enable-session-affinity-and-least-inflight-request }

1. Rotayı 1. Adımdan düzenleyin.
2. `Session Affinity`'i etkinleştir — varsayılan oturum anahtar zincirini kabul edin (`session-id` başlığı, `x-client-request-id` başlığı, ardından `prompt_cache_key` gövde alanı) veya kendinizinkini belirtin.
3. `Least Inflight Request`'i varsayılan ağırlığıyla etkinleştirin.
4. Rotayı kaydedin ve `Route By`'nin `Policy` olduğunu doğrulayın.

Oturumlar artık yapışkan: aynı oturum anahtarını taşıyan istekler aynı hedefte kalıyor — istem önbellekleri sıcak kalıyor — ağır yüklü bir sahip, yukarıdaki tabloya göre oturumları sonunda boşta kalan hedefe aktarıyor.

Her yolda gerçekte hangi anahtar kaynakların geçerli olduğuna dikkat edin: başlık anahtarları (örn. `session-id`) her yerde çalışır, ancak `prompt_cache_key` gibi gövde anahtarları yalnızca `/responses` ve `/messages` istekleri üzerinde etkili olur — standart oturum gövde alanı olmayan `/chat/completions`'de yalnızca zincirin başlık anahtarları dikkate alınır. Sohbet trafiği için başlık anahtarını tercih edin.

## Adım 4: Karar Hizmeti Yönlendirmesi Ekleme { #step-4-add-decision-service-routing }

Şimdi rotayı modele uygun hale getirin: Bir oturumu her zaman hangi modelde başlamış olursa olsun sabitlemek yerine, karar motoruna her isteğin hangi modeli hak ettiğini sorun.

1. Rotayı düzenleyin ve `Decision Service Routing`'ı etkinleştirin:
    - `Decision Service` — kayıtlı sağlayıcıyı ve motorlarından birini (örn. `jev-latest`) seçin ya da bunun yerine yerel olarak dağıtılmış bir karar modeli seçin.
    - `Instructions` — "İsteği iyi karşılayabilecek en küçük modeli seçin. Basit, kısa veya olgusal istekler hızlı modele; karmaşık akıl yürütme, matematik veya kod istekleri büyük modele gider."
    - `Model Criteria` — hedef başına bir giriş (veya `Generate from targets`'e tıklayın ve hassaslaştırın):
        - `qwen3-0.6b` — "Çok hızlı, düşük maliyetli 0,6B modeli. Selamlaşmayı, basit gerçek soruları, biçimlendirmeyi ve kısa sohbetleri yönetir. Çok adımlı akıl yürütme, matematik ve kod konularında başarısız olur."
        - `qwen3-32b` — "Yüksek kapasiteli 32B akıl yürütme modeli. Karmaşık akıl yürütme, matematik, kod oluşturma ve uzun bağlam analizini yönetir. İstek başına daha yavaş ve daha pahalı."

2. Kaydedin ve ardından `qwen3` rota adına karşı iki kimliği doğrulanmış istek gönderin (`$GPUSTACK_API_KEY`'i `API Keys` sayfasında oluşturulan bir API anahtarıyla değiştirin):

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Write a one-line greeting card."}]}'
```

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, then implement the proof sketch in Python."}]}'
```

İlk istek `qwen3-0.6b`'e, ikincisi ise `qwen3-32b`'e yönlendirilir.

## Ağırlıklar Nasıl Çalışır? { #how-weights-work }

Ağırlıklar iki farklı yerde farklı anlamlarla görünür:

### Hedef Ağırlık (trafik payı) { #target-weight-traffic-share }

`Target Weight` modunda, bir hedefin ağırlığı, kaç örnek çalıştırdığına bakılmaksızın hedefin trafikteki **toplam** payıdır. GPUStack, bölünmenin her zaman yapılandırılmış hedef ağırlıklarla eşleşmesi için örnek başına ağırlıkları ölçeklendirir:

- Hedef A (ağırlık 100, 1 örnek) ve Hedef B (ağırlık 100, 2 örnek) trafiği iki hedef arasında 50/50 böler — örnek başına 1/3 değil.
- Bir hedefin ölçeğini 1'den 2'ye çıkarmak payını değiştirmez; her örnek bunun yarısını alır.

`qwen3-0.6b` ve `qwen3-32b` ile örnek: `qwen3-0.6b`'de ağırlık 100 ve `qwen3-32b`'de ağırlık 300, kopyalarının sayısı ne olursa olsun, isteklerin %25'ini küçük modele ve %75'ini büyük modele gönderir.

### Politika Ağırlığı (toplamdaki etki) { #policy-weight-influence-in-the-sum }

Her politika eklentisinin `Weight` değeri, `Policy` modundaki ağırlıklı toplama katkısını belirler. Pozitif bir sayıdır; kesirli değerler de anlamlıdır. `Weight` ayarlanmazsa eklentinin yerleşik varsayılanı kullanılır: Session Affinity ve Least Inflight Request için 1, Decision Service Routing için 10.

Oturumun mevcut hedefinde `x` işlenmekte olan istek varken ve diğer tüm hedefler boşta iken Session Affinity × Least Inflight Request dengesi şöyle hesaplanır:

- Session Affinity puan farkı: `affDiff = 0.5 / (2·(1 − 2⁻ᴺ))`
- Least Inflight Request puan farkı: `loadDiff = x / ((1+x)·(N−1) + 1)`
- oturum, `w_leastInflight × loadDiff > w_affinity × affDiff` olduğunda hedef değiştirir

Ağırlık oranı hedef değiştirme eşiğini değiştirir. Aşağıdaki tablo, her ağırlık oranı (Least Inflight Request : Session Affinity) ve hedef sayısı N için oturumun hedef değiştirmesini sağlayan en düşük `x` etkin istek sayısını gösterir:

| N | 1:1 (varsayılan) | 1.5:1 | 2:1 |
| - | ------------- | ----- | --- |
| 2 | 2 | 1 | 1 |
| 3 | 3 | 1 | 1 |
| 4 | 6 | 2 | 1 |
| 5 | asla | 3 | 2 |
| 6 | asla | 7 | 3 |
| 7 | asla | asla | 4 |
| 8 | asla | asla | 9 |

Tablonun okunması:

- Varsayılan 1:1 oranında beş veya daha fazla hedefi olan rotalarda oturum asla hedef değiştirmez: Least Inflight Request farkının üst sınırı Session Affinity farkının altında kalır.
- Session Affinity `Weight` değerini yarıya indirmek (2:1 oranı), 8 hedefe kadar olan rotalarda `x` = 2–4 düzeyinden itibaren yükün etkili olmasını sağlar. Least Inflight Request `Weight` değerini ikiye katlamak da aynı etkiyi verir; yalnızca oran önemlidir.
- Kesin eşitlikler (örn. N=2/x=1'de 1:1) bitirici tarafından rastgele bozulur.

## Politikalar İşi Nasıl Bölüyor? { #how-the-policies-divide-the-work }

Etkinleştirilen tüm politikalar aynı ağırlıklı toplama katkıda bulunur ve her biri bir iş yapar:

- **Karar Hizmeti Yönlendirme** *hangi modelin* isteğe uygun olduğuna karar verir (varsayılan `Weight` 10, bunun genellikle baskın ses olduğunu yansıtır);
- **Session Affinity**, istem önbellek konumunu koruyarak oturumu mevcut modelinde tutar;
- **Least Inflight Request**, istekleri seçilen modelin örnekleri arasında dağıtır.

Model birden fazla örnek olarak dağıtıldığında Decision Service Routing seçimi o modelle sınırlar; Least Inflight Request ise bu modelin örneğini seçer.
