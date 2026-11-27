---
translation_status: translated
---

# Video Playground { #video-playground }

Video Playground, GPUStack video üretim API'lerini denemenizi, parametreleri özelleştirmenizi, üretilen videoları oynatmanızı ve API entegrasyonu için kod örneklerini incelemenizi sağlar.

## Video Üretme { #generate-video }

### İstem { #prompt }

Bir istem girin ve eşzamansız video oluşturmayı başlatmak için **Gönder** düğmesine tıklayın. Sayfa yalnızca son kaydedilen görevi izler ve istemi tekrar kullanabilmeniz için giriş alanında tutar. Durum sorgusu hatasından sonra yeni bir istek gönderirseniz önceki görev çıkarım motorunda çalışmaya devam edebilir.

### Üretim İlerlemesi { #generation-progress }

Sayfa görünürken Playground geçerli görevin durumunu 5 saniyede bir sorgular. Her sorgunun zaman aşımı 5 saniyedir. Sonuç alanı görevin sırada olduğunu veya oluşturmanın sürdüğünü gösterir. Motor sıfırdan büyük bir oluşturma ilerlemesi bildirdiğinde yüzde gösterilir. Durum sorgusu başarısız olursa veya zaman aşımına uğrarsa Playground sıra ya da oluşturma durumunu gizler ve o anki sorgulamayı durdurur. Hata, sonuç alanının altında ve istem kutusunun üstünde görünür. Ardından yeni bir istek gönderebilirsiniz. Durum sorgusu HTTP 404 döndürürse bu tarayıcıdaki son kayıt da temizlenir. Diğer durum hatalarında kayıt korunur. Sayfayı yenileyerek veya sayfaya geri dönerek aynı görevi yeniden sorgulayabilirsiniz.

### Sayfa Değiştirme { #switch-pages }

Playground, bu tarayıcıda her hesap için son video kimliğini ve gönderilen parametreleri saklar. Sayfa yenilendiğinde formu geri yükler ve bu kimliğin durumunu sorgular. Sayfadan ayrılınca veya sekme gizlenince sorgulama duraklar. Geri dönünce kayıtlı görev yeniden sorgulanır.

### Geçerli Görev { #tasks }

Sonuç alanında geçerli görev görünür. Tamamlanmış bir görev varken sayfayı yenilerseniz, video indirildikten ve boyutları belirlendikten sonra oynatıcı görünür. Videonun erişilebilirliği, görevi oluşturan çıkarım örneğine ve motorun görevi ne kadar süre sakladığına bağlıdır.

### İsteği Durdurma { #stop-request }

İstem araç çubuğundaki **Oluşturmayı iptal et** düğmesine tıkladığınızda iptal işlemi hemen başlar. Video kimliği beklenirken yalnızca tarayıcının görev oluşturma isteği iptal edilir. Çıkarım motoru isteği aldıysa oluşturma devam edebilir. Kimlik alındıktan sonra Playground iptal isteğini çıkarım motoruna iletir. İptal başarılı olduğunda tarayıcıdaki kayıt temizlenir ve sonuç alanı boş duruma döner. Dosyaların temizlenmesini motor yönetir.

Rastgele istem düğmesinin yanındaki **Video kaydını temizle**, kayıtlı video kimliğini, gönderilen parametreleri ve önizlemeyi onayınızın ardından yalnızca bu tarayıcıdan kaldırır.

## Model Seçme { #select-model }

Parametreler panelindeki model açılır listesinden kullanılabilir bir model seçin. Yalnızca `video` kategorisindeki modeller listelenir. Bir model görünmüyorsa çalıştığından ve dağıtımı ile rotasının doğru kategoriyi kullandığından emin olun.

Uyumlu bir SGLang Diffusion veya vLLM-Omni dağıtımı kullanın. Video anlama modelleri `llm` kategorisini kullanır ve Chat Playground üzerinden denenir.

## Parametreler { #parameters }

| Parametre | Varsayılan | Açıklama |
| --- | --- | --- |
| `Duration (seconds)` | Modelin varsayılan değeri veya `4` | Modelin süre meta verileri varsa belirtilen aralıkta bir tam sayı girin. MiniMax H3 için aralık 4–15 saniyedir. Süre meta verileri yoksa `4`, `8` veya `12` saniyeyi seçin. Kare sayısı hizalaması nedeniyle oluşturulan videonun süresi biraz farklı olabilir. |
| `Resolution` | `832x480` veya model ön ayarı | `widthxheight` biçimindeki çıktı boyutu. Ön ayar seçildiğinde gerçek boyutlar gönderilir. |
| `Seed` | Rastgele | Sabit bir değer girmek için **Rastgele Tohum** seçimini kaldırın. |
| `Reference Image` | (boş) | Gelişmiş ayarlardaki isteğe bağlı referans görseli. PNG/JPEG yükleyin veya bir HTTP(S) görsel URL'si girin. |

Referans görselini veya URL'yi kaldırmak için temizleme düğmesini kullanın. Referans görseli desteği seçilen modele bağlıdır; `video` kategorisini seçmek, yalnızca metinden video üreten bir modele görsel girdisi desteği kazandırmaz.

Oranı kutucuklarla, ardından çözünürlüğü düğmelerle seçin. Kullanılabildiğinde 480p, 720p, 1080p ve 4k sunulur; gerçek genişlik ve yükseklik seçeneğin üzerine gelindiğinde gösterilir. Model meta verileri boyut ekleyebilir veya sınırlandırabilir. Bunlar arayüz ön ayarlarıdır; motorun her boyutu kabul edeceğini garanti etmez. 21:9 yoktur; Oran bölümünde `Custom` ile genişlik ve yükseklik kaydırıcılar veya sayısal girişlerle ayarlanabilir. Başlangıç boyutu varsa model meta verilerinden alınır; yoksa model sınırları içinde `832x480` (16:9) kullanılır.

## Model yetenekleri { #model-capabilities }

MiniMax H3 için model meta verileri, yüklenen ağırlık bölümünün desteklediği görevleri, varsayılan çözünürlüğü ve 4–15 saniyelik süre aralığını belirtir. `t2va` kullanılabiliyorsa Playground bu görevi varsayılan olarak seçer. Sunucu, hem Catalog hem de doğrudan Hugging Face dağıtımlarında model dosyalarını ve yüklenen ağırlık bölümünü tanır. Mevcut dağıtımlar yetenek meta verilerini arka planda alır.

Playground üzerinden eşzamansız bir MiniMax H3 görevi oluşturulurken arayüz, `POST /v1/videos` isteğini göndermeden önce form girdilerini (`task`, `size`, `seconds` ve referans medyası) seçilen çıkarım hedefinin özgün parametrelerine dönüştürür. SGLang `task`, `conditions` ve `target` alanlarını, vLLM-Omni ise `extra_params` ile motora özgü referans alanlarını (`image_reference`, `video_reference`, `audio_reference`, `frame_indices`) kullanır.

Playground'daki `Duration (seconds)` alanı tam sayı olarak saniye kabul eder. Motor destekliyorsa gelişmiş parametrelerde motora özgü süre parametreleriyle kesirli saniyeler belirtilebilir.

Kullanılabilir görev türleri dağıtılan modelin yetenek meta verilerine göre belirlenir. Tek görev varsa otomatik seçilir ve seçici gizlenir. Birden fazla görev varsa gelişmiş ayarlardan seçilir.

- `t2va`: referans kontrolleri gizlenir; istekler ve kod örnekleri referans alanlarını içermez.
- `fl2va`: ilk kare, son kare veya her ikisi için birer görsel yükleyin.
- `ref2va`: URL veya yükleme ile tek bir referans görsel veya video sağlayın. Birden fazla referans ya da ayrı ses koşullandırması için API kullanın.

Görev veya model değiştiğinde referanslar temizlenir. Dosya başına sınırlar: PNG/JPEG için 1 MiB, MP4/MOV için 2 MiB.

## Kodu Görüntüleme { #view-code }

İstemleri ve parametreleri denedikten sonra, API'nin aynı girdilerle nasıl çağrılacağını görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` kod örnekleri sunulur.

Örnekler asenkron bir video görevi oluşturur, durumunu sorgular ve tamamlanan videoyu indirir. API uç noktaları için [Video üretimi](../../integrations/inference-apis.md#video-generation) bölümüne bakın.
