---
translation_status: translated
---

# Video Playground { #video-playground }

Video Playground, GPUStack video üretim API'lerini denemenizi, parametreleri özelleştirmenizi, üretilen videoları oynatmanızı ve API entegrasyonu için kod örneklerini incelemenizi sağlar.

## Video Üretme { #generate-video }

### İstem { #prompt }

Bir istem girin veya rastgele istem düğmesini kullanın, ardından video üretmek için `Submit` düğmesine tıklayın.

### Üretim İlerlemesi { #generation-progress }

Görev çalışırken video alanının üzerinde bir yükleme animasyonu ve üretimin sürdüğünü belirten bir mesaj (`Generating…`) görünür. Sayfa görev durumunu her 5 saniyede bir kontrol eder. Yüzde yalnızca çıkarım motoru sıfırdan büyük bir ilerleme bildirdiğinde gösterilir; bazı motorlar ara ilerleme bildirmez.

Üretim ve indirme tamamlandığında video oynatılabilir. Üretim sırasında oynatma desteklenmez.

### Sayfa Değiştirme { #switch-pages }

Bir üretim görevi izlenirken başka bir sayfaya geçmek formu ve görevi korur. Video sayfasından ayrıldığınızda veya tarayıcı sekmesi arka plana geçtiğinde durum kontrolleri duraklatılır. Video sayfasına ön planda döndüğünüzde aynı görevin durumu hemen kontrol edilir ve sorgulama devam eder.

Bu durum yalnızca mevcut tarayıcı sayfasında tutulur. Sayfanın yenilenmesi veya kapatılması durumunda korunmaz.

### İsteği Durdurma { #stop-request }

Geçerli tarayıcı isteğini iptal etmek, durum kontrollerini durdurmak ve yükleme durumundan çıkmak için `Stop` düğmesine tıklayın. Çıkarım motorunun kabul ettiği üretim görevi çalışmaya devam edebilir: bu düğme motora görev iptal isteği göndermez. Yeniden göndermek yeni bir görev oluşturur.

## Model Seçme { #select-model }

Playground arayüzünün sağ üst köşesindeki model açılır listesinden kullanılabilir bir model seçin. Yalnızca `video` kategorisindeki modeller listelenir. Bir model görünmüyorsa çalıştığından ve dağıtımı ile rotasının doğru kategoriyi kullandığından emin olun.

Uyumlu bir SGLang Diffusion veya vLLM-Omni dağıtımı kullanın. Video anlama modelleri `llm` kategorisini kullanır ve Chat Playground üzerinden denenir.

## Parametreler { #parameters }

| Parametre | Varsayılan | Açıklama |
| --- | --- | --- |
| `Duration` | `4` saniye | İstenen video süresi. Sayfa `4`, `8` ve `12` saniye seçeneklerini sunar; kabul edilen değerler modele ve motora bağlıdır. |
| `Resolution` | `720x1280` | `widthxheight` biçimindeki çıktı boyutu. Kullanılabilir seçenekler ve başlangıç değeri model meta verilerine göre değişebilir. |
| `Reference Image` | (boş) | Gelişmiş ayarlardaki isteğe bağlı referans görseli. PNG/JPEG yükleyin veya bir HTTP(S) görsel URL'si girin. |

Referans görselini veya URL'yi kaldırmak için temizleme düğmesini kullanın. Referans görseli desteği seçilen modele bağlıdır; `video` kategorisini seçmek, yalnızca metinden video üreten bir modele görsel girdisi desteği kazandırmaz.

## Kodu Görüntüleme { #view-code }

İstemleri ve parametreleri denedikten sonra, API'nin aynı girdilerle nasıl çağrılacağını görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` kod örnekleri sunulur.

Örnekler asenkron bir video görevi oluşturur, durumunu sorgular ve tamamlanan videoyu indirir. API uç noktaları için [Video üretimi](../../integrations/inference-apis.md#video-generation) bölümüne bakın.
