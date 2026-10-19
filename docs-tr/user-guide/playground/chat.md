---
translation_status: translated
---

## Sohbet Deneme Alanı { #chat-playground }

Sohbet tamamlama API'siyle etkileşim kurun. Örnek ekran görüntüsü aşağıdadır:

![Deneme alanı ekran görüntüsü](../../assets/quick-start/quick-chat.png)

### İstemler { #prompts }

Deneme alanının sol tarafında istem mesajlarını düzenleyebilirsiniz. Mesajlarda üç rol bulunur: sistem, kullanıcı ve asistan.

- **Sistem**: Genellikle bağlamı belirleyen, davranışı tanımlayan veya modelin yanıt üretme biçimine belirli kısıtlamalar getiren önceden tanımlı bir talimattır.
- **Kullanıcı**: Kullanıcının (LLM ile etkileşime giren kişinin) sağladığı giriş veya sorgudur.
- **Asistan**: LLM'nin ürettiği yanıttır.

#### Sistem Mesajını Düzenleme { #edit-system-message }

Deneme alanının üst kısmında sistem mesajını ekleyebilir ve düzenleyebilirsiniz.

#### Kullanıcı ve Asistan Mesajlarını Düzenleme { #edit-user-and-assistant-messages }

Kullanıcı veya asistan mesajı eklemek için `Add` düğmesine tıklayın.

Kullanıcı veya asistan mesajını kaldırmak için mesajın sağ köşesindeki eksi düğmesine tıklayın.

Mesajın rolünü değiştirmek için başındaki `User` veya `Assistant` metnine tıklayın.

### Görüntü Yükleme { #upload-image }

İsteme görüntü eklemek için `Upload Image` düğmesine tıklayın.

### Ses Yükleme { #upload-audio }

İsteme ses dosyası eklemek için `Upload Audio` düğmesine tıklayın.

### İstemleri Temizleme { #clear-prompts }

Tüm istemleri temizlemek için `Clear` düğmesine tıklayın.

### Model Seçimi { #select-model }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir modelleri seçebilirsiniz. Modellerin nasıl dağıtılacağını öğrenmek için [Model Dağıtımı Yönetimi](../model-deployment-management.md) bölümüne bakın.

### Parametreleri Özelleştirme { #customize-parameters }

`Parameters` bölümünde yanıt üretme parametrelerini özelleştirebilirsiniz.

### Yanıt Üretme { #do-completion }

Yanıt üretmek için `Submit` düğmesine tıklayın veya Enter tuşuna basın.

### Kodu Görüntüleme { #view-code }

İstemleri ve parametreleri denedikten sonra aynı girişle API'yi koddan nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` örnekleri sunulur.

## Karşılaştırma Deneme Alanı { #compare-playground }

Deneme alanında birden fazla modeli karşılaştırabilirsiniz. Örnek ekran görüntüsü aşağıdadır:

![Model karşılaştırma ekran görüntüsü](../../assets/playground/compare-playground.png)

### Karşılaştırma Modu { #comparison-mode }

Karşılaştırma görünümü düğmelerinden 2, 3, 4 veya 6 model karşılaştırmasını seçebilirsiniz.

### İstemler { #prompts_1 }

İstem mesajlarını sohbet deneme alanındaki gibi düzenleyebilirsiniz.

### Görüntü Yükleme { #upload-image_1 }

İsteme görüntü eklemek için `Upload Image` düğmesine tıklayın.

### Ses Yükleme { #upload-audio_1 }

İsteme ses dosyası eklemek için `Upload Audio` düğmesine tıklayın.

### İstemleri Temizleme { #clear-prompts_1 }

Tüm istemleri temizlemek için `Clear` düğmesine tıklayın.

### Model Seçimi { #select-model_1 }

Her model panelinin sol üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir modelleri seçebilirsiniz.

### Parametreleri Özelleştirme { #customize-parameters_1 }

Her modelin ayarlar düğmesine tıklayarak yanıt üretme parametrelerini özelleştirebilirsiniz.
