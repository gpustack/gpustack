---
translation_status: translated
---

# Ses Deneme Alanı { #audio-playground }

Ses Deneme Alanı, GPUStack'in Metinden Konuşmaya (TTS) ve Konuşmadan Metne (STT) API'lerini test etmek ve denemek için ayrılmıştır. Metni sese ve sesi metne etkileşimli biçimde dönüştürmenizi, parametreleri özelleştirmenizi ve API entegrasyonu için kod örneklerini incelemenizi sağlar.

## Metinden Konuşmaya { #text-to-speech }

TTS modellerini test etmek için "Text to Speech" sekmesine geçin.

### Metin Girişi { #text-input }

Dönüştürmek istediğiniz metni girin, ardından konuşma üretmek için `Submit` düğmesine tıklayın.

![Metinden konuşmaya](../../assets/playground/text-to-speech.png)

### Model Seçimi { #select-model }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir bir TTS modelini seçin.

### Parametreleri Özelleştirme { #customize-parameters }

Üretilen sesin sesini ve biçimini özelleştirin.

!!! tip

    Desteklenen sesler modele göre değişebilir.

### Kodu Görüntüleme { #view-code }

Giriş metnini ve parametreleri denedikten sonra aynı girişle API'yi nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sunulur.

## Konuşmadan Metne { #speech-to-text }

STT modellerini test etmek için "Speech to Text" sekmesine geçin.

### Ses Dosyası Sağlama { #provide-audio-file }

Yazıya dönüştürmek için sesi iki şekilde sağlayabilirsiniz:

1. **Ses dosyası yükleme**
2. **Çevrimiçi ses kaydetme**

!!! note

    Çevrimiçi kayıt kullanılamıyorsa bunun nedeni aşağıdakilerden biri olabilir:

    1. HTTPS veya `http://localhost` üzerinden erişirken tarayıcınızda mikrofon izni etkin olmalıdır.
    2. `http://{host IP}` üzerinden erişirken URL, tarayıcınızın güvenilir adresler listesine eklenmelidir.

          **Örnek:**
          Chrome'da `chrome://flags/` adresini açın, GPUStack URL'sini "Insecure origins treated as secure" ayarına ekleyin ve bu seçeneği etkinleştirin.

![Konuşmadan metne](../../assets/playground/audio-permission.png)

![Konuşmadan metne](../../assets/playground/speech-to-text.png)

### Model Seçimi { #select-model_1 }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir bir STT modelini seçin.

### Metni Kopyalama { #copy-text }

Modelin ürettiği döküm metnini kopyalayın.

### Parametreleri Özelleştirme { #customize-parameters_1 }

Yazıya dönüştürme doğruluğunu artırmak için ses dosyanızın dilini seçin.

### Kodu Görüntüleme { #view-code_1 }

Ses dosyalarını ve parametreleri denedikten sonra aynı girişle API'yi nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sunulur.
