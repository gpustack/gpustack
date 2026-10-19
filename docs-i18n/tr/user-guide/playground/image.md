---
translation_status: translated
---

# Görüntü Deneme Alanı { #image-playground }

Görüntü Deneme Alanı, GPUStack'in görüntü üretme API'lerini test etmek ve denemek için ayrılmıştır. Farklı modellerin yeteneklerini etkileşimli biçimde keşfetmenizi, parametreleri özelleştirmenizi ve API entegrasyonu için kod örneklerini incelemenizi sağlar.

## Görüntü Üretme { #generate-image }

### İstem { #prompt }

Bir istem girin veya rastgele oluşturun, ardından görüntü üretmek için `Submit` düğmesine tıklayın.

![Görüntü](../../assets/playground/create-image.png)

## Görüntü Düzenleme { #edit-image }

Bir görüntü yükleyin ve değiştirmek istediğiniz alanları boyayarak işaretleyin. Ardından bir istem girin ve `Submit` düğmesine tıklayın. Hiçbir alan boyanmazsa görüntünün tamamı değiştirilir.

![Görüntü](../../assets/playground/image-edit.png)

### Maskeyi Kaydetme { #save-mask }

Boyanan alanları ayrı bir görüntü olarak kaydetmek için `Save Mask` düğmesine tıklayın.

### Görüntüyü İndirme { #download-image }

Düzenlenmiş görüntüyü kaydetmek için `Download Image` düğmesine tıklayın.

## Model Seçimi { #select-model }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir modelleri seçebilirsiniz.

## Parametreler { #parameters }

| Parametre | Varsayılan | Açıklama |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Size` | `512x512` | Üretilen görüntünün 'genişlikxyükseklik' biçimindeki boyutu. |
| `Seed` | (boş) | Rastgele sayı üretecinin başlangıç değeri. |

## Kodu Görüntüleme { #view-code }

İstemleri ve parametreleri denedikten sonra aynı girişlerle API'yi nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sunulur.
