---
translation_status: translated
---

# Yeniden Sıralama Deneme Alanı { #rerank-playground }

Yeniden Sıralama Deneme Alanı, birden fazla metni sorguyla ilgisine göre yeniden sıralayan modelleri test etmenizi sağlar. Çeşitli giriş metinlerini deneyin, parametreleri özelleştirin ve API entegrasyonu için kod örneklerini inceleyin.

## Metin Ekleme { #add-text }

Yeniden sıralamak için belgeye birden fazla metin girişi ekleyin.

## Toplu Metin Girişi { #batch-input-text }

Çok satırlı metni satır sonlarına göre ayrı girişlere bölmek için `Batch Input Mode` seçeneğini etkinleştirin. Bu, birden fazla metin parçasını verimli biçimde işlemek için kullanışlıdır.

## Temizleme { #clear }

Belgeyi ve sorgu sonuçlarını sıfırlamak için `Clear` düğmesine tıklayın.

## Sorgu { #query }

Bir sorgu girin ve metinleri sorguyla ilgilerine göre sıralanmış liste halinde almak için `Submit` düğmesine tıklayın.

![Yeniden sıralama](../../assets/playground/reranker.png)

## Model Seçimi { #select-model }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir bir yeniden sıralama modelini seçin.

## Parametreleri Özelleştirme { #customize-parameters }

Parametreler bölümünde, getirilecek eşleşen metin sayısını belirtmek için `Top N` değerini ayarlayın.

## Kodu Görüntüleme { #view-code }

Giriş metnini ve sorguyu denedikten sonra aynı girişle API'yi nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sunulur.
