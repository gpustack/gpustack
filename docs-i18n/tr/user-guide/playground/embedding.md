---
translation_status: translated
---

# Vektör Gömme Deneme Alanı { #embedding-playground }

Vektör Gömme Deneme Alanı, modelin metni vektör temsillerine dönüştürme yeteneğini test etmenizi sağlar. Birden fazla giriş metnini deneyebilir, vektör temsillerini görselleştirebilir ve API entegrasyonu için kod örneklerini inceleyebilirsiniz.

## Metin Ekleme { #add-text }

En az iki metin girişi ekleyin ve vektör temsilleri üretmek için `Submit` düğmesine tıklayın.

## Toplu Metin Girişi { #batch-input-text }

Çok satırlı metni satır sonlarına göre ayrı girişlere otomatik olarak bölmek için `Batch Input Mode` seçeneğini etkinleştirin. Bu, birden fazla metin parçasını tek işlemde işlemek için kullanışlıdır.

## Görselleştirme { #visualization }

Vektör temsillerinin boyutunu PCA (Temel Bileşen Analizi) ile azaltarak sonuçları 2B grafikte görselleştirin. Sonuçlar iki biçimde görüntülenebilir:

1. **Grafik** - PCA sonuçlarını görsel olarak gösterir.
2. **JSON** - Ham vektör temsillerini JSON biçiminde gösterir.

Grafikte noktalar arasındaki uzaklık, ilgili metinlerin benzerliğini temsil eder. Birbirine daha yakın noktalar daha yüksek benzerlik gösterir.

![Vektör gömme](../../assets/playground/embedding.png)

## Temizleme { #clear }

Metin girişlerini sıfırlamak ve çıktıyı temizlemek için `Clear` düğmesine tıklayın.

## Model Seçimi { #select-model }

Deneme alanının sağ üst köşesindeki model açılır menüsünden GPUStack'teki kullanılabilir modelleri seçebilirsiniz.

## Kodu Görüntüleme { #view-code }

Giriş metinlerini denedikten sonra aynı girişle API'yi nasıl çağıracağınızı görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sunulur.
