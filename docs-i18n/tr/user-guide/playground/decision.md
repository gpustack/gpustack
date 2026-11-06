---
translation_status: translated
---

# Karar Playground { #decision-playground }

Karar Playground, verilen bir duruma karşı tiplendirilmiş soruları yanıtlayan karar modellerini test etmenizi sağlar. State ve questions öğelerini düzenleyin, dağıtılmış bir karar modelinde çalıştırın, tiplendirilmiş yanıtları inceleyin ve API entegrasyonu için kod örneklerini görüntüleyin.

## State { #state }

`State` düzenleyicisi, kararın verildiği durumu tutar — düz metin veya sohbet mesajları gibi bir JSON nesnesi/dizisi. Düzenleyici yazarken JSON'u doğrular; çalıştırmak için state gereklidir.

## Questions { #questions }

`Questions` düzenleyicisi, state hakkında sorulan soruları tutar; soru kimliğini soru tanımına eşleyen bir JSON nesnesidir. `+ choice`, `+ score` ve `+ noul` düğmelerini kullanarak doğrudan çalıştırılabilir şablonlar ekleyin:

- `choice` — `criteria` içindeki (seçenek adı → açıklama) seçeneklerden birini seçer.
- `score` — state'i `criteria` içindeki sıralı düzey listesine göre puanlar.
- `noul` — ikili (evet/hayır) bir sorudur.

Düzenleyici JSON'u doğrular ve şema ipuçları sunar, ancak soru seti açık kalır — motora özgü alanlar olduğu gibi iletilir.

## Run { #run }

Bir karar modeli seçin, ardından `Run` düğmesine tıklayın veya `Ctrl`/`Cmd` + `Enter` tuşlarına basarak state ve questions öğelerini `/v1/systemone` uç noktasına gönderin. Çalışan bir isteği iptal etmek için `Stop` öğesine tıklayın.

![Karar Playground](../../assets/playground/decision.png)

## Answers { #answers }

Her yanıt kendi soru kimliği altında döner ve kart olarak görüntülenir: sonuç (`Choice`, `Score` veya `noul` için evet/hayır etiketi), `Confidence` ve seçenek başına olasılıklar. Kartların üstündeki özet satırı, hangi modelin yanıtladığını ve token kullanımını gösterir. Tanınamayan yanıt biçimleri ham JSON görünümüne düşer.

## Model Seçme { #select-model }

Playground arayüzünün sağ üst köşesindeki model açılır listesine tıklayarak GPUStack'te kullanılabilir bir karar modeli seçin. Yalnızca `decision` kategorisindeki modeller listelenir — bir model eksikse, çalıştığından ve doğru şekilde kategorilendirildiğinden emin olun; gerekirse modelin ayarlarından kategoriyi düzeltin.

## Kod Görüntüleme { #view-code }

State ve questions öğelerini düzenledikten sonra, aynı girdiyle API'nin nasıl çağrılacağını görmek için `View Code` düğmesine tıklayın. `curl`, `Python` ve `Node.js` için kod örnekleri sağlanır.
