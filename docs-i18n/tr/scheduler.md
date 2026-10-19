---
translation_status: translated
---

# Zamanlayıcı { #scheduler }

## Özet { #summary }

Zamanlayıcı iki ilgili görevden sorumludur:

- **Ölçek büyütme:** yeni model örnekleri için en iyi çalışan/GPU yerleşimini seçin.
- **Ölçeği küçült:** Mevcut model örneklerini sıralayın ve kopyalar azaldığında en az tercih edilenleri kaldırın.

Her iki yönde de zamanlayıcı iki basit aşamada çalışır:

1. Filtre: modeli gerçekten çalıştırabilecek adayları bulun. Bunu aday listesini oluşturmak gibi düşünebilirsiniz.
2. Puan: Her adaya bir puan verin, ardından mevcut hedef için en iyi olanı seçin.

## Ölçek Büyütme Planlaması { #scale-up-scheduling }

### 1. Temel Çalışan Filtreleme { #1-basic-worker-filtering }

Zamanlayıcı ilk olarak çalışan listesinde temel bir filtreleme turu gerçekleştirir. Bu adım temel olarak kaynak dışı kısıtlamalarla ilgilidir: küme, etiketler, arka uç uyumluluğu, seçilen GPU'lar ve yerel yol kullanılabilirliği.

Filtre zinciri şu sırayla çalışır:

1. **Küme Filtresi** Yalnızca çalışanları modelin hedef kümesinde tutar.
2. **GPU Eşleştirme Filtresi** Model açıkça GPU'ları seçtiğinde çalışan kümesini daraltır.
3. **Etiket Eşleştirme Filtresi** Yalnızca etiketleri modelin etiket seçicilerini karşılayan çalışanları tutar.
4. **Durum Filtresi** Yalnızca çalışanları `READY` durumunda tutar.
5. **Arka Uç Çerçeve Filtresi** Hızlandırıcıları veya çalışma zamanı yetenekleri seçilen arka uçla eşleşmeyen çalışanları kaldırır.
6. **Yerel Yol Filtresi** GPU'lar açıkça seçildiğinde `LOCAL_PATH` modellerine uygulanır. Yapılandırılmış model yolunun mevcut olmadığı çalışanları kaldırır.

Yalnızca bu temel filtreleme adımını geçen çalışanlar kaynak tabanlı filtrelemeye geçer.

### 2. Kaynak Bazlı Aday Filtreleme { #2-resource-based-candidate-filtering }

Bu adım aynı zamanda bir filtredir. Meta verilere veya çalışan durumuna göre filtrelemek yerine kaynaklara göre filtreler: Bu çalışan veya yerleşim, modeli çalıştırmak için gerçekten yeterli RAM/VRAM sağlayabilir mi?

Kaynak gereksinimleri model türüne bağlı olarak farklı şekilde belirlenir:

- **GGUF modelleri:** kaynak gereksinimleri [GGUF ayrıştırıcısı](https://github.com/gpustack/gguf-parser-go) ile tahmin edilir.
- **Diğer model türleri:** kaynak gereksinimleri, vLLM, SGLang, MindIE veya VoxBox gibi ilgili arka uç tarafından tahmin edilir.

Arka uç yetenekleri farklıdır, dolayısıyla mevcut geri dönüş yolları da farklıdır:

- **vLLM, SGLang, MindIE:** esas olarak GPU tabanlı yerleşimleri kullanır ve burada yalnızca CPU veya kısmi boşaltma geri dönüş yollarını kullanmayın.
- **GGUF, özel arka uçlar, VoxBox:** Modelin kaynak gereksinimlerini tahmin etmek için [GGUF ayrıştırıcısını](https://github.com/gpustack/gguf-parser-go) kullanır. GPU boşaltmayı, kısmi boşaltmayı veya CPU yürütmeyi destekler.
- **Özel arka uçlar, VoxBox:** GPU aktarımını veya CPU yürütmeyi destekler.

Adaylar daha sonra sırayla değerlendirilir ve bir strateji çalıştırılabilir adaylar getirdiğinde süreç durdurulur. Genel olarak zamanlayıcı şunları dener:

1. **Tek işçi, tek GPU** Bir işçi üzerindeki bir GPU, model gereksinimlerini karşılamak için yeterlidir.
2. **Tek işçi, birden fazla GPU** Bir GPU'nun yeterli olmadığı durumlarda aynı işçi üzerinde birden fazla GPU birlikte kullanılır.
3. **Çalışanlar arasında dağıtılmış çıkarım** Arka uç dağıtılmış yürütmeyi desteklediğinde, birden fazla çalışandaki GPU'lar kullanılabilir.
4. **Kısmi boşaltma veya CPU yürütme** Yalnızca söz konusu model türü ve arka uç bu geri dönüş modlarını desteklediğinde kullanılır.

Burada birkaç ayrıntı önemlidir:

- Kaynaklara uygun filtreleme, adayları ortaya çıkaran ilk stratejide durur.
- Dağıtılan adaylar, asıl çalışanın yanı sıra alt çalışanları da kapsayabilir.
- Açık GPU seçimleri, hem temel çalışan filtrelemesi hem de kaynağa uygun filtreleme adımı sırasında dikkate alınır.

### 3. Aday Puanlaması { #3-candidate-scoring }

Çalıştırılabilir adaylar bulunduğunda zamanlayıcı, bunları bir puanlayıcı zinciriyle değerlendirir ve toplam puanı en yüksek olan adayı seçer.

Mevcut ölçek büyütme puanlaması:

1. **Yerleştirme Skoru**
2. **Model Dosyası Yerellik Puanlayıcısı**

Toplam aday puanı, tüm etkin puanlayıcıların toplamıdır.

#### Yerleştirme Puanlayıcı { #placement-scorer }

Yerleştirme puanlayıcı her zaman ölçek büyütme için etkindir.

- **Binpack**
  
  Bu strateji, kaynak kullanımını optimize etmek için mümkün olduğunca çok sayıda model örneğini en az sayıda "kutuya" (örneğin, İşçiler/GPU'lar) "paketlemeyi" amaçlar. Amaç, kaynak verimliliğini en üst düzeye çıkarırken kullanılan silo sayısını en aza indirmek, her bir silonun kapasitesini aşmadan mümkün olduğunca verimli bir şekilde doldurulmasını sağlamaktır. Model örnekleri, her bölmede kalan kapasiteyi en aza indirmek için en az alana sahip bölmeye yerleştirilir.
- **Yayılma**
  
  Bu strateji, birden fazla model örneğini farklı çalışanlar arasında mümkün olduğunca eşit bir şekilde dağıtmayı, sistemin hata toleransını ve yük dengelemeyi iyileştirmeyi amaçlamaktadır.

Ek davranış:

- GPU yerleşimleri için VRAM basıncı, RAM basıncından daha ağırdır.
- Yalnızca CPU yerleşimlerinde yalnızca RAM kullanımı dikkate alınır.

#### Model Dosyası Yerellik Puanlayıcısı { #model-file-locality-scorer }

Model dosyası yerellik puanlayıcısı, gerekli model dosyaları zaten `READY` durumunda olan worker'lara yerleştirmeyi tercih etmek için eklenmiştir.

Ne işe yarar:

- Ana model dosyaları zaten hazır olan çalışanları arar.
- Spekülatif kod çözme yapılandırılmışsa taslak model için hazır dosyaları da arar.
- Her adayı, halihazırda bu dosyalara sahip olan katılımcı çalışanların oranına göre puanlar.
- Ana model konumuna taslak model konumuna göre daha fazla ağırlık verir.

### 4. Son Seçim { #4-final-selection }

Puanlamanın ardından zamanlayıcı en yüksek toplam puana sahip adayı seçer ve model örneğini o yerleşime atar.

## Ölçek Aşağı Planlama { #scale-down-scheduling }

İstenilen kopya sayısı mevcut model örneklerinin sayısından düşük olduğunda, denetleyici mevcut örnekleri sıralar ve önce en düşük sıradakileri siler.

Ölçek küçültme, mevcut model örnekleri üzerinde ayrı bir puanlayıcı zinciri kullanır:

1. **Durum Puanlayıcı**
2. **Yük Boşaltma Katman Puanlayıcı**
3. **Yerleştirme Skoru**

Ortaya çıkan örnekler artan sırada puana göre sıralanır ve en düşük puana sahip örnekler ilk önce kaldırılır.

### Durum Puanlayıcı { #status-scorer }

Durum puanlayıcı sağlıklı kopyaları tercih eder. Bu, sağlıksız veya henüz hazır olmayan kopyaları kaldırılacak ilk adaylar haline getirir.

### GPU'ya Aktarılan Katman Puanlayıcısı { #offload-layer-scorer }

`total_layers` ve `offload_layers` bildiren GGUF modellerinde bu puanlayıcı, daha fazla katmanı GPU'ya aktarılmış örnekleri tercih eder:

- Tüm katmanların GPU'ya aktarılması en yüksek puanı alır.
- Kısmi boşaltma orantılı bir puan alır.
- Yük boşaltma katmanı meta verileri olmayan örnekler bu puanlayıcıdan `0` alır.

### Ölçek Küçültme Modunda Yerleştirme Puanlayıcısı { #placement-scorer-in-scale-down-mode }

Aynı yerleştirme puanlayıcı, ölçek küçültme sırasında yeniden kullanılır ancak mevcut yerleşimleri, yerleştirme perspektifi yerine kaldırma perspektifinden değerlendirir. Yerleştirme hala modelin `binpack` veya `spread` politikasını yansıtıyor ancak puan, tutma tercihi olarak yorumlanıyor, dolayısıyla daha düşük yerleştirme puanlarının ilk önce kaldırılması daha olası.
