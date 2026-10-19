---
translation_status: translated
---

# GPU Hizmeti Örnek Türleri { #gpu-service-instance-types }

GPU Hizmet Örnek Türleri, [GPU Hizmet Örneklerinin](gpuservice-instances.md) oluşturulduğu hesaplama şekillerini (yalnızca CPU veya hızlandırıcı destekli) tanımlar.

[GPUStack Operatörü](https://github.com/gpustack/gpustack-operator) birden fazla üreticinin hızlandırıcılarını destekler — AMD, Ascend, Cambricon, Hygon, Iluvatar, MetaX, Moore Threads, NVIDIA ve T-Head. Her üretici için bir hızlandırıcı **münhasıran** (tüm cihaz), **paylaşılan** veya **mantıksal olarak dilimlenmiş** (bağımsız VRAM ve hesaplama bütçeleriyle yazılım bölünmesi) tahsis edilebilir. NVIDIA, T-Head ve Hygon ayrıca cihazın donanıma göre bölündüğü **fiziksel bölümlemeyi** (MIG) destekler. Üreticiye özel ayrıntılar için operatörün [hızlandırıcı destek matrisine](https://github.com/gpustack/gpustack-operator#accelerator-support) bakın.

## Örnek Türlerine Göz Atma { #browsing-instance-types }

`GPU Service` > `Instance Types`'i açın. Sayfa, görebileceğiniz her GPU Hizmeti kümesinin örnek türlerini, sahip olan kümenin her satırda adlandırıldığı şekilde listeler:

![Ekran görüntüsü: kümelerdeki bulut sunucusu türlerini listeleyen Bulut Sunucusu Türleri sayfası](../assets/gpuservice/instance-types/list.png)

- **Ad / Flavour**: Tür adı (veya görünen adı) ve ait olduğu tat — `NVIDIA-H100-80GB-HBM3` veya `CPU-only` gibi bir cihaz modeli.
- **Birim CPU / Birim RAM / Depolama**: Bu türdeki birim başına verilen ana bilgisayar kaynakları. Hızlandırıcı bir lezzet için, bir ünite bütün bir cihazdır; mantıksal dilimler ve fiziksel bölümler, oluşturma sırasında bunlardan boyutlandırılır.
- **Platform**: İşletim sistemi ve CPU mimarisi, uyumlu bir görüntü seçerken faydalıdır.
- **Cluster**: Türün ait olduğu küme.
- **Durum**: `Active` türler *GPU Örneği Ekle* formunda seçilebilir; `Inactive` olanlar yapamaz.

Listeyi tek bir kümeye daraltmak için araç çubuğundaki `Filter by cluster` seçiciyi ve türleri görünen ad veya ada göre eşleştirmek için arama kutusunu kullanın; `Name` sütununun gösterebileceği değerlerden herhangi biri. Filtreleme, sıralama ve sayfalandırmanın tümü sunucu tarafında uygulanır ve `Name` sütun sıraları görüntülediği etikete göre sıralanır. Satır eylemleri (devre dışı bırakma, etkinleştirme, silme) her zaman satırın Küme sütununda adı geçen küme üzerinde işlem yapar.

Liste, her kümeden canlı okuma yerine kontrol düzlemi kaydıdır: bir kümeye ulaşılamadığında satırları en son gözlemlendiği gibi okunur. Aşağıdaki canlı kapasite görünümü, mevcut kalan kapasiteye ilişkin otorite olmayı sürdürüyor.

Ayrıca her türün canlı kapasitesini `kubectl` ile okuyabilirsiniz. Dört hızlandırıcı görünümü **EX**clusive, **SH**ared, **SL**iced (mantıksal) ve **PT** (fiziksel olarak bölümlenmiş)'tir; hızlandırıcı görünümleri ve `CPU` sütunu aynı şekilde `onceMaxRequest/remaining`'ı okuyor:

```bash
kubectl get instancetypes
```

```
NAME                                          ENTRANCE                          UNIT(CPU/RAM)/STORAGE   ACCELERATOR(EX/SH/SL/PT)   CPU     PHASE
gpustack--generic-linux-amd64                 gpustack-fnv64-3b93966fd73eb9ec   1/2Gi/100Gi             0/0 0/0 0/0 0/0            16/40   Active
gpustack--nvidia-h100-80gb-hbm3-linux-amd64   gpustack-fnv64-e4768a65ca0ce96b   12/192Gi/100Gi          1/1 10/10 100/100 1/7      0/0     Active
gpustack--nvidia-l40s-linux-amd64             gpustack-fnv64-a730f1dca9e26fca   12/128Gi/100Gi          1/1 10/10 100/100 0/0      0/0     Active
```

Yeni eklenen kümenin bu okumasında, H100 türü özel olarak bir tam cihaz (`EX 1/1`), on paylaşılan yuva (`SH 10/10`), tam bir mantıksal dilim bütçesi (`SL 100/100`) ve yedi MIG bölme yuvası (`PT 1/7`, MIG etkinleştirilmiş ikinci bir H100 düğümünden) sunar. L40S tipi, tek cihazını bölümleme hariç aynı şekilde sunar (`PT 0/0` — L40S'de MIG yoktur) ve genel yalnızca CPU tipi hiçbir hızlandırıcı kapasitesi taşımaz.

## Türetilmiş Örnek Türleri { #derived-instance-types }

Bir GPU Hizmet kümesinde `Derive Instance Types from Nodes` **Etkin** (varsayılan) olduğunda, operatör her düğümün aygıtlarını keşfeder ve eşleşen örnek türlerini otomatik olarak yazar (her hızlandırıcı modeli için bir tane artı yalnızca CPU türü). Yukarıdaki liste böyle türetilmiş bir kümedir.

Türetilmiş bir türün birim CPU/RAM'i, [Örnek Türü Birim Kaynakları referansında](https://docs.gpustack.ai/gpustack-operator/main/docs/reference/instance-type-unit-resources/) belgelenen, operatör tarafından tutulan ürün başına ön ayarlardan gelir. Depolama her zaman 100 GiB'dir ve türetilmiş yalnızca CPU türü her zaman 1 CPU / 2 GiB RAM'dir.

Türetilmiş türler operatöre aittir:

- **Kullanıcı arayüzünden silinemezler** — `Delete` işlemi onlar için devre dışıdır:

![Ekran görüntüsü: Türetilmiş bir örnek türü için Sil işlemi devre dışı bırakıldı](../assets/gpuservice/instance-types/derived-delete-disabled.png)

- `kubectl` ile birini silmek de kalıcı olmaz: operatör onu bir sonraki mutabakatında yeniden yazar. (Türetilmiş bir türü bu şekilde yeniden oluşturmak, aslında bir operatör yükseltmesinden sonra onu yeniden boyutlandırmanın desteklenen yoludur, çünkü birim kaynaklar oluşturma sırasında bir kez damgalanır.)

Türetilmiş bir türü kalıcı olarak kaldırmak için, önce kümede `Derive Instance Types from Nodes`'i **Disabled** olarak ayarlayın (kümenin [Gelişmiş Seçenekler](gpuservice-instances.md#advanced-options)'ine bakın); operatör daha sonra tür yazmayı durdurur ve mevcut türetilmiş olanlar silinebilir.

## Örnek Türü Ekleme { #adding-an-instance-type }

Türetilenler uymadığında özel bir örnek türü ekleyin (örneğin, yalnızca CPU'ya özel daha büyük bir şekil). `Add Instance Type`'a tıklayın ve formu doldurun:

![Ekran görüntüsü: Örnek Türü Ekle formu](../assets/gpuservice/instance-types/add-form.png)

- **Cluster**: Türün oluşturulduğu küme.
- **Name**: Tür adı. Temel kaynak adının bir parçası haline geldiğinden Küçük harf ve Kubernetes açısından güvenlidir. Küme içinde benzersiz olmalıdır: Türetilmiş bir tür de dahil olmak üzere mevcut bir türün adının yeniden kullanılması, üzerine yazmak yerine reddedilir.
- **Görünen Ad**: Kullanıcı arayüzünde gösterilen isteğe bağlı kolay bir ad.
- **Lezzet**: `CPU-only` veya seçilen kümenin taşıdığı hızlandırıcı lezzet. Seçimler Küme alanını takip eder.
- **OS / Arch**: Türün hedeflediği platform.
- **Birim CPU / Birim RAM / Depolama**: Birim başına verilen ana bilgisayar kaynakları.

`Save`'a tıklayın. Yeni tür listede ve *GPU Örneği Ekle* formunda görünür:

![Ekran görüntüsü: listedeki özel CPU-Large örnek türü](../assets/gpuservice/instance-types/added.png)

!!! note

    Özel bir örnek türü hiçbir zaman operatör tarafından değiştirilmez; türetme yalnızca türler oluşturur ve henüz oluşturmadığı bir türe asla dokunmaz.

## Bir Örnek Türünü Devre Dışı Bırakma ve Etkinleştirme { #deactivating-and-activating-an-instance-type }

Bir türü silmeden yeni örneklerin oluşturulmasını engellemek için türü devre dışı bırakın. Türün satırındaki devre dışı bırakma eylemine tıklayın ve onaylayın:

![Ekran görüntüsü: örnek türünün devre dışı bırakılmasının onaylanması](../assets/gpuservice/instance-types/deactivate-confirm.png)

Tür `Inactive` olur ve artık *GPU Örneği Ekle* formunda seçilemez:

![Ekran görüntüsü: Etkin olmayan bir örnek türü](../assets/gpuservice/instance-types/inactive.png)

Aynı eyleme tekrar tıklayın ve tekrar etkinleştirmek için onaylayın:

![Ekran görüntüsü: örnek türünün etkinleştirilmesinin onaylanması](../assets/gpuservice/instance-types/activate-confirm.png)

## Örnek Türünü Silme { #deleting-an-instance-type }

Türün satırındaki eylemler menüsünü açın ve `Delete`'a tıklayın:

![Ekran görüntüsü: Örnek türünün eylemler menüsündeki Sil eylemi](../assets/gpuservice/instance-types/delete-menu.png)

Silme işlemini onaylayın:

![Ekran görüntüsü: örnek türünün silinmesinin onaylanması](../assets/gpuservice/instance-types/delete-confirm.png)

Özel bir tür kalıcı olarak kaldırılır. Türetme etkinken türetilmiş türler silinemez — bkz. [Türetilmiş Örnek Türleri](#derived-instance-types).

## Fiziksel Bölümleme (MIG) { #physical-partitioning-mig }

Fiziksel olarak bölümlenmiş (MIG) bir aygıt, kendi örnek tipini almaz: bölümleri, aygıtın örnek tipinde bölüm kapasitesi (`PT` görünümü) olarak görünür ve [bir örnek eklerken](gpuservice-instances.md#instance-type-selection) bir bölüm profili seçersiniz.

Operatör, NVIDIA, T-Head ve Hygon hızlandırıcılarda MIG'yi destekler. MIG modunun kendisi düğüm başına **, yönetici tarafından yönetilen** bir özelliktir; operatör bunu gözlemler ancak hiçbir zaman etkinleştirmez, devre dışı bırakmaz veya yeniden yapılandırmaz. Bir düğümde bölümler sunmak için satıcının araçlarıyla düğümde MIG'yi etkinleştirin ve operatörün cihaz yöneticisini orada, örneğin bir NVIDIA düğümünde yeniden başlatın:

```bash
# On the node, per GPU or for all GPUs:
sudo nvidia-smi -i <id> -mig 1
# Then, from anywhere with cluster access:
kubectl -n gpustack-system rollout restart ds/gpustack-operator-device-manager-nvidia
```

MIG örneklerinin düğüm yeniden başlatılmasından sonra asla hayatta kalamayacağı da dahil olmak üzere her satıcının tüm prosedürü, ön koşulları ve sınırlamaları için operatörün MIG operasyonları runbook'larına bakın:

- [NVIDIA MIG işlemleri](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/nvidia-mig/)
- [T-Kafalı MIG operasyonları](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/thead-mig/)
- [Hygon MIG işlemleri](https://docs.gpustack.ai/gpustack-operator/main/docs/modules/devices/hygon-mig/) — Hygon'da mod düğüm çapındadır ve bölümlenmiş bir düğüm yalnızca bölümlere hizmet eder.
