---
translation_status: translated
---

# GPU Hizmet Örnekleri { #gpu-service-instances }

GPU Hizmet Örnekleri, SSH ile erişilebilen bir GPU örneğini birkaç dakika içinde başlatmanıza olanak tanır.

Bu size öğrenme, çıkarım, test etme ve ince ayar yapma gibi görevler için tek bir cihazı veya birden fazla cihazı içeren sanal makine benzeri bir ortam sağlar.

Temel olarak, her bir GPU Hizmet Örneği bir Kubernetes Pod tarafından desteklenir, dolayısıyla GPU örnekleri ile yalnızca CPU örneklerinin karıştırılması tüm makinenin (VM veya çıplak donanım) kaynak kullanımını iyileştirir.

GPUStack birden fazla Kubernetes kümesini yönetir ve bunlardan herhangi birinde GPU Hizmet Örneklerini başlatmak için birleşik bir arayüz sağlar.

!!! note "GPUStack 2.2'den yükseltme mi yapıyorsunuz?"

    GPUStack 2.3, yerinde yükseltmeyi kendisi tamamlayan güncel GPUStack Operator sürümünü içerir: çalışanın Deployment'ı eskisiyle yenisi aynı anda çalıştırılmadan değiştirilir (`Recreate`), çalışana 900 saniyelik başlangıç süresi tanınır ve v0.5.x'ten kalan uygulama başına eski Helm sürümleri çalışanın yönetimine alınır.

    Yükseltilen çalışan sağlıklı olduktan sonra, operatörün temizleme komut dosyasıyla artık kalan v0.5.x planlama nesnelerini kaldırın:

    ```bash
    curl -fsSLO https://raw.githubusercontent.com/gpustack/gpustack-operator/main/docs/operate/migration/cleanup-v0.5-orphans.sh
    bash cleanup-v0.5-orphans.sh --dry-run   # preview, changes nothing
    bash cleanup-v0.5-orphans.sh             # delete the orphans
    ```

    Daha önceki bir yükseltme girişimi zaten sıkıştıysa (Kueue CRD'leri `Terminating` takılı kaldı, çalışan asla Hazır olmuyor), operatörün [geçiş sorun giderme](https://docs.gpustack.ai/gpustack-operator/main/docs/operate/migration/troubleshooting/) bölümüne bakın. Prosedürün tamamı için operatörün [geçiş kılavuzunu](https://docs.gpustack.ai/gpustack-operator/main/docs/operate/migration/from-v0.5/) izleyin.

## Önkoşullar { #prerequisites }

GPU Hizmet Örneklerini kullanmak için en az bir Kubernetes kümesine ihtiyacınız vardır.

`GPU Service` > `GPU Instances` sayfasını herhangi bir küme eklenmeden ilk açtığınızda, bir küme eklemeniz istenir.

![Ekran görüntüsü: Kubernetes kümesi eklenmesini isteyen boş GPU Örnekleri sayfası](../assets/gpuservice/instances/cluster-no-item.png)

`Resources` > `Clusters` sayfasına gitmek için `Add a Kubernetes Cluster`'e tıklayın.

![Ekran görüntüsü: Küme formu oluştur](../assets/gpuservice/instances/cluster-add.png)

Küme Oluştur sihirbazı `Select Provider` (`Kubernetes`'i seçin) → `Configure` → `Complete` adımlarını adım adım açıklar. `Configure` adımında, `Name` ve isteğe bağlı `Description` kümesini doldurun, `GPU Service` `Cluster Type`'i seçin ve ardından `Save`'e tıklayın.

!!! warning

    Bir Kubernetes kümesi yalnızca tek bir amaca hizmet edebilir — `Model Service` veya `GPU Service`.

### Gelişmiş Seçenekler { #advanced-options }

`GPU Service` için bir küme eklerken, daha fazla ayar yapılandırmak için `Advanced` seçeneklerini genişletin.

#### Düğümlerden Örnek Türlerini Türetme { #derive-instance-types-from-nodes }

Varsayılan olarak etkindir. Operatör, her düğümün aygıtlarını keşfeder ve ürün birimi başına CPU/RAM ön ayarlarıyla eşleşen örnek türlerini (hızlandırıcı modeli başına bir tane ve yalnızca CPU türü) otomatik olarak türetir. Her örnek tipini kendiniz yazmayı tercih ediyorsanız bunu `Disabled` olarak ayarlayın; operatör daha sonra tür oluşturmayı durdurur ve türetilmiş olanlar kalıcı olarak silinebilir.

![Ekran Görüntüsü: Düğümlerden Örnek Türlerini Türetme ayarı, varsayılan Etkin](../assets/gpuservice/instances/cluster-instance-type-derived.png)

Türetilen türleri yönetmek için [GPU Hizmet Örnek Türleri](gpuservice-instance-types.md)'e bakın.

#### Bir Düğümde Karma Örnek Türlerine İzin Ver { #allow-mixed-instance-types-on-a-node }

Varsayılan olarak etkindir. Bir düğümün aynı anda farklı örnek türlerine hizmet vermesine olanak tanır; örneğin, bir GPU düğümü hem GPU örneklerini hem de yalnızca CPU örneklerini barındırabilir. Her düğümü tek bir örnek tipine ayırmak için bunu `Disabled` olarak ayarlayın.

![Ekran Görüntüsü: Bir Düğümde Karma Örnek Türlerine İzin Ver ayarı, varsayılan Etkin](../assets/gpuservice/instances/cluster-instance-type-mixed.png)

#### GPU Hizmet Örneklerine Erişilmesine İzin Ver { #allow-gpu-service-instances-to-be-accessed }

Genellikle tüm Kubernetes düğümleri bir NAT veya güvenlik duvarının arkasında bulunur, dolayısıyla düğüm IP'lerine kümenin dışından erişilemeyebilir:

```bash
$ kubectl get nodes -o wide
NAME                                 STATUS   ROLES    AGE   VERSION   INTERNAL-IP   EXTERNAL-IP   OS-IMAGE              KERNEL-VERSION         CONTAINER-RUNTIME
computeinstance-e00agjsc8n3yhxqkh6   Ready    <none>   2h    v1.33.7   10.0.51.32    <none>        Ubuntu 24.04.4 LTS   6.11.0-1016-nvidia     containerd://1.7.34
computeinstance-e00bemd340vg9ypxrv   Ready    <none>   2h    v1.33.7   10.0.51.13    <none>        Ubuntu 24.04.4 LTS   6.11.0-1016-nvidia     containerd://1.7.34
computeinstance-e00g7g0y04ga384xj7   Ready    <none>   2h    v1.33.7   10.2.0.0      <none>        Ubuntu 24.04.4 LTS   6.11.0-1016-nvidia     containerd://1.7.34
computeinstance-e00prd70dh2mne0ajb   Ready    <none>   2h    v1.33.7   10.0.0.57     <none>        Ubuntu 24.04.4 LTS   6.11.0-1016-nvidia     containerd://1.7.34
```

Bu durumda, GPU Hizmet Örneklerine erişilebilmesi için `GPU Service Static Access Address`'de bir adres ayarlayın (örneğin, LoadBalancer VIP).

![Ekran görüntüsü: GPU Hizmeti statik erişim adresini yapılandırma](../assets/gpuservice/instances/cluster-access-address-configure.png)

#### GPUStack Çalışanının Ulaşılabilir Olduğundan Emin Olun { #ensure-the-gpustack-worker-is-reachable }

Çoğu zaman, GPU Hizmetinin örnekleri yönetmek, oluşturmak, silmek ve izlemek için Kubernetes kümesine (tersine) ulaşması gerekir.

GPUStack sunucusu Kubernetes kümesinin dışında konuşlandırılmışsa, GPUStack Worker **Tunnel** modunu etkinleştirmek için `Worker Configuration`'de `proxy_mode=tunnel`'i ayarlayın. Bu, çalışandan GPUStack sunucusuna uzun ömürlü bir bağlantı sağlar ve GPU Hizmetinin kümeye ulaşması için bir tünel sağlar.

![Ekran görüntüsü: Çalışan Yapılandırmasında proxy_mode'u tünel olarak ayarlama](../assets/gpuservice/instances/cluster-worker-proxy-mode-configure.png)

GPUStack sunucusu ile çalışan arasındaki bağlantıyı `Resources` > `Workers` sayfasında çalışanın `Status` sütununu kontrol ederek doğrulayabilirsiniz.

![Ekran görüntüsü: Çalışan bağlantı durumu kontrol ediliyor](../assets/gpuservice/instances/cluster-worker-connectivity-state-check.png)

## Örnek Türlerinin Otomatik Keşfi { #automatic-discovery-of-instance-types }

`GPU Service` özellikli bir Kubernetes kümesi ekledikten sonra, [GPUStack Operator](https://github.com/gpustack/gpustack-operator) otomatik olarak kümedeki GPU cihazlarını keşfeder, bilgilerini toplar ve GPU Hizmeti için karşılık gelen örnek türlerini oluşturur.

Bulut sunucusu türlerini yönetmek için bkz. [GPU Hizmet Bulut Sunucusu Türleri](gpuservice-instance-types.md). Bu sayfanın geri kalanında örneklerin dağıtılması ve yönetilmesi ele alınmaktadır.

## Örnek Ekleme { #adding-an-instance }

`GPU Service` > `GPU Instances` sayfasında, oluşturma formunu açmak için `Add GPU Instance`'ye tıklayın.

![Ekran görüntüsü: GPU Örneği Ekleme formu](../assets/gpuservice/instances/add.png)

### Örnek Türü Seçimi { #instance-type-selection }

En soldaki `Instance Types` sütunu, Kubernetes kümelerinden keşfedilen tüm örnek türlerini listeler.

Her bulut sunucusu tipi kartı aşağıdaki bilgileri gösterir:

![Ekran görüntüsü: Örnek türü kartı](../assets/gpuservice/instances/type-item.png)

- **Name**: Örnek türünün ürün adı, örneğin `NVIDIA-H100-80GB-HBM3`. Özel karakterler içeren ürünler Kubernetes açısından güvenli olacak şekilde sterilize edilir.
- **Üretici/Satıcı**: Sağ üstte üreticiyi veya satıcıyı gösteren etiket (örneğin, `NVIDIA` veya yalnızca CPU için `GENERIC`).
- **RAM/VRAM/CPU**: Bu türdeki birim başına verilen ana bilgisayar RAM'i, cihaz bellek kapasitesi ve CPU çekirdekleri.
- **Arch**: `AMD64` veya `ARM64` gibi CPU mimarisi — uyumlu bir görüntü seçerken kullanışlıdır.
- **Max**: Aynı anda seçebileceğiniz maksimum cihaz sayısı. **Max** 0 olduğunda, o bulut sunucusu tipindeki tüm cihazlar tahsis edilir ve bazıları serbest bırakılana kadar bunu seçemezsiniz.
- **Sliceable**: Bu türdeki cihazlar nasıl bölünebilir — mantıksal dilim bütçesi (yüzde olarak) ve bölümleme özellikli cihazlarda hala mevcut olan donanım bölümlerinin sayısı.

#### Örnek Türüne ait Birim Kaynakları { #unit-resources-of-an-instance-type }

Birim kaynakları, bir hızlandırıcı tipi için, tüm cihaz başına, bir türün birimi başına verilen ana bilgisayar CPU'su ve RAM'dir. Düğümlerden türetilen türlerin boyutu, operatör tarafından sağlanan ürün başına ön ayarlardan belirlenir (bkz. [birim kaynakları referansı](https://docs.gpustack.ai/gpustack-operator/main/docs/reference/instance-type-unit-resources/)); mantıksal dilim veya fiziksel bölüm daha sonra VRAM paylaşımına göre ön ayardan boyutlandırılır. Ayrıca özel birim kaynaklarıyla [kendi türlerinizi](gpuservice-instance-types.md#adding-an-instance-type)yazabilirsiniz.

#### Bütün, Dilimlenmiş veya Bölümlenmiş Bir Cihazı Seçme { #selecting-a-whole-sliced-or-partitioned-device }

Hızlandırıcı türü için `Configuration` paneli üç adede kadar tahsis modu sunar:

- **Tam GPU** — tüm cihazın özel kullanımı.
- **Oranla** — mantıksal bir dilim: bir `VRAM Percentage` ve bir `Compute Percentage` seçin. Cihaz bir bütün olarak kalır ve aynı anda diğer dilimlere hizmet eder; her bütçe çalışma zamanında uygulanır.

![Ekran görüntüsü: VRAM ve hesaplama oranına göre mantıksal bir dilim seçme](../assets/gpuservice/instances/type-select-sliced.png)

- **Profile Göre** — MIG özellikli bir cihazın fiziksel bölümü: bir `Partition Profile` seçin (örneğin, H100'de `1g.10gb`). Operatör sizin için MIG örneğini gerçekleştirir. Bu sekme yalnızca kümedeki bir düğümde MIG etkinleştirildiğinde görünür; MIG'nin etkinleştirilmesi, düğüm başına bir yönetici işlemidir, bkz. [GPU Hizmet Örneği Türleri](gpuservice-instance-types.md#physical-partitioning-mig).

![Ekran görüntüsü: MIG profiline göre fiziksel bir bölüm seçme](../assets/gpuservice/instances/type-select-partitioned.png)

#### Yalnızca CPU Örnek Türü { #cpu-only-instance-type }

Genel kaynak kullanımını iyileştirmek için GPU Hizmeti ayrıca yalnızca CPU örnek türlerini de destekler.

Başlangıçta, GPUStack Operatörü yalnızca CPU örnek türü için sabit bir profil sağlar: `1 CPU + 2 GB RAM`. Özelleştirmek için [GPU Hizmet Örnek Türleri](gpuservice-instance-types.md)'ye bakın.

### Örnek Şablonu Seçimi { #instance-template-selection }

Ortadaki `Instance Templates` sütun, mevcut şablonları listeler ve farklı örnek türlerini seçtiğinizde liste güncellenir.

Şablonlar `GPU Service` > `Instance Templates` sayfasından yönetilir; bkz. [GPU Hizmet Örneği Şablonları](gpuservice-instance-templates.md).

### Örnek Yapılandırması { #instance-configuration }

En sağdaki `Configuration` formu, yeni örneğin ayrıntılarını ayarlamanızı sağlar. Beş bölüme ayrılmıştır: `Basic`, `Instance Type`, `Instance Template`, `Storage` ve `SSH Access`.

- **Basic**: Örnek adı ve görünen ad.
- **Örnek Türü**: Örnek türü ve tahsis modu — tüm cihaz, mantıksal dilim veya fiziksel bölüm.
- **Örnek Şablonu**: Devralınacak şablon (görüntü, komut, ortam değişkenleri vb.). Bir şablon seçtikten sonra da yapılandırmayı ayarlayabilirsiniz.
- **Depolama**: Ya `Ephemeral` depolama ya da [`Persistent` depolama](gpuservice-storage.md).
- **SSH Erişimi**: Örneğe atanacak [SSH ortak anahtarları](gpuservice-ssh-public-keys.md) veya SSH erişimini devre dışı bırakma seçeneği.

Formu doldurduktan sonra örneği oluşturmak için `Save`'e tıklayın.

## Örneklere Göz Atın { #browse-instances }

Oluşturma işleminden sonra, tüm örneklerin `Name` (ayarlanmışsa görünen ad), `Connect`, `Status`, `Instance Type`, `GPU`, `VRAM`, `CPU`, gibi sütunlarla listelendiği `GPU Service` > `GPU Instances` sayfasına dönersiniz. `RAM`, `Storage`, `Cluster`, `Creator`, `Created` ve `Operations`.

![Ekran görüntüsü: GPU Örnekleri listesi](../assets/gpuservice/instances/list.png)

Örnekleri görünen ada veya ada göre filtreleyebilirsiniz; `Name` sütununun gösterebileceği değerlerden herhangi biri. `Name` sütun sıralarını görüntülediği etikete göre sıralama.

### Bir Örneğe Erişim { #accessing-an-instance }

`Connect` sütunu, örneğin bağlantı noktası yapılandırmasına bağlı olarak örneğin erişim adreslerini (bir kopyala-yapıştır SSH komutu ve/veya web sayfalarına tıklanabilir bağlantılar) gösterir.

Örneğin doğrudan örneğe bağlanmak için SSH komutunu terminalinize yapıştırın.

![Ekran görüntüsü: SSH komutunu Bağlan sütunundan kopyalama](../assets/gpuservice/instances/list-item-ssh-access.png)

### Metrikler { #metrics }

`GPU`, `VRAM`, `CPU`, `RAM` ve `Storage` sütunları, sayfa görünür durumdayken birkaç saniyede bir yenilenen, çalışan her örnek için canlı kullanım göstergelerini gösterir. Her gösterge, tahsis modu ne olursa olsun, örneğin **kendi** kullanımını **kendi** kotasına göre raporlar: özel bir örnek tüm cihazı okur, bir dilim kendi paylaşımını okur ve fiziksel bir bölüm, bölümün kendi kapasitesini okur.

Ölçülemeyen bir rakam sıfır yerine boş olarak gösterilir - örneğin, bir MIG bölümü GPU çekirdek kullanımı olmadığını bildirir (sürücü bunu ölçemez), dolayısıyla `GPU` göstergesi `--` kalırken `VRAM` göstergesi hala çalışır.

Rakamlar, operatörün örnek metrikleri alt kaynağından ve düğüm aktarıcısından gelir; her alan, kaynağı ve sınırları için [Örnek Metrikleri referansına](https://docs.gpustack.ai/gpustack-operator/main/docs/reference/instance-metrics/) bakın.

!!! note "Yukarıdaki göstergeler hakkında"

    Liste ekran görüntüsü, yapay bir yük altında üç demo örneğini gösteriyor: her biri, ana bilgisayar RAM'ini tahsis eden ve cihazında 8192×8192 matris çarpımını tekrarlayan bir Python/PyTorch döngüsü çalıştırıyordu (bir döngüde `a = a @ a`). Bu nedenle `GPU` göstergesi özel L40S'de ve %50 dilimli H100'de %100 okur (kendi hesaplama payını doyuran bir dilim %100 okur) ve MIG örneğinin `GPU` göstergesi `--` gösterirken `CPU` %100 okur.

## Bir Örneği Düzenleme { #editing-an-instance }

Bir örneğin yapılandırmasını açmak için `Edit` öğesine tıklayın.

![Ekran görüntüsü: Örnek formunu düzenleyin](../assets/gpuservice/instances/edit.png)

Alanların çoğu henüz düzenlenemez; yalnızca görünen ad ve SSH erişim yapılandırması değiştirilebilir. SSH genel anahtarlarını ekleyebilir veya kaldırabilir ya da SSH erişimini devre dışı bırakabilirsiniz.

!!! note

    Gelecekteki bir sürüm için örnek yapılandırmasının daha fazla düzenlenmesi planlanmaktadır.

## Bir Örneği Çalıştırmak { #operating-an-instance }

GPU Hizmeti örnekler üzerinde çeşitli işlemler sağlar: günlükleri görüntüleme, etkinlikleri görüntüleme, durdurma/başlatma ve silme.

### Günlükleri Görüntüle { #view-logs }

Örneğin günlüklerini görmek için `View Logs` öğesine tıklayın. Bu, Jupyter Notebook veya TensorBoard gibi uygulamaların çıktılarını izlemek için kullanışlıdır.

![Ekran görüntüsü: Örnek günlükleri görüntüleniyor](../assets/gpuservice/instances/view-logs.png)

### Etkinlikleri Görüntüle { #view-events }

Örneğin olaylarını görmek için `View Events`'a tıklayın. Bu, durum değişikliklerini izlemenize yardımcı olur; örneğin, `Pending`, `Running` veya `Failed`'de kalmış bir örnek.

![Ekran görüntüsü: Örnek olaylarını görüntüleme](../assets/gpuservice/instances/view-events.png)

### Durdur/Başlat/Sil { #stopstartdelete }

Bir örneği durdurmak için `Stop`'e ve tekrar başlatmak için `Start`'e tıklayın; kaynakları geçici olarak serbest bırakmak ve daha sonra çalışmaya devam etmek için kullanışlıdır.

!!! warning

    Bir örneğin durdurulması bilgi işlem kaynaklarını serbest bırakır ve başlatılması, örneği yeniden oluşturur. Daha sonra örneğe yeni bir IP adresi atanır ve geçici depolamadaki tüm veriler kaybolur.

Bir örneği silmek ve tüm kaynaklarını serbest bırakmak için `Delete` seçeneğine tıklayın.
