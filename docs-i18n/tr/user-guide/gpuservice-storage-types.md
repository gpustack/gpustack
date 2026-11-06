---
translation_status: translated
---

# GPU Hizmeti Depolama Türleri { #gpu-service-storage-types }

GPU Hizmeti Depolama Türleri, GPU Hizmeti Örnekleri için paylaşılan dosya sistemi depolama arka uçlarını (NFS veya S3) tanımlamanıza olanak tanır.

Bu, yöneticilerin birden çok Kubernetes kümesindeki depolama tanımlarını yönetmesine yardımcı olur. Tanımlandıktan sonra, [GPU Hizmet Depolaması](gpuservice-storage.md) oluşturulurken bir depolama türü seçilebilir.

Temel olarak her depolama türü, hedef kümede bir Kubernetes `StorageClass` olarak gerçekleştirilir.

## Depolama Türlerine Göz Atın { #browse-storage-types }

Tüm depolama türlerine ve ayrıntılarına göz atmak için `GPU Service` > `Storage Types` sayfasına gidin.

![Ekran görüntüsü: Depolama Türleri listesi](../assets/gpuservice/storage-types/list.png)

Depolama türlerini görünen ad veya ada göre filtreleyebilirsiniz; `Name` sütununun gösterebileceği değerlerden herhangi biri. `Name` sütun sıralarını görüntülediği etikete göre sıralama.

## Depolama Türü Ekleme { #adding-a-storage-type }

`Storage Types` sayfasında, oluşturma formunu açmak için `Add Storage Type`'e tıklayın.

![Ekran görüntüsü: Depolama Türü formunu ekleyin](../assets/gpuservice/storage-types/add.png)

Bir `Name`, isteğe bağlı olarak `Display Name` doldurun ve ardından bir `Type` seçin. Kalan alanlar seçtiğiniz türe bağlıdır.

**NFS** — NFS sunucusundan dışa aktarılan bir paylaşımı bağlayın:

- **NFS Sunucusu** (gerekli): NFS sunucusunun ana bilgisayar adı veya IP adresi.
- **Paylaşım Yolu** (gerekli): NFS sunucusunda dışa aktarılan paylaşım yolu.
- **Alt Dizin**: Birim için kullanılacak paylaşım içindeki bir alt dizin. Boş bırakılırsa paylaşımın kökü kullanılır.
- **Bağlama İzinleri**: Takılan dizine uygulanan izinler, örneğin `0777`. NFS sunucusunda yapılandırılan izinlere uygun olarak varsayılan olarak `0`'dir.
- **Montaj Seçenekleri**: NFS montaj seçenekleri, ayrı girişler olarak eklendi. Varsayılan olarak `hard`, `vers=4`, `rsize=1048576`, `wsize=1048576`, `noatime` ve `nodiratime` şeklindedir.

**S3** — S3 uyumlu bir nesne depolama paketi monte edin:

- **Endpoint** (gerekli): S3 uyumlu depolama uç noktası URL'si.
- **Region**: Varsa, S3 depolamanın bölgesi.
- **Bucket**: Kullanılacak paket adı. Boş bırakılırsa `gpu-instance-pv-{id}` adlı varsayılan bir paket oluşturulur.
- **Erişim Anahtarı** (gerekli): Depolama alanında kimlik doğrulaması yapmak için kullanılan erişim anahtarı.
- **Gizli Anahtar** (gerekli): Depolama alanında kimlik doğrulaması yapmak için kullanılan gizli anahtar. Salt yazılır ve API yanıtlarında asla döndürülmez.
- **Insecure**: Uç noktaya bağlanırken TLS/SSL sertifika doğrulamasını atlayın. Varsayılan olarak devre dışıdır.
- **Montaj Seçenekleri**: Temel [GeeseFS](https://github.com/yandex-cloud/geesefs) sürücüsü için montaj seçenekleri, ayrı girişler olarak eklendi. Varsayılan değerler, çoğu iş yüküne uygun olan **Büyük dosyalar için yoğun yazma** ön ayarını takip eder.

S3 birimleri [GeeseFS](https://github.com/yandex-cloud/geesefs) ile bağlanır ve `Mount Options` doğrudan ona iletilir. Erişim düzeninize uygun ön ayarı seçin ve uç noktanız Yandex Nesne Depolaması olmadığında **Yandex S3 uyumlu olmayan hizmet** seçeneklerini ekleyin.

??? note "Ortak GeeseFS Montaj Seçenekleri ön ayarları"

    | Kullanım örneği | Montaj seçenekleri |
    | --- | --- |
    | Büyük dosyalar için yoğun yazma (varsayılan) | `--no-checksum` `--memory-limit=4000` `--max-flushers=32` `--max-parallel-parts=32` `--part-sizes=25` |
    | Büyük dosyalar için sıralı okuma | `--read-ahead-large=200000` `--large-read-cutoff=10240` `--read-ahead-parallel=40000` `--memory-limit=8000` |
    | Küçük dosyalar için rastgele okuma | `--read-ahead-small=64` `--small-read-cutoff=64` `--read-ahead=1024` `--stat-cache-ttl=300s` `--entry-limit=200000` |
    | Yazma için yüksek kullanılabilirlik | `--sdk-max-retries=10` `--read-retry-attempts=5` `--fsync-on-close` `--cache=/mnt/disk-cache` |
    | Yandex dışı S3 uyumlu hizmet | `--list-type=2` `--no-specials` |

Gerekli alanları doldurduktan sonra `Save` tuşuna basarak depolama tipini oluşturun.

## Depolama Türünü Düzenleme { #editing-a-storage-type }

Oluşturulduktan sonra yalnızca görünen ad değiştirilebilir.

!!! note

    Gelecekteki bir sürüm için depolama türü yapılandırmasının daha fazla düzenlenmesi planlanmaktadır.

## Depolama Türünü Silme { #deleting-a-storage-type }

Bir depolama türünde `Delete` öğesine tıklayın ve onaylayın. Depolama türü daha sonra listeden kaldırılır.

!!! warning

    Bir depolama türünü silmeden önce, ilk olarak ondan oluşturulan herhangi bir [GPU Hizmet Depolaması](gpuservice-storage.md)'i silin.
