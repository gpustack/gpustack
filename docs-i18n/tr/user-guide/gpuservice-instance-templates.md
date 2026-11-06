---
translation_status: translated
---

# GPU Hizmet Örneği Şablonları { #gpu-service-instance-templates }

GPU Hizmet Örnek Şablonları, GPU Hizmet Örnekleri için yeniden kullanılabilir şablonlar tanımlamanıza olanak tanır.

Bir şablon, ortak yapılandırmayı (görüntü, komut, bağlantı noktaları, ortam değişkenleri vb.) bir kez yakalar, böylece aynı ayarlarla birden fazla örnek oluşturabilirsiniz.

## Şablonlara Göz Atın { #browse-templates }

Mevcut tüm şablonlara ve bunların ayrıntılarına göz atmak için `GPU Service` > `Instance Templates` sayfasına gidin.

![Ekran görüntüsü: Örnek Şablonları listesi](../assets/gpuservice/instance-templates/list.png)

Şablonları görünen ada veya ada (bir kartın gösterebileceği değere) veya satıcıya göre filtreleyebilirsiniz.

## Şablon Ekleme { #adding-a-template }

`Instance Templates` sayfasında, oluşturma formunu açmak için `Add Instance Template`'e tıklayın.

![Ekran görüntüsü: Örnek Şablonu Ekle formu](../assets/gpuservice/instance-templates/add.png)

!!! note

    GPU Hizmet Örneği esas olarak Kubernetes kümesinde çalışan bir Pod'dur, dolayısıyla yapılandırması kapsayıcıları çalıştırma kurallarına uyar.

Bir şablon aşağıdaki özellikleri belirlemenizi sağlar:

- **Name**: Şablon için benzersiz bir tanımlayıcı. Oluşturulduğunda ayarlanır ve daha sonra değiştirilemez.
- **Görünen Ad**: Şablon için kullanıcı dostu bir ad.
- **Satıcı**: GPU satıcısı (örneğin, `NVIDIA`) veya CPU tabanlı örnekler için `CPU`.
- **Image**: Bu şablondan oluşturulan örnekler tarafından kullanılan geçerli bir kapsayıcı görüntüsü.
- **Resim Çekme Politikası**: Kapsayıcı imajını çekme politikası (`Always`, `IfNotPresent` veya `Never`).
- **Konteyner Başlatma Komutu**: Konteyner başlatıldığında komut çalıştırılır.
- **Container Disk**: Container kök dosya sistemi için ayrılan disk kapasitesi.
- **Mount Path**: Çalışma dizininin monte edildiği konteynerin içindeki yol.
- **Bağlantı Noktaları**: Konteynerin açığa çıkardığı ağ bağlantı noktaları.
- **Ortam Değişkenleri**: Kapta ayarlanan ortam değişkenleri.

Gerekli alanları doldurduktan sonra `Save` tuşuna basarak şablonu oluşturun.

## Üretilen Değer Yer Tutucuları { #generated-value-placeholders }

Bir örnek oluştururken, GPUStack, başlangıç komutunda ve bağlantı noktası erişim parametrelerinde bulunan `{{generated_*}}` yer tutucuları çözer, örnek başına bir değer oluşturur ve bunu örnek spesifikasyonunda sürdürür; böylece değer, örneğin durdurulması/başlatılması sırasında aynı kalır. Şu anda desteklenenler:

- `{{generated_token}}`: rastgele 32 karakterlik onaltılık token. Örneğin, yerleşik JupyterLab şablonları, başlatma komutu olarak `--ServerApp.token={{generated_token}}` kullanır ve JUPYTER bağlantı noktasının erişim parametrelerinde aynı yer tutucuyu taşır; dolayısıyla `Connect` sütununda gösterilen web erişim bağlantısı, otomatik oturum açma için `?token=` sorgu parametresi olarak belirteci içerir.

Yer tutucular ve bağlantı noktası erişim parametreleri API (`spec.command` ve `spec.ports[].accessParams`) aracılığıyla ayarlanabilir; şablon formu henüz bunları göstermiyor.

## Şablon Düzenleme { #editing-a-template }

Yapılandırmasını açmak için şablon kartındaki `Edit`'ye tıklayın, değişikliklerinizi yapın ve `Save`'e tıklayın.

![Ekran görüntüsü: Örnek Şablonunu Düzenleme formunu](../assets/gpuservice/instance-templates/edit.png)

## Şablonu Silme { #deleting-a-template }

Bir şablon kartında `Delete` öğesine tıklayın ve onaylayın. Daha sonra şablon listeden kaldırılır.

## Şablonlardan Dağıtım { #deploying-from-templates }

Şablonlarınız tanımlandıktan sonra, onlardan örnekleri [GPU Hizmet Örnekleri](gpuservice-instances.md) sayfasında dağıtabilirsiniz.
