---
translation_status: translated
---

# Shuihua Kullanarak GPU Kümesi Ekleme { #adding-a-gpu-cluster-using-shuihua }

Bir küme oluştururken GPUStack, çalışanlar oluşturmak ve onları GPUStack kümesine eklemek için SHUIHUA FUTURE'dan yararlanabilir.

## Hazırlık { #preparation }

Bir SHUIHUA FUTURE hesabına ve [Shuihua konsolu](https://hub.do.top/cn/signin)'den oluşturulmuş bir API anahtarına ihtiyacınız var. Anahtarın önüne `amp_live_` eklenir.

![shuihua-api-anahtarı](../assets/tutorials/adding-gpucluster-using-shuihua/shuihua-api-key.png)

GPUStack Sunucusunu başlatırken `--server-external-url` parametresini belirtmeniz gerekmektedir. Shuihua bulut sunucuları kayıt olmak ve tünellerini açık tutmak için sunucuya bu adres üzerinden ulaşır, dolayısıyla sunucuya onlardan ulaşılabilir olması gerekir. Sunucunuz bir proxy arkasında çalışıyorsa proxy adresini buradan ayarlayın.

## Shuihua Kümesini Oluştur { #create-shuihua-cluster }

### Bulut Kimlik Bilgisi Oluşturun { #create-cloud-credential }

`Cloud Credentials` sayfasında bir `SHUIHUA FUTURE` bulut kimlik bilgisi oluşturun. `API Key` doldurulması gereken tek alandır.

![cloud-kimlik bilgisi oluştur](../assets/tutorials/adding-gpucluster-using-shuihua/create-cloud-credential.png)

### Bulut Kimlik Bilgileriyle Küme Oluşturun { #create-cluster-with-cloud-credential }

Küme oluşturma sayfasında `Cloud Provider` olarak `SHUIHUA FUTURE`'ı seçin:

![bulut-sağlayıcıyı seçin](../assets/tutorials/adding-gpucluster-using-shuihua/select-cloud-provider.png)

Bir ad girin, az önce oluşturduğunuz bulut kimlik bilgisini seçin ve `GPUStack Server URL`'ı yapılandırın. Shuihua'nın bölgesi olmadığı için seçilecek bölge de yok.

`Default Container Registry` burada gereklidir. Shuihua bulut sunucuları Docker Hub'a erişemediğinden GPUStack çalışan imajının başka bir yerden gelmesi gerekir; alan `quay.io` ve `swr.cn-south-1.myhuaweicloud.com`'yi önerir ve kendi Harbor kurulumunuz veya aynanız gibi yazabileceğiniz diğer tüm kayıtları kabul eder. Docker Hub ana bilgisayarına çözümlenen bir değer reddedilir.

![temel-yapılandırma](../assets/tutorials/adding-gpucluster-using-shuihua/basic-configuration.png)

> Not: Sunucunun `--image-name-override`'si bu alana göre önceliklidir. Sunucu bir geçersiz kılma ile çalışıyorsa, bu imaj adı kelimesi kelimesine kullanılır ve kayıt defteri ayarının hiçbir etkisi olmaz; bu nedenle geçersiz kılmanın kendisinin ulaşılabilir bir kayıt defterini adlandırması gerekir.

Küme için bir çalışan havuzu oluşturmak için `Next` öğesine tıklayın.

### Çalışan Havuzu Oluşturun { #create-worker-pool }

`Name`, `Replicas` ve `Batch Size`'yi gerektiği gibi doldurun.

![işçi havuzu yapılandırması](../assets/tutorials/adding-gpucluster-using-shuihua/worker-pool-configuration.png)

Ardından `Instance Type` seçin; Shuihua'da bu bir donanım yapılandırma şablonudur. Her seçenek GPU modelini, saatlik fiyatı ve kalan stok miktarını gösterir. Stoku tükenen şablon listede kalır ancak seçilemez; böylece neden kullanılamadığını görebilirsiniz.

![örnek tipini seçin](../assets/tutorials/adding-gpucluster-using-shuihua/select-instance-type.png)

Ardından `OS Image`'ı seçin.

Shuihua yalnızca NVIDIA GPU'ları kiralar ve imajları sürücü ve konteyner araç seti önceden yüklenmiş olarak gönderilir. Bu nedenle Cloud-init, Docker'ı yalnızca NVIDIA çalışma zamanına yönlendirir ve çalışanı başlatır; hiçbir şey yüklenmez ve örnek yeniden başlatılmaz, dolayısıyla bu çalışanlar, bir sürücü oluşturmak zorunda olanlardan daha hızlı ortaya çıkar.

Bu sağlayıcı için birimler sunulmuyor: Shuihua'nın blok depolama API'si yok. Etiketler desteklenir.

Her şey ayarlandıysa `Save`'a tıklayın.

## İşçilerin Tedarik Edilmesi Bekleniyor { #waiting-for-workers-to-be-provisioned }

Kümeyi kaydettikten sonra Shuihua çalışanlarının temel hazırlık ilerlemesini imajlemek için `Workers` sayfasına gidin.

![tedarik-işçiler](../assets/tutorials/adding-gpucluster-using-shuihua/provisioning-workers.png)

Sağlama süreci birkaç adım içerir:

1. Bir SSH anahtarı oluşturun. Shuihua'nın SSH anahtarı API'si yoktur, dolayısıyla anahtar sağlayıcıya kayıtlı değildir; bunun yerine örneğin cloud-init'ine yazılır.
2. Örneği oluşturun. Yaratılış eş zamanlı değildir — Shuihua isteği kabul eder ve `creating`, ardından `processing` ve ardından `active` raporunu verir.
3. Örneğin başlamasını bekleyin.
4. Adresinin atanmasını bekleyin.
5. Worker `Initialized` durumuna girer ve Worker Container'ın başlayıp sunucuya bağlanmasını bekler.

Worker `Ready` durumuna ulaştığında modelleri onun üzerine dağıtabilirsiniz.

![işçinin sağladığı](../assets/tutorials/adding-gpucluster-using-shuihua/worker-provisioned.png)

## Bir Çalışana SSH Üzerinden Bağlanma { #connecting-to-a-worker-over-ssh }

Shuihua, her örneği tek bir paylaşılan genel IP'nin arkasında yayınlayarak yalnızca 22 ve 80 numaralı bağlantı noktalarını örneğe özel bir bağlantı noktasıyla eşleştirir. Çalışan için gösterilen IP, örneğin kendisini tanımlayan ancak bağlanılamayan kendi özel adresidir.

Bunun yerine çalışanın işlemleri menüsünde `View SSH Access` kullanın. Örneğe gerçekten ulaşan `Host`, `Port` ve `Connection Command`'yi gösterir ve `Private Key`'i (`worker-<id>-private_key.pem` olarak kaydedilmiş) indirmenize izin verir.

![işçi-ssh-bilgisi](../assets/tutorials/adding-gpucluster-using-shuihua/worker-ssh-info.png)

İşçinin kendi portu hiçbir zaman haritalandırılmadığından ağ geçidi gelen bu işçilere ulaşamaz. Sunucunun WebSocket tüneli aracılığıyla çıkarım sunarlar; bu nedenle Shuihua kümesi varsayılan olarak `tunnel` proxy modunu kullanır. Ağınız örneklere doğrudan yönlendirme yapmadığı sürece olduğu gibi bırakın.

## Shuihua İşçilerinin Ölçeklerini Güvenle Azaltın { #safely-scale-down-shuihua-workers }

Bir Shuihua çalışanına artık ihtiyaç duyulmadığında, örneği güvenli bir şekilde yok etmek için şu adımları izleyin:

- Çalışan havuzunun kopya sayısını, silmek istediğiniz çalışan sayısıyla eşleşecek şekilde ayarlayın. Bu adımda çalışanların otomatik olarak silinmediğini unutmayın.
- Silmek istediğiniz çalışanlara hiçbir model örneğinin dağıtılmadığından emin olun.
- Gerektiğinde işçileri silin. İlgili örnekler buna göre sonlandırılacaktır.

## Sorun giderme { #troubleshooting }

**İşçi `Initialized`'de kalır ve hiçbir zaman `Ready` olmaz.** Örnek önyüklendi ancak çalışan kapsayıcısı başlamadı, bunun nedeni çoğunlukla imaj çekilememesiydi. SSH üzerinden bağlanın ve şunları kontrol edin:

```bash
sudo cloud-init status --long
sudo grep -iE "pull|denied|manifest" /var/log/cloud-init-output.log
docker ps -a
```

**İşçi `Ready` ancak GPU bildirmiyor.** Docker, NVIDIA çalışma zamanını kullanmıyor:

```bash
nvidia-smi
which nvidia-ctk
docker info | grep -i runtime
```

**Bir çalışan temel hazırlığı yapamadı ve yerinde yeniden denenemiyor.** Çalışanı silin ve havuzun yeni bir çalışan oluşturmasına izin verin. Her oluşturma girişimi, çalışan satırından türetilen bir tekrar koruma anahtarını taşır; bu nedenle, yeni bir örnek almak için yeni bir çalışan gerekir.
