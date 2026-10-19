---
translation_status: translated
---

# DigitalOcean Kullanarak GPU Kümesi Ekleme { #adding-a-gpu-cluster-using-digitalocean }

Bir küme oluştururken GPUStack, çalışanlar oluşturmak ve onları GPUStack kümesine eklemek için DigitalOcean'dan yararlanabilir.

## Hazırlık { #preparation }

DigitalOcean hesabına kaydolmanız ve API sayfasında Kişisel Erişim Tokenı oluşturmanız gerekmektedir.

![dijital-okyanus-pat](../assets/tutorials/adding-gpucluster-using-digitalocean/digital-ocean-pat.png)

> Not: Belirteç kapsamı Tam Erişim olarak ayarlanmalıdır. Özel Kapsamları kullanarak izinleri seçerseniz Droplet örnekleri silerken sorunlarla karşılaşabilirsiniz.

GPUStack Sunucusunu başlatırken `--server-external-url` parametresini belirtmeniz gerekmektedir. Bu parametre, Droplet oluşturulduktan ve çalışan başlatıldıktan sonra çalışanın `--server-url`'ini yapılandırmak için kullanılır. Sunucunuz bir proxy arkasında çalışıyorsa lütfen proxy adresini, genel ağda çalışan Droplet örneklerinin başlatma sonrasında bu adresi kullanarak GPUStack Sunucu API'sine erişebilmesini sağlayacak şekilde ayarlayın.

## DigitalOcean Kümesi Oluşturun { #create-digitalocean-cluster }

Lütfen [DigitalOcean Kümesi Oluşturma](../user-guide/cluster-management.md#creating-digitalocean-cluster) bölümündeki adımlara bakın.

### Bulut Kimlik Bilgisi Oluşturun { #create-cloud-credential }

`Cloud Credentials` sayfasında bir DigitalOcean bulut kimlik bilgisi oluşturun.

![bulut oluştur](../assets/tutorials/adding-gpucluster-using-digitalocean/create-cloud-credential.png)

### Bulut Kimlik Bilgileriyle Küme Oluşturun { #create-cluster-with-cloud-credential }

Küme oluşturma sayfasında `Cloud Provider` olarak `DigitalOcean`'ı seçin:

![bulut sağlayıcı](../assets/tutorials/adding-gpucluster-using-digitalocean/select-cloud-provider.png)

Bir ad girin, az önce oluşturduğunuz bulut kimlik bilgisini seçin ve GPU Droplet örneklerini destekleyen bir bölge seçin.

![bölge seç](../assets/tutorials/adding-gpucluster-using-digitalocean/select-region.png)

Küme için bir çalışan havuzu oluşturmak için `Next` öğesine tıklayın.

### Çalışan Havuzu Oluşturun { #create-worker-pool }

`Name`, `Replicas` ve `Batch Size`'yi gerektiği gibi doldurun. Ardından çalışan havuzu için `Instance Type` (DigitalOcean terimleriyle Droplet boyutu) seçeneğini seçin.

![örnek tipini seçin](../assets/tutorials/adding-gpucluster-using-digitalocean/select-instance-type.png)

Ardından `OS Image`'ı seçin.

![os-resmini seç](../assets/tutorials/adding-gpucluster-using-digitalocean/select-os-image.png)

AMD GPU Droplet örnekleri çoğu zaman mevcut olmadığından şu anda yalnızca Nvidia serisi GPU Droplet örnekleri desteklenmektedir. Yalnızca Debian tabanlı işletim sistemleri desteklenir.

- Ubuntu 22.04 tabanlı ve sürücüleri, CUDA'yı ve önceden yüklenmiş kapsayıcı araç kitini içeren Nvidia AI/ML Ready (**Önerilen**) için ekstra paketlere gerek yoktur.
- Debian dağıtımları için Droplet, cloud-init aracılığıyla `nvidia-open` sürücüsü, CUDA 12.8 ve nvidia-container-toolkit 1.17.8-1 ile önyüklenir.
- Ubuntu dağıtımları için Droplet, cloud-init aracılığıyla `nvidia-driver-570` sürücüsü, CUDA 12.8 ve nvidia-container-toolkit 1.17.8-1 ile önyüklenir.

Çalışan havuzunda çalışan sağlamak üzere etiketlerin ve birimlerin ayarlanması desteklenir. Droplet için oluşturulan ve eklenen birimler, çalışan kapsayıcısına otomatik olarak eklenmeyecektir. Birimleri gerektiği gibi bağlamak için çalıştırma çalışan komut dosyasını `/opt/gpustack-run-worker.sh`'da değiştirebilirsiniz.

Her şey ayarlandıysa `Save`'a tıklayın.

## İşçilerin Tedarik Edilmesi Bekleniyor { #waiting-for-workers-to-be-provisioned }

Kümeyi kaydettikten sonra DigitalOcean çalışanlarının temel hazırlık ilerlemesini görüntülemek için `Workers` sayfasına gidin.

![erzak-işçisi](../assets/tutorials/adding-gpucluster-using-digitalocean/provisioning-workers.png)

Sağlama süreci birkaç adım içerir:

1. Bir SSH anahtarı oluşturun.
2. SSH anahtarıyla bir Droplet oluşturun.
3. Droplet örneklerin başlamasını bekleyin.
4. Genel IP'nin atanmasını bekleyin.
5. Hacimler oluşturun ve bunları Droplet örneğine ekleyin.
6. Worker `Initialized` durumuna girer ve Worker Container'ın başlayıp sunucuya bağlanmasını bekler.

Worker `Ready` durumuna ulaştığında modelleri onun üzerine dağıtabilirsiniz.

![işçinin sağladığı](../assets/tutorials/adding-gpucluster-using-digitalocean/worker-provisioned.png)

## DigitalOcean Çalışanlarının Ölçeklerini Güvenle Azaltın { #safely-scale-down-digitalocean-workers }

Bir DigitalOcean çalışanına artık ihtiyaç duyulmadığında Droplet örneğini güvenli bir şekilde yok etmek için şu adımları izleyin:

- Çalışan havuzunun kopya sayısını, silmek istediğiniz çalışan sayısıyla eşleşecek şekilde ayarlayın. Bu adımda çalışanların otomatik olarak silinmediğini unutmayın.
- Silmek istediğiniz çalışanlara hiçbir model örneğinin dağıtılmadığından emin olun.
- Gerektiğinde işçileri silin. İlgili Droplet örnekleri buna göre silinecektir.
