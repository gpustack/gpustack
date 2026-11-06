---
translation_status: translated
---

# Dağıtılmış vLLM ile DeepSeek R1 671B'yi Çalıştırma { #running-deepseek-r1-671b-with-distributed-vllm }

Bu eğitim, bir GPUStack kümesinde **Distributed vLLM** kullanarak orijinal **DeepSeek R1 671B**'yi yapılandırma ve çalıştırma sürecinde size rehberlik eder. Modelin son derece büyük boyutundan dolayı, genellikle birden fazla çalışana dağıtılmış çıkarım yapılması gerekir.

GPUStack, vLLM kullanarak dağıtılmış çıkarımın kolay kurulumunu ve düzenlenmesini sağlayarak DeepSeek R1 gibi büyük modellerin minimum manuel yapılandırmayla çalıştırılmasını mümkün kılar.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdaki gereksinimlerin karşılandığından emin olun:

- Her biri gerekli GPU'larla donatılmış yeterli sayıda Linux düğümüne erişiminiz var. Örneğin:

<div class="center-table" markdown>

| **GPU** | **Düğüm Sayısı** |
| ---------------- | ------------------- |
| H100/H800:8      | 2                   |
| A100/A800-80GB:8 | 4                   |
| A100/A800:8      | 8                   |

</div>
- Optimum performans için NVLink veya InfiniBand gibi yüksek hızlı ara bağlantılar önerilir.
- Model dosyaları her düğümde aynı yola indirilmelidir. GPUStack anında model indirmeyi desteklerken, ağ hızına bağlı olarak zaman alıcı olabileceğinden ön indirme önerilir.

!!! note

    - Bu eğitimde, her biri 8 A800-80GB GPU ile donatılmış ve 200G InfiniBand aracılığıyla bağlanan 4 düğümden oluşan bir kurulum varsayıyoruz.
    - A100/A800 GPU'lar, orijinal olarak DeepSeek R1 tarafından kullanılan FP8 hassasiyetini desteklemez. Bu nedenle [Unsloth](https://huggingface.co/unsloth/DeepSeek-R1-BF16)'in BF16 versiyonunu kullanıyoruz.

## Adım 1: GPUStack Sunucusunu Kurun { #step-1-install-gpustack-server }

[Installation](../installation/installation.md)'e göre GPUStack sunucusunu başlatmak için aşağıdaki komutu kullanabilirsiniz:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    gpustack/gpustack

```

!!! note

    - `/path/to/your/model`'i gerçek yolla değiştirin.

GPUStack sunucusu açıldıktan ve çalışmaya başladıktan sonra, başlangıç yönetici şifresini almak için aşağıdaki komutları çalıştırın:

```bash
sudo docker exec gpustack \
    cat /var/lib/gpustack/initial_admin_password

```

## Adım 2: GPUStack Kullanıcı Arayüzü'ne erişin { #step-2-access-gpustack-ui }

`admin` kullanıcısını ve elde edilen şifreyi kullanarak GPUStack kullanıcı arayüzünde oturum açın.

```
http://your_gpustack_server_ip_or_hostname
```

## Adım 3: GPUStack Workers'ı yükleyin { #step-3-install-gpustack-workers }

GPUStack kullanıcı arayüzünde `Workers` sayfasına gidin, işçi ekleme komutunu almak için `Add Worker` düğmesine tıklayın.

Ve ardından **her çalışan düğümünde**, bir GPUStack çalışanını başlatmak için çalışan ekleme komutunu çalıştırın:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime nvidia \
    gpustack/gpustack \
    --server-url http://your_gpustack_server_ip_or_hostname \
	--token your_gpustack_cluster_token

```

!!! note

    - Yer tutucu yollarını, IP adresini/ana bilgisayar adını ve küme belirtecini uygun şekilde değiştirin.
    - `/path/to/your/model`'i, sisteminizde DeepSeek R1 model dosyalarının depolandığı gerçek yolla değiştirin.

Tüm çalışanlar eklendikten sonra GPUStack kullanıcı arayüzüne geri dönün.

Tüm çalışanların Hazır durumda olduğunu ve GPU'larının listelendiğini doğrulamak için `Workers` sayfasına gidin.

![başlangıç-kaynakları](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/initial-resources.png)

## Adım 4: DeepSeek R1 Modelini Dağıtın { #step-4-deploy-the-deepseek-r1-model }

1. `Deployments` sayfasına gidin.
2. `Deploy Model`'a tıklayın.
3. Kaynağınız olarak `Local Path`'ı seçin.
4. `Name` alanına bir ad girin (ör. `DeepSeek-R1`).
5. Her çalışan düğümde DeepSeek R1 model dosyalarını içeren dizin olarak `Model Path`'ı belirtin.
6. `Backend`'nin `vLLM` olarak ayarlandığından emin olun.
7. Uyumluluk kontrolünü geçtikten sonra dağıtmak için `Save` seçeneğine tıklayın.

![dağıtım modeli](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/deploy-model.png)

## Adım 5: Dağıtımı İzleyin { #step-5-monitor-deployment }

Dağıtım durumunu `Deployments` sayfasından izleyebilirsiniz. GPU ve çalışan kullanımını görüntülemek için `distributed across workers`'in üzerine gelin. Model yükleme ilerlemesini gösteren gerçek zamanlı günlükleri görmek için `View Logs`'ye tıklayın. Modelin yüklenmesi birkaç dakika sürebilir.

![model-bilgisi](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/model-info.png)

Model çalıştıktan sonra GPU kullanımını kontrol etmek için `Workers` sayfasına gidin. Varsayılan olarak vLLM, GPU belleğinin %90'ını kullanır. Bunu model yapılandırma ayarlarından ayarlayabilirsiniz.

![kaynaklar yüklü](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/resources-loaded.png)

## Adım 6: Çıkarımı Oyun Alanı Aracılığıyla Çalıştırın { #step-6-run-inference-via-playground }

Model dağıtıldıktan ve çalıştırıldıktan sonra GPUStack Playground'u kullanarak test edebilirsiniz.

1. `Playground` -> `Chat` seçeneğine gidin.
2. Yalnızca bir model dağıtılırsa varsayılan olarak seçilecektir. Aksi takdirde, `DeepSeek-R1` seçeneğini seçmek için açılır menüyü kullanın.
3. İstemleri girin ve modelle etkileşime geçin.

![oyun alanı-sohbet](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-chat.png)

Eşzamanlı çıkarım senaryolarını test etmek için `Compare` sekmesini de kullanabilirsiniz.

![oyun alanı karşılaştırması](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-compare.png)

Artık DeepSeek R1 671B'yi Dağıtılmış vLLM'yi kullanarak GPUStack kümesinde başarıyla dağıttınız ve çalıştırdınız. Modelin performansını ve yeteneklerini kendi uygulamalarınızda keşfedin.

Daha fazla yardım için GPUStack topluluğu veya destek ekibiyle iletişime geçmekten çekinmeyin.
