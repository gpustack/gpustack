---
translation_status: translated
---

# Hızlı Başlangıç { #quickstart }

Bu kılavuz, GPUStack'i kendi kendine barındırılan GPU sunucularında çalıştırma konusunda size yol gösterir. [bulut GPU'larını](./tutorials/adding-gpucluster-using-digitalocean.md) kullanmak veya [mevcut bir Kubernetes kümesiyle](./tutorials/adding-gpucluster-using-kubernetes.md) entegre etmek için ilgili eğitimlere bakın.

!!! info "Önkoşullar"

    1. En az bir NVIDIA GPU'ya sahip bir düğüm. Diğer GPU türleri için lütfen bir çalışan eklerken GPUStack kullanıcı arayüzündeki yönergeleri kontrol edin veya daha fazla ayrıntı için [Kurulum belgelerine](./installation/requirements.md) bakın.
    2. NVIDIA sürücüsünün, [Docker](https://docs.docker.com/engine/install/) ve [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)'in çalışan düğümde yüklü olduğundan emin olun.
    3. **(İsteğe bağlı)** GPUStack sunucusunu barındırmak için bir CPU düğümü. GPUStack sunucusu GPU gerektirmez ve yalnızca CPU içeren bir makinede çalışabilir. [Docker](https://docs.docker.com/engine/install/) kurulu olmalıdır. Docker Desktop (Windows ve macOS için) de desteklenmektedir. Özel bir CPU düğümü yoksa GPUStack sunucusu, GPU çalışan düğümüyle aynı makineye kurulabilir.
    4. GPUStack çalışan düğümleri için yalnızca Linux desteklenir. Windows kullanıyorsanız WSL2 kullanmayı düşünün ve Docker Desktop'ı kullanmaktan kaçının. macOS, GPUStack çalışan düğümleri için desteklenmez.

## GPUStack'i yükleyin { #install-gpustack }

[Docker](https://docs.docker.com/engine/install/) kullanarak GPUStack sunucusunu kurmak ve başlatmak için aşağıdaki komutu çalıştırın:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

??? Note "Alternatif: Quay Container Registry Mirror'ı kullanın"

    Docker Hub'dan görüntü alamıyorsanız veya indirme işlemi çok yavaşsa kayıt defterinizi `quay.io`'e yönlendirerek Quay Container Registry aynamızı kullanabilirsiniz:

    ```bash
    sudo docker run -d --name gpustack \
        --restart unless-stopped \
        -p 80:80 \
        --volume gpustack-data:/var/lib/gpustack \
        quay.io/gpustack/gpustack \
        --system-default-container-registry quay.io
    ```

GPUStack başlangıç günlüklerini kontrol edin:

```bash
sudo docker logs -f gpustack
```

GPUStack başladıktan sonra varsayılan yönetici şifresini almak için aşağıdaki komutu çalıştırın:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Tarayıcınızı açın ve GPUStack kullanıcı arayüzüne erişmek için `http://your_host_ip` seçeneğine gidin. Giriş yapmak için varsayılan kullanıcı adı olan `admin`'i ve yukarıda aldığınız şifreyi kullanın.

## GPU Kümesi Kurma { #set-up-a-gpu-cluster }

1. GPUStack kullanıcı arayüzünde `Clusters` sayfasına gidin.

2. `Add Cluster` düğmesine tıklayın.

3. Küme sağlayıcısı olarak `Docker`'ı seçin.

4. Yeni küme için `Name` ve `Description` alanlarını doldurun ve ardından `Save` düğmesine tıklayın.

5. Yeni çalışan düğümü yapılandırmak için kullanıcı arayüzü yönergelerini izleyin. Çalışan düğümü GPUStack sunucusuna bağlamak için bir Docker komutu çalıştırmanız gerekecektir. Komut aşağıdakine benzer görünecektir:

```bash
sudo docker run -d --name gpustack-worker \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --runtime nvidia \
      gpustack/gpustack \
      --server-url http://your_gpustack_server_url \
      --token your_worker_token \
      --advertise-address 192.168.1.2
```

6. Çalışan düğümü GPUStack sunucusuna bağlamak için komutu çalıştırın.

7. Çalışan düğüm başarıyla bağlandıktan sonra GPUStack kullanıcı arayüzündeki `Workers` sayfasında görünecektir.

## Bir Model Dağıtın { #deploy-a-model }

1. GPUStack kullanıcı arayüzünde `Catalog` sayfasına gidin.

2. Mevcut modeller listesinden `Qwen3.5-0.8B` modelini seçin.

3. Dağıtım uyumluluğu kontrolleri başarılı olduktan sonra modeli dağıtmak için `Save` düğmesine tıklayın.

![katalogdan qwen3'ü dağıtın](assets/quick-start/quick-start-qwen3.png)

4. GPUStack model dosyalarını indirmeye ve modeli dağıtmaya başlayacaktır. Dağıtım durumu `Running` gösterdiğinde model başarıyla dağıtılmıştır.

!!! note

    GPUStack, modelleri çalıştırmak için konteynerleri kullanır. İlk kez model dağıtımında, model dosyalarının ve kapsayıcı görüntülerinin indirilmesi biraz zaman alabilir. Dağıtım ilerlemesini izlemek için kullanıcı arayüzünde `View Logs` seçeneğine tıklayabilirsiniz.

![model çalışıyor](assets/quick-start/model-running.png)

5. Gezinme menüsünde `Playground - Chat` seçeneğine tıklayın, sağ üstteki `Model` açılır menüsünden `qwen3.5-0.8b` modelinin seçildiğini kontrol edin. Artık kullanıcı arayüzü oyun alanında modelle sohbet edebilirsiniz.

![hızlı sohbet](assets/quick-start/quick-chat.png)

## Modeli API aracılığıyla kullanın { #use-the-model-via-api }

1. `Access Control` > `API Keys` sayfasına gidin, ardından `New API Key` düğmesine tıklayın.

2. `Name` alanını doldurun ve `Save` düğmesine tıklayın.

3. Oluşturulan API anahtarını kopyalayın ve güvenli bir yere kaydedin. Oluşturulduğunda yalnızca bir kez görebileceğinizi lütfen unutmayın.

4. Artık GPUStack tarafından sağlanan OpenAI uyumlu API uç noktalarına erişmek için API anahtarını kullanabilirsiniz. Örneğin, curl'u aşağıdaki gibi kullanın:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Temizleme { #cleanup }

Dağıtılan modeli kullanmayı tamamladıktan sonra, GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidebilir ve kaynakları boşaltmak için modeli silebilirsiniz.
