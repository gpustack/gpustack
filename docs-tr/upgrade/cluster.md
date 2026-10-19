---
translation_status: translated
---

# Küme Dağıtımını Yükseltme { #upgrade-a-cluster-deployment }

GPUStack, çalışan düğümlerini [kümeler](../user-guide/cluster-management.md) halinde düzenler. Bu sayfa, GPUStack sunucusu yükseltildikten sonra bir kümenin çalışan dağıtımının yükseltilmesini kapsar.

!!! note

    1. Önce GPUStack **server**'i yükseltin (bkz. [Docker](docker.md) aracılığıyla yükseltme veya [Helm](helm.md) aracılığıyla yükseltme), ardından küme dağıtımlarını yükseltin.

    2. Bir çalışanı sunucuyla **aynı sürüm**'e yükseltin. `main`(dev) veya sürüm adayı(rc) çalışan imajını kararlı bir sunucuyla karıştırmayın.

Yükseltme prosedürü, kümenin başlangıçta nasıl dağıtıldığını yansıtır (bkz. [Küme Yönetimi](../user-guide/cluster-management.md)): Docker kümesi için dağıtımı yeniden çalıştırın veya yeni imajı kullanarak bir Kubernetes kümesi için bildirimleri yeniden uygulayın.

## Docker Kümesi { #docker-cluster }

Docker kümesindeki çalışanlar, kümenin `Add Worker` adımındaki `docker run` komutuyla başlatılan kapsayıcılar olarak çalışır. Yeni imajı çekerek, eski kabı kaldırarak ve onu daha önce olduğu gibi **aynı argümanlar ve hacimlerle** yeniden oluşturarak her bir çalışanı yükseltin:

```bash
docker pull gpustack/gpustack:vx.y.z   # match the server version

docker stop gpustack-worker
docker rm gpustack-worker

docker run -d --name gpustack-worker \
  ... \
  gpustack/gpustack:vx.y.z \
  --server-url http://your_gpustack_server_url \
  --token your_worker_token \
  --advertise-address your_worker_ip
```

!!! tip

    GPUStack kullanıcı arayüzünden hedef sürüm için güncel bir `docker run` komutu alabilirsiniz: `Clusters` sayfasında kümenin `Add Worker` adımını açın ve oluşturulan `Run command`'i kopyalayın.

## Kubernetes Kümesi { #kubernetes-cluster }

Kubernetes kümesindeki çalışanlar, küme kaydı sırasında uygulanan bildirimlerden GPUStack Operatörü tarafından yönetilen DaemonSet'ler olarak dağıtılır. Yükseltmek için yeni sürüm için oluşturulan bildirimleri yeniden uygulayın:

1. `Clusters` sayfasına gidin ve yükseltmek istediğiniz Kubernetes kümesini bulun.
2. İşlemler sütunundaki üç nokta düğmesine tıklayın, ardından `Register Cluster`'ı seçin.
3. `Run command` adımına kadar olan adımları takip edin ve oluşturulan komutu kopyalayın.
4. Komutu, hedef küme için `kubectl`'in kurulu olduğu ve `kubeconfig`'in yapılandırıldığı bir ortamda çalıştırın. Bu, güncellenen bildirimleri yeniden uygular ve Operatör, çalışan DaemonSets'i yeni sürüme aktarır.

Çalışan bölmeleri `Ready`'e ulaştığında küme yeni sürümü çalıştırır.

## Bulut Sağlayıcı Kümesi { #cloud-provider-cluster }

Bulut tarafından sağlanan kümeler için (örneğin, DigitalOcean), her çalışan düğümü, çalışan yapılandırmasını `/var/lib/gpustack/config.yaml`'e yazan ve `/opt/gpustack-run-worker.sh` aracılığıyla bir `gpustack-worker` kapsayıcısını başlatan cloud-init tarafından önyüklenir. Mevcut bir düğümü yükseltmek için SSH'yi kullanın ve aynı yapılandırmayı ve birimleri yeniden kullanarak kapsayıcıyı yeni imajla yeniden oluşturun:

```bash
# SSH into the worker node, then:
NEW_IMAGE=gpustack/gpustack:vx.y.z   # match the server version

docker pull "$NEW_IMAGE"

docker rm -f gpustack-worker

docker run -d --name gpustack-worker \
  -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$NEW_IMAGE" \
  --config-file=/var/lib/gpustack/config.yaml
```

!!! note

    **yeni sağlanan** düğümlerin (örneğin, bir çalışan havuzunun ölçeğinin büyütülmesinden) yeni sürümde de ortaya çıkması için, ayrıca kullanıcı arayüzündeki kümenin imaj yapılandırmasını da güncelleyin. Bkz. [Küme Yönetimi](../user-guide/cluster-management.md#operating-worker-pools).
