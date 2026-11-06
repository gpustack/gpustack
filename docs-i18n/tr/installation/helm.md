---
translation_status: translated
---

# Helm Üzerinden Kurulum { #installation-via-helm }

V2.2.0'dan bu yana GPUStack, resmi Helm chart'ı kullanılarak küme içi [Higress](https://higress.io/) API ağ geçidiyle Kubernetes'e dağıtılabilir.

!!! note

    Higress ile GPUStack'i Kubernetes'e dağıtmak şu anda **deneysel** olarak kabul ediliyor. Devam etmeden önce [sınırlamaları](#limitations) inceleyin.

## Önkoşullar { #prerequisites }

| Bileşen | Versiyon |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

Ayrıca:

- Sunucunun veri birimi için kümede varsayılan bir `StorageClass` yapılandırılmalıdır (k3s'de varsayılan `local-path`'dir). Alternatif olarak, PVC yerine ana bilgisayar yolu birimini kullanmak için `server.dataVolume.hostPath`'yi ayarlayın.
- GPU çalışanları için, uygun GPU sürücülerinin ve konteyner araç setlerinin düğümlere yüklendiğinden emin olun (bkz. [Yükleme Gereksinimleri](./requirements.md)) ve [Düğüm Özellik Keşfi (NFD)](https://kubernetes-sigs.github.io/node-feature-discovery/), GPU düğümleri etiketlenecek şekilde yüklendiğinden emin olun.

## Sınırlamalar { #limitations }

- GPUStack sunucusu `StatefulSet` olarak dağıtılır ve şu anda birden fazla replikayı **desteklemez**.
- Varsayılan olarak, paket halindeki gömülü `PostgreSQL` veritabanı kullanılır. Üretim için `server.externalDatabaseURL` üzerinden harici bir veritabanının belirtilmesi tavsiye edilir.
- Higress eklentileri, GPUStack ile birlikte kurulan özel bir `gpustack/higress-plugins` Dağıtım tarafından sunulur. Higress ağ geçidi yeniden başlatıldığında bu hizmetten eklentileri indirir; hizmet kullanılamıyorsa, eklentiler erişilebilir hale gelene kadar ağ geçidinin başlatılması engellenir.
- Paketlenmiş `higress-core` alt chart'ı, Higress'i kümenin giriş denetleyicisi olarak dağıtır. Başka bir giriş denetleyicisi zaten çalışıyorsa, `higress-core.enabled=false`'i ayarlayın ve `gateway.ingressClassname`'yi mevcut bir Higress örneğine yönlendirin.

## k3s'yi yükleyin (isteğe bağlı) { #install-k3s-optional }

Aşağıdaki adımlarda örnek Kubernetes dağıtımı olarak k3s kullanılmıştır. Diğer CNCF uyumlu dağıtımlar (RKE2, kubeadm tabanlı kümeler veya yönetilen bulut Kubernetes), yukarıdaki sürüm gereksinimlerini karşıladıkları sürece de çalışır.

Giriş denetleyicisi olarak Higress kullanıldığından k3s'yi Traefik devre dışı bırakarak yükleyin. Yüksek kullanılabilirliğe sahip k3s kümeleri için [k3s belgelerine](https://docs.k3s.io/datastore/ha) bakın.

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

Kurulumu doğrulayın:

```bash
kubectl version
```

## Helm ile GPUStack'ı yükleyin { #install-gpustack-with-helm }

Chart, Docker Hub'da `oci://registry-1.docker.io/gpustack/gpustack-chart`'da OCI yapısı olarak yayınlanır. Chart sürümü GPUStack sürümünü izler (örn. chart `2.2.0`, GPUStack `v2.2.0`'yi gönderir).

En son kararlı sürümü yükleyin. `--version` atlandığında, Helm en yeni kararlı chart sürümünü otomatik olarak çözer ve geliştirme/yayın öncesi derlemeleri atlar:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

Belirli bir sürümü sabitlemek için `--version <chart-version>` ekleyin (chart sürümü GPUStack sürümüyle eşleşir, örneğin `2.2.0`):

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

Varsayılan olarak `higress-core` alt chart'ı etkinleştirilir ve GPUStack ile birlikte dağıtılır. Kümenizde zaten Higress yüklüyse, birlikte verilen Higress'i devre dışı bırakın ve GPUStack'i mevcut örneğinize yönlendirin:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

Parametreleri özelleştirmek için kurulum sırasında `--set key=value` veya `-f your-values.yaml` tuşlarını kullanın. Aşağıdaki [Chart Parametreleri](#chart-parameters)'e bakın.

### Klonlanmış Bir Depodan Kurulum { #installing-from-a-cloned-repository }

Alternatif olarak, depoyu kopyalayın ve yerel chart dizininden yükleyin (ağdan yalıtılmış ortamlar için veya chart'ı değiştirirken kullanışlıdır):

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### Higress'i Ayrı Olarak Kurmak { #installing-higress-separately }

`higress-core.enabled=false`'ı ayarlarsanız GPUStack'i dağıtmadan önce uyumlu bir Higress örneği yükleyin:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

IngressClass'ın mevcut olduğunu ve adının `gateway.ingressClassname` ile eşleştiğini doğrulayın:

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

Higress özelleştirmesi için [Higress belgelerine](https://higress.cn/en/docs/latest/ops/deploy-by-helm) bakın.

## GPUStack'e erişim { #accessing-gpustack }

Sunucu bölmesinin hazır olmasını bekleyin:

```bash
kubectl get pods -n gpustack-system -w
```

İlk yönetici şifresini alın:

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

`server.ingress.hostname` ayarlamadıysanız, girişten GPUStack UI adresini alın:

```bash
kubectl get ingress -n gpustack-system gpustack \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Adresi bir tarayıcıda açın ve `admin` kullanıcı adı ve yukarıda alınan şifreyle giriş yapın.

## Ortak Yapılandırma { #common-configuration }

### Worker DaemonSets'i Etkinleştirme { #enabling-worker-daemonsets }

Worker DaemonSet'ler varsayılan olarak devre dışıdır (`worker.enabled=false`). Etkinleştirildiğinde, chart bir CPU çalışan DaemonSet'i (`<release>-worker`) ve `worker.gpuVendors`'de listelenen GPU satıcısı başına bir DaemonSet'i (`<release>-worker-<vendor>` olarak adlandırılır) oluşturur.

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

Desteklenen `worker.gpuVendors` değerleri: `nvidia`, `mthreads`, `amd`, `ascend`, `hygon`, `metax`, `iluvatar`, `cambricon`, `thead`.

!!! note

    Her GPU DaemonSet, Node feature Discovery tarafından tanıtılan otomatik bir PCI varlığı `nodeSelector` etiketi (örneğin NVIDIA için `feature.node.kubernetes.io/pci-10de.present: "true"`) alır. Kümede **NFD kurulmalıdır**, aksi takdirde hiçbir düğüm gerekli etiketleri taşımaz ve tüm çalışan bölmeleri `Pending` kalır.

En az bir GPU satıcısı listelendiğinde (yani `worker.gpuVendors` boş değilse), her çalışan podu ayrıca gerekli bir `podAntiAffinity` (topologyKey=ana bilgisayar adı) alır, böylece iki çalışan bir düğümü paylaşamaz — bu, `hostNetwork: true` bağlantı noktalarını çarpışmaya karşı korur.

CPU işçisi DaemonSet, `nodeSelector` `feature.gpustack.ai/acceleratable: "false"` aracılığıyla GPU satıcısının talep etmediği düğümleri kapsar. Bu düğümleri yalnız bırakmak için `worker.cpuEnabled=false`'yi ayarlayın; genellikle kontrol düzlemi kümeyi GPU düğümleriyle paylaştığında ve işçi kazanmaması gerektiğinde:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

`worker.gpuVendors` daha sonra desteklenen en az bir satıcının adını vermelidir; aksi takdirde yayın hiçbir çalışan DaemonSet oluşturmaz ve kurulum reddedilir. GPU DaemonSet'leri her iki durumda da `-<vendor>` son ekini korur, dolayısıyla CPU'yu kapatmak asla bunlardan birinin adını `<release>-worker` olarak değiştirmez.

Alternatif olarak, kurulumdan sonra **Clusters** ve **Workers** sayfalarındaki kullanıcı arayüzü aracılığıyla GPU kümeleri ve çalışan düğümleri ekleyin.

### Harici Veritabanı Kullanma { #using-an-external-database }

GPUStack varsayılan olarak yerleşik PostgreSQL veritabanını kullanır. Harici bir PostgreSQL veya MySQL veritabanı kullanmak için `server.externalDatabaseURL` değerini ayarlayın:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### HTTPS'yi Özel Sertifikayla Etkinleştirme { #enabling-https-with-a-custom-certificate }

Sertifikayı ve anahtar içeriklerini (PEM) `server.ingress.tls` aracılığıyla sağlayın. Her ikisi de ayarlandığında giriş şeması HTTPS olur:

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### Özel Kayıttan İmaj Çekme { #pulling-images-from-a-private-registry }

Tüm imajları (GPUStack sunucusu/çalışanı, higress eklentileri ve paket halindeki higress-core ağ geçidi/denetleyici/pilot) yansıtılmış özel bir kayıt defterinden çekmek için `global.hub`'i geçersiz kılın. Bu, Helm'in global değerleri yayılımına dayanır, dolayısıyla her imajı kapsayan tek bir ayar vardır:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` aynı zamanda sunucuya `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` olarak aktarılarak çıkarım motoru imajlarının (örn. vLLM, llama.cpp) aynı kayıt defterinden çekilmesi sağlanır.

Çekme kimlik bilgilerini sağlamak için chart, `gpustack-image-pull-secret` adında bir `docker-registry` Sırrı oluşturabilir ve bunu tüm bölmelere bağlayabilir:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

Bunun yerine önceden var olan kendi Sırlarınıza referans vermek için `global.imagePullSecrets` ile değiştirin:

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## Chart Parametreleri { #chart-parameters }

En sık kullanılan parametreler aşağıda listelenmiştir. Tam ve yetkili liste için chart'ın [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) ve [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md)'lerine bakın.

| Parametre | Varsayılan | Açıklama |
| -------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `debug`                                | `false`                  | Hata ayıklama modunu etkinleştirin. |
| `registrationToken`                    | `null`                   | İşçi kayıt jetonu; `null` ise rastgele bir tane oluşturulur ve yeniden kullanılır. |
| `clusterDomain`                        | `cluster.local`          | Kubernetes küme hizmeti etki alanı son eki. |
| `global.hub`                           | `docker.io`              | Konteyner kayıt defteri ana bilgisayarı; özel bir kayıt defteri için geçersiz kıl. |
| `global.imagePullSecrets`              | `[gpustack-image-pull-secret]` | Görüntü çekme Sırları tüm bölmelere eklenir ve alt chart'lara yayılır. |
| `global.nodeSelector`                  | `{}`                     | Her bileşen için varsayılan nodeSelector; bileşen düzeyindeki değerle değiştirilir. |
| `image.repository`                     | `gpustack/gpustack`      | Ad alanına sahip imaj deposu; son referans `{global.hub}/{repository}:{tag}`. |
| `image.tag`                            | `null`                   | İmaj etiketi; varsayılan olarak chart'ın `appVersion`'sine ayarlanır. |
| `image.pullPolicy`                     | `IfNotPresent`           | İmaj çekme politikası. |
| `imagePullSecret.credentials.registry` | `docker.io`              | Chart, imaj çekme Secret kaynağını oluşturduğunda kullanılan kayıt defteri ana bilgisayarı. |
| `imagePullSecret.credentials.username` | `null`                   | Kayıt defteri kullanıcı adı; Parolayla ayarlandığında Sırrı oluşturur. |
| `imagePullSecret.credentials.password` | `null`                   | Kayıt defteri şifresi; Kullanıcı adı ile ayarlandığında Sırrı oluşturur. |
| `server.ingress.hostname`              | `null`                   | Sunucu için giriş ana bilgisayar adı. |
| `server.ingress.tls.cert`              | `null`                   | Ingress TLS sertifikası (PEM); özel anahtarla birlikte ayarlandığında HTTPS'yi etkinleştirir. |
| `server.ingress.tls.key`               | `null`                   | Ingress TLS özel anahtarı (PEM). |
| `server.externalDatabaseURL`           | `null`                   | Harici veritabanı bağlantı dizesi (PostgreSQL veya MySQL). |
| `server.dataVolume.hostPath`           | `null`                   | Sunucu veri birimi için ana bilgisayar yolu; PVC yerine hostPath kullanır. |
| `server.dataVolume.size`               | `100Gi`                  | Sunucu veri birimi boyutu (PVC). |
| `server.apiPort`                       | `30080`                  | API hizmet bağlantı noktası. |
| `server.metricsPort`                   | `10161`                  | Sunucu ölçümleri bağlantı noktası. |
| `server.environmentConfig`             | `{}`                     | GPUStack sunucusu için ekstra ortam değişkenleri. |
| `server.nodeSelector`                  | `{}`                     | Sunucu bölmesi nodeSelector; boş olmadığında `global.nodeSelector`'in yerine geçer. |
| `gateway.ingressClassname`             | `higress`                | Higress Giriş Sınıfı adı; bulunduğunda küme içi ağ geçidi modunu etkinleştirir. |
| `higress-core.enabled`                 | `true`                   | Birlikte verilen Higress ağ geçidini dağıtın; Zaten yüklüyse devre dışı bırakın. |
| `worker.enabled`                       | `false`                  | Worker DaemonSet manifestlerini oluşturur. |
| `worker.gpuVendors`                    | `[nvidia]`               | GPU satıcıları; Satıcı başına bir DaemonSet artı CPU DaemonSet. |
| `worker.cpuEnabled`                    | `true`                   | CPU worker DaemonSet manifestini oluşturur; `false` bir GPU satıcısı gerektirir. |
| `worker.nodeSelector`                  | `{}`                     | Temel çalışan nodeSelector; boş olmadığında `global.nodeSelector`'in yerine geçer. |
| `worker.port`                          | `10150`                  | Worker hizmetinin bağlantı noktası. |
| `worker.metricsPort`                   | `10151`                  | Çalışan ölçümleri bağlantı noktası. |
| `worker.dataDir`                       | `/var/lib/gpustack`      | Ana makine yolu, her çalışan bölmesinin içinde `/var/lib/gpustack`'ye monte edilmiştir. |
| `worker.environmentConfig`             | `{}`                     | GPUStack çalışanı için ekstra ortam değişkenleri. |

## Kaldırma { #uninstallation }

Bu Helm sürümü sunucudan fazlasını kurar: `gpustack-operator` alt chart'ı Kueue ve Node Feature Discovery denetleyicilerini getirir; operatör de finalizer içeren özel kaynaklar oluşturur. `helm uninstall` denetleyicileri de kaldırdığı için özel kaynaklar **önce**, denetleyiciler hâlâ çalışırken silinmelidir. Aksi hâlde nesneler sonsuza kadar `Terminating` durumunda kalır ve `gpustack-system` ad alanı da bu durumdan çıkamaz. Şu sırayı izleyin:

1. **her** denetleyici hala çalışırken iş yüklerini boşaltın. Operatör, `Instance` nesnelerindeki sonlandırıcıları temizliyor ve Kueue, kabul edilen `Workload`'lerin hala onu işgal etmesine rağmen `ClusterQueue`'i serbest bırakmayı reddediyor - bu nedenle herhangi bir şey kapatılmadan önce bunların gitmesi gerekiyor. `Instance`'ler ad alanına sahiptir ve kiracıların ad alanlarında yaşar, bu nedenle hepsini `-A` ile numaralandırın. Modelleri sunucudan (veya örneklerini doğrudan) silin; her bulut sunucusunun kapsülü onunla sona erer ve İş Yükü Bulut Sunucusu ile birlikte gider. Ad alanının bölmelerini toptan silmeyin; bu, operatörü, Kueue ve NFD denetleyicilerini de beraberinde getirir ve temizliği bitiren onlar olur:

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    Kueue diğer uygulamalarla paylaşılıyorsa burada veya 3. adımda `--all` kullanmayın; yalnızca GPUStack nesnelerini silin. Operatörün Kueue'de oluşturduğu her şey `gpustack` önekiyle (`clusterqueue/gpustack--…`, `admissioncheck/gpustack-node-devices`) adlandırılır. Küme kapsamlı türler yalnızca ada göre seçilir; ad alanı olanların ad alanının taşınmasına ihtiyacı vardır, çünkü `-o name` bunu atlar:

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. Operatörün kendi denetleyicilerini durdurun, ancak sürümü kurulu bırakın, böylece Kueue denetleyicileri çalışmaya devam eder; bir sonraki adımda silme işlemlerini sonlandıranlar onlardır. `gpustack-operator-worker` ve `gpustack-operator-device-manager-*` DaemonSets çalıştığı sürece, `LocalQueue`'leri, `ClusterQueue`'leri ve `gpustack-node-devices` AdmissionCheck'i sildiğiniz anda yeniden oluştururlar, böylece sayılar asla sıfıra ulaşmaz:

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. Kueue kaynaklarını bağımlılık sırasına göre silin — `LocalQueue`s, sonra `ClusterQueue`s, sonra `AdmissionCheck`s, sonra `ResourceFlavor`s. Kueue, kullanıcıları hala mevcut olan bir nesne üzerinde `kueue.x-k8s.io/resource-in-use` sonlandırıcıyı tutar: `gpustack-node-devices` AdmissionCheck, her ClusterQueue bitene kadar ve her ClusterQueue, LocalQueue'ları ve kabul edilen İş Yükleri bitene kadar silinmiş durumda kalır. Bir sonrakine geçmeden önce her türün boş olmasını bekleyin. Paylaşılan bir kümede, 1. adımdaki `--all` yerine her komuta `gpustack` önek kapsamını uygulayın:

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. Sürümü kaldırın. `--wait` parametresini kullanmayın: Node Feature Discovery'nin silme sonrası temizleme hook'u, silme işleminin tamamlanmasını `--timeout` süresinden daha uzun süre bekletir. Bundan çıkmak için Helm'i zorla sonlandırmak, temizlemeye ait RBAC kaynaklarını sahipsiz bırakır (5. adımda bunlar elle kaldırılır; `--wait` kullanmasanız da 5. adımı uygulayın):

    ```bash
    helm uninstall gpustack -n gpustack-system
    ```

5. `helm uninstall` komutunun kaldırmadığı kaynakları kaldırın. Helm, CRD'leri hiçbir zaman silmez; operatör de hiçbir sürümün sahip olmadığı admission webhook'ları ve APIServices kaydeder. Operatörün Service'i artık bulunmadığından her ikisi de hata durumunda istekleri reddeder (fail closed); webhook'lar silinene kadar aşağıdaki finalizer kaldırma işlemini de reddeder:

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    `worker.gpustack.ai` grubu her zaman GPUStack'e aittir — başka hiçbir bileşen onu kullanmaz — dolayısıyla CRD'lerinin koşulsuz olarak silinmesi güvenlidir.

6. Sürümün geride bıraktığı CRD'leri silin — **yalnızca bu sürümün yüklediği CRD'leri**. Bir CRD'nin silinmesi tüm kümedeki bu türden her nesneyi siler; dolayısıyla küme GPUStack'ten önce zaten Kueue, Node Özellik Keşfi veya Higress çalıştırıyorsa (yukarıdaki `higress-core.enabled`'e bakın) ve başka herhangi bir şey hala bunları kullanıyorsa, CRD'lerini yerinde bırakın ve bu gruplar için bu adımı atlayın. Tahmin etmek yerine bu sürümün neler içerdiğini keşfedin:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") == "gpustack")
      | .metadata.name'
    ```

    Kueue'nin CRD'lerini ve chart tarafından kurulmuşsa NFD'leri ve Higress'leri listeleyen bir stok kurulumda:

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    Küme sağlayıcısının kendi CRD'leri hiçbir zaman GPUStack'in silinecekleri değildir — örneğin bir k3s kümesindeki `*.k3s.cattle.io` ve `helm.cattle.io` kümeye aittir.

7. *(İsteğe bağlı)* Higress nesnelerini, bootstrap ConfigMap'i, ServiceAccount'u ve Secret kayıt belirtecini kendisiyle birlikte alan ad alanını silin. Bununla birlikte ne olacağını bilin: Ad alanı, sunucunun kalıcı verilerini (kullanıcıları, modelleri ve kümeleriyle birlikte yerleşik PostgreSQL veritabanı) taşıyan sunucunun PVC'sini (`gpustack-data-dir-gpustack-server-0`, `<volumeClaimTemplate>-<statefulset>-<ordinal>`) tutar. `Delete` geri alma politikasında (`local-path` varsayılan) temeldeki birim de silinir. Verilerin yeniden yüklenmesini istiyorsanız ad alanını koruyun; aksi halde ad alanını silin (ve geri alma politikası zaten kaldırmıyorsa önce PVC'yi):

    ```bash
    kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
