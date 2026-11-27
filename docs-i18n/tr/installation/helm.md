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
- Higress eklentileri, GPUStack ile birlikte kurulan özel bir Dağıtım (Deployment) tarafından sunulur; varsayılan olarak GPUStack sunucu imajı üzerinde çalıştığından eklenti modülleri sunucu ile sürüm uyumlu kalır. Higress ağ geçidi yeniden başlatıldığında bu hizmetten eklentileri indirir; hizmet kullanılamıyorsa, eklentiler erişilebilir hale gelene kadar ağ geçidinin başlatılması engellenir.
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
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

!!! note

    Sürüm adı bilinçli olarak `gpustack` yerine `gpustack-server` olarak seçilmiştir: Sunucunun kendi kümesini **Register Cluster** aracılığıyla çalışan olarak kaydettiğinizde, aynı `gpustack-system` ad alanına `gpustack` adı sabitlenmiş yalnızca çalışan (worker-only) bir Helm sürümü yüklenir. Sunucu sürümü için farklı bir ad bu çakışmayı önler; böylece aynı ad alanına küme kaydetme ve çalışanları sonradan yükseltme işlemlerinde sürüm adı çakışması riski olmaz. `gpustack` adıyla kurulmuş mevcut bir kurulum çalışmaya devam eder; `helm upgrade gpustack-server …` komutunu yalnızca yukarıdaki adla kurulum yaptıysanız kullanın.

    Bu, varsayılan `worker.enabled=false` değeri için geçerlidir. Sunucu sürümünde chart yönetimli çalışanları etkinleştirirseniz (`worker.enabled=true`), sunucu sürümü de sabit adlı nesneleri (`worker-config` ConfigMap'i, `worker` Service'i ve `registration-token` Secret'ı) işler ve aynı ad alanına kayıt, bootstrap ön kontrolü tarafından reddedilir (bu reddedilmeden önce kayıt manifesti, sunucu bölmesinin okuduğu `registration-token` Secret'ını zaten değiştirmiş olur). Bu durumda kümeyi farklı bir ad alanına kaydedin.

Belirli bir sürümü sabitlemek için `--version <chart-version>` ekleyin (chart sürümü GPUStack sürümüyle eşleşir, örneğin `2.2.0`):

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

Varsayılan olarak `higress-core` alt chart'ı etkinleştirilir ve GPUStack ile birlikte dağıtılır. Kümenizde zaten Higress yüklüyse, birlikte verilen Higress'i devre dışı bırakın ve GPUStack'i mevcut örneğinize yönlendirin:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
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
helm install gpustack-server ./gpustack-chart \
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

İlk yönetici şifresini alın. Sunucu bölmesinin adı `<release-name>-server-0` şeklindedir (varsayılan `gpustack-server` sürümü için `gpustack-server-server-0`):

```bash
kubectl exec -it -n gpustack-system gpustack-server-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

`server.ingress.hostname` ayarlamadıysanız, girişten GPUStack UI adresini alın. Giriş adı sürüm adıyla aynıdır (varsayılan olarak `gpustack-server`):

```bash
kubectl get ingress -n gpustack-system gpustack-server \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Adresi bir tarayıcıda açın ve `admin` kullanıcı adı ve yukarıda alınan şifreyle giriş yapın.

## Ortak Yapılandırma { #common-configuration }

### Worker DaemonSets'i Etkinleştirme { #enabling-worker-daemonsets }

Worker DaemonSet'ler varsayılan olarak devre dışıdır (`worker.enabled=false`). Etkinleştirildiğinde, chart bir CPU çalışan DaemonSet'i (`<release>-worker`) ve `worker.gpuVendors`'de listelenen GPU satıcısı başına bir DaemonSet'i (`<release>-worker-<vendor>` olarak adlandırılır) oluşturur.

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
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
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
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
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
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
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### Higress Ağ Geçidi Kaynak Boyutlandırma { #sizing-the-higress-gateway }

Higress ağ geçidi (Envoy), tüm istemci çıkarım trafiğini taşır. Bellek kullanımı, rota/yapılandırma değişim sıklığı ve eşzamanlı bağlantı sayısıyla büyür; LLM iş yükleri akış yanıtları nedeniyle bağlantıları uzun süre açık tutar, bu yüzden sürekli yük altında bellek tırmanmaya devam eder. Paketlenen `higress-core` alt chart'ı hem requests hem limits için varsayılan olarak `2` CPU / `2Gi` verir; bu, sürekli işletim için yeterli değildir. Bu nedenle GPUStack chart'ı ağ geçidinin **limits** değerlerini `4` CPU / `4096Mi`'ye yükseltir (requests, alt chart'ın varsayılanları olan `2` CPU / `2048Mi`'de kalır).

!!! warning

    OOMKilled olan bir ağ geçidi yalnızca çıkarım kesintisi değildir. Ağ geçidinden geçen işlemdeki tüm istekler düşer ve çalışanlar sunucuya ağ geçidi adresi üzerinden ulaşıyorsa, tüm çalışan bağlantıları da aynı anda kopar; küme çapında yeniden bağlanma tetiklenir ve model hizmetleri kullanılamaz hale gelebilir. Aşağıdaki [Çalışanları Sunucuya Doğrudan Bağlama](#connecting-workers-directly-to-the-server) bölümüne bakın.

Ayarlama kuralları:

- **Belleği limite ulaşmadan önce izleyin.** Birlikte verilen Prometheus ağ geçidinin Envoy istatistiklerini toplar; `envoy_server_memory_allocated` ve `envoy_server_memory_heap_size` için uyarı ayarlayın (Higress başka bir ad alanındaysa `gateway.namespace`'e bakın). `container_memory_working_set_bytes` bir kubelet/cAdvisor metriğidir ve yalnızca küme düzeyinde bir Prometheus'tan alınabilir. Bellek normal işletimde limite yaklaşmayı sürdürüyorsa `higress-core.gateway.resources.limits.memory` değerini yükseltin; ilk OOMKill'i beklemeyin.
- **Erişilebilirlik için en az 2 kopya tutun.** Ağ geçidi Deployment'ı varsayılan olarak `2` kopya ile gelir (`higress-core.gateway.replicas`); böylece dönen yeniden başlatma veya tek pod arızası tüm trafiği kesmez. Kopyalar, yapılandırma ve bağlantı yükü açısından pod başına belleği azaltmaz (her kopya rotaların tamamını tutar); ancak istek trafiğini böldüğü için her podda işlemdeki büyük istek sayısı azalır. `higress-core.gateway.service.externalTrafficPolicy` değerinin `Local` olduğuna dikkat edin: LoadBalancer/NodePort hizmetlerinde dış LB `healthCheckNodePort` adresine sağlık kontrolü yapmalıdır, aksi halde ağ geçidi pod'u olmayan düğümler trafiği düşürür.
- **OOM'a karşı araç bellek limitidir.** Alt chart, CPU kullanımına dayalı bir HPA ile gelir. İstek yükünü düzleştirir; ancak OOM'a karşı **koruma sağlamaz**, çünkü ağ geçidi belleği CPU'dan değil bağlantı sayısı ve istek boyutundan büyür. Otomatik ölçeklendirme açıkken bile bellek limitlerine mutlaka pay bırakın.
- **HPA `replicas` değerini geçersiz kılar.** `higress-core.gateway.autoscaling.enabled=true` olduğunda Deployment artık `replicas` değerini kullanmaz ve alt chartın `autoscaling.minReplicas` varsayılanı `1` olduğundan, yalnızca HPA'yı açmak ağ geçidinin tek pod'a küçülmesine izin verir. HPA açıkken `autoscaling.minReplicas` değerini en az `2` yapın.

```yaml
# values.yaml
higress-core:
  gateway:
    replicas: 2
    resources:
      requests:
        cpu: 2000m
        memory: 4096Mi
      limits:
        cpu: 4000m
        memory: 8192Mi   # bellek artmayı sürdürüyorsa daha da yükseltin
    autoscaling:
      enabled: true
      minReplicas: 2    # HPA `replicas` değerini yok sayar; bunu >= 2 tutun
      maxReplicas: 5
      targetCPUUtilizationPercentage: 80
```

#### Resim İstekleri İçin Boyutlandırma

Rota ve bağlantı yükü, bellek ayak izinin sabit kısmıdır; ani artışın nedeni istek boyutudur. İsteği inceleyen veya yeniden yazan ağ geçidi eklentileri isteğin gövdesinin tamamını arabelleğe alır; bu yüzden bellek işlemdeki istek baytlarıyla (eşzamanlılık × gövde boyutu) ölçeklenir. Buna en sık ulaşılan yol base64 kodlu resimler taşıyan isteklerdir: JSON içinde her resim dosya boyutunun 1,33 katı yer kaplar.

Aşağıdaki rakamlar, GPUStack'in varsayılan eklenti zinciri ve bir mock upstream ile tek bir ağ geçidi pod'unda (2 CPU / 4Gi) her durum için 60 saniyelik akış yükü ölçülerek, her durum yeni başlatılan bir pod'dan başlatılarak elde edilmiştir. Bunlar bir boyutlandırma formülü değil, büyüklük düzenine dair bir referanstır:

| Resim | İstek gövdesi | Eşzamanlı | İşlemde | Boşta çalışmaya göre bellek |
|---|---|---|---|---|
| 1 | 1,3 MB | 64 | 85 MB | yaklaşık 0,1 GiB |
| 8 | 10,7 MB | 4 | 43 MB | yaklaşık 0,2 GiB |
| 32 | 42,7 MB | 4 | 171 MB | yaklaşık 1,8 GiB |
| 32 | 42,7 MB | 16 | 683 MB | **OOMKilled** |

İstek başına maliyet gövdenin kendisinden daha hızlı artar ve yük bittikten sonra bellek hemen geri verilmez; bu yüzden bütçeyi istek sayısına değil işlemdeki baytlara göre yapın. 4Gi'lik bir pod için işlemdeki istek gövdelerini toplamda yaklaşık 200 MB'ın altında tutun ya da `higress-core.gateway.resources.limits.memory` değerini yükseltin (çok resimli trafik için 8Gi makul bir başlangıçtır); ek kopyalar da yardımcı olur çünkü istekleri böler.

Daha fazla Higress özelleştirmesi için [Higress Helm belgelerine](https://higress.cn/en/docs/latest/ops/deploy-by-helm) bakın.

### Çalışanları Sunucuya Doğrudan Bağlama { #connecting-workers-directly-to-the-server }

Çalışan ile sunucu arasındaki kayıt, kalp atışları ve model/örnek watch'ları (ve `tunnel` vekil modundaki kalıcı WebSocket), istemci çıkarım trafiğinden bağımsız kontrol düzlemi trafiğidir ve **Higress ağ geçidinden geçmemelidir** — aksi halde ağ geçidinin yeniden başlatılması veya OOMKill edilmesi tüm çalışan bağlantılarını aynı anda keser.

Chart yönetimli çalışanlar için bu doğru şekilde bağlanmıştır: `worker.enabled=true` ve `worker.serverURL` ayarlanmadığında, çalışanlar küme içi Service üzerinden (`http://<release>-server.<namespace>.svc:<server.apiPort>`) kayıt olur ve ağ geçidini tamamen atlar. Adresi kendiniz ayarlarken bu headless Service yerine ClusterIP Service'i (`<release>-server-cluster-ip`) tercih edin: headless ad, sunucu yük devrettiğinde bayatlayabilecek pod IP'lerine çözümlenir.

Kontrol edilmesi gereken yol, **Register Cluster** ile kaydedilen bir kümedir (veya `--server-url` ile başlatılan herhangi bir çalışan). Kayıt manifestosundaki `worker.serverURL` değerini sunucu üretir; sunucu adresi önce kümenin kendi **Server URL**'si, sonra `server_external_url`, sonra istek Host'u olacak şekilde çözer. Dolayısıyla kümede bir geçersiz kılma yoksa bu, sunucunun *dış* URL'sidir. Helm dağıtımında bu URL genellikle ingress adresi, yani Higress ağ geçididir. Bu düzende ağ geçidi OOM'unun etkisi kesilen isteklerin ötesine geçer:

- Tüm çalışanlar sunucuya olan bağlantılarını aynı anda kaybeder. `tunnel` modunda tüneller üzerinden aktarılan tüm işlemdeki çıkarım akışları düşer ve tünel istemcisi üstel geri çekilmeyle (1 sn'den 60 sn'ye kadar, jitter'lı) yeniden bağlanır. `direct`/`worker` modlarında tünel yoktur; ancak model ve örnek watch'ları uzun ömürlü HTTP akışlarıdır, onlar da düşer ve toplu yeniden bağlanma ile yeniden listeleme gerçekleşir. Kalp atışları ise periyodik çağrılardır; sadece başarısız olurlar ve bir sonraki aralıkta yeniden denerler.

- Kesinti, kalp atışı tolerans süresini (`GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`, varsayılan 150 sn) aşarsa çalışanlar çevrimdışı işaretlenir ve model rotaları geri çekilir — çıkarım süreçleri çalışmaya devam etse bile model hizmetleri kullanılamaz hale gelir.

Kontrol yolunu ağ geçidinden ayırmak için çalışanları sunucuya doğrudan ulaşan bir adrese yönlendirin:

- **Register Cluster ile kaydedilen kümeler:** kümenin **Server URL** ayarını doğrudan bir adrese yapın (çalışanlar sunucunun kümesindeyse küme içi Service, değilse sunucu Service'ine yönelik ayrılmış bir NodePort/LoadBalancer), ardından kayıt manifestosunu yeniden alıp uygulayın. Bunu küme üzerinde yapın; `helm upgrade --set worker.serverURL=…` ile yapmayın: manifest bir dahaki sefere uygulandığında bootstrap Job, worker sürümünü sunucunun render ettiği değerlerle yeniden kurar ve ağ geçidi URL'si geri gelir. Manifestin yönettiği sürümün yanına ikinci bir worker-only sürüm de eklemeyin; bu, yinelenen worker DaemonSet'lerine (pod'ları `hostNetwork` port anti-affinity nedeniyle Pending kalır) ve ikinci bir operatöre yol açar.

- **Kubernetes dışındaki çalışanlar:** bunları `--server-url` ile doğrudan adresi gösterecek şekilde başlatın.

- **Kayıt manifestosu olmadan Helm ile kurulan çalışanlar:** sunucunun kümesinde worker-only bir sürüm. Sunucu sürümüyle aynı ad alanında yaşar; bu nedenle o sürümün zaten sahip olduğu Secret'ları render etmemelidir: `imagePullSecret.create: false` ayarlayın ve `registrationTokenSecretName` ile sunucunun `registration-token` Secret'ını gösterin (`worker.enabled` varsayılan olarak `false`, `server` ve `higress-core` ise `true` olduğundan bu üçünü de açıkça belirtmek gerekir):

```yaml
# Sunucunun kümesindeki worker-only sürümü için values.yaml
server:
  enabled: false
higress-core:
  enabled: false
imagePullSecret:
  create: false
registrationTokenSecretName: registration-token
worker:
  enabled: true
  serverURL: http://gpustack-server-server-cluster-ip.gpustack-system.svc:30080
```

Böylece ağ geçidi yalnızca istemci çıkarım trafiğini taşır ve ağ geçidinin yeniden başlatılması artık çalışan kaydını, kalp atışlarını ve watch'ları etkilemez. Vekil modlarının kendisi için [Çalışan Bağlantı Modları](../user-guide/cluster-management.md#worker-connection-modes) bölümüne bakın.

### Özel Kayıttan İmaj Çekme { #pulling-images-from-a-private-registry }

Tüm imajları (GPUStack sunucusu/çalışanı, higress eklentileri ve paket halindeki higress-core ağ geçidi/denetleyici/pilot) yansıtılmış özel bir kayıt defterinden çekmek için `global.hub`'i geçersiz kılın. Bu, Helm'in global değerleri yayılımına dayanır, dolayısıyla her imajı kapsayan tek bir ayar vardır:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` aynı zamanda sunucuya `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` olarak aktarılarak çıkarım motoru imajlarının (örn. vLLM, llama.cpp) aynı kayıt defterinden çekilmesi sağlanır.

Çekme kimlik bilgilerini sağlamak için chart, `gpustack-image-pull-secret` adında bir `docker-registry` Sırrı oluşturabilir ve bunu tüm bölmelere bağlayabilir:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
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
| `higress-core.gateway.replicas`        | `2`                      | Higress ağ geçidi kopya sayısı; kullanılabilirlik için >= 2 tutun. |
| `higress-core.gateway.resources`       | limits `4` CPU / `4096Mi` | Ağ geçidi kaynakları; limits, alt chart varsayılanlarının üzerine çıkarılmıştır. Bkz. [Higress Ağ Geçidi Kaynak Boyutlandırma](#sizing-the-higress-gateway). |
| `higress-core.gateway.autoscaling.enabled` | `false`              | Ağ geçidi için CPU tabanlı HPA; OOM'u önlemez — bellek limits değerlerini yine de pay bırakarak ayarlayın. |
| `higress-core.gateway.autoscaling.minReplicas` | `1`                      | HPA açıkken en az kopya sayısı. HPA `replicas` değerini geçersiz kılar; en az 2 yapın. |
| `worker.serverURL`                     | `""`                     | Çalışanların kayıt olduğu sunucu adresi; boşsa küme içi sunucu Service'i kullanılır. Ağ geçidi üzerinden yönlendirmekten kaçının. |
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

4. Ad alanındaki **tüm** GPUStack sürümlerini kaldırın. Bu sayfanın önerdiği yapıda — sunucu `gpustack-server` adıyla varsayılan `worker.enabled=false` ile kurulur ve aynı küme **Register Cluster** aracılığıyla kaydedilerek `gpustack-system` içine `gpustack` adlı yalnızca çalışan bir sürüm yüklenir — operatör, Kueue ve Node Feature Discovery çalışan sürümüne aittir, sunucu sürümüne değil; bu nedenle yalnızca sunucu sürümünü kaldırmak, çalışan sürümünü ve denetleyicilerini çalışır durumda bırakır. `helm list -n gpustack-system` ile kontrol edin ve listelenen her GPUStack sürümünü kaldırın. Kümeyi farklı bir ad alanına kaydettiyseniz (örneğin sunucu sürümünde `worker.enabled=true` etkinken — yukarıdaki nota bakın), aynı kaldırma işlemini o ad alanında da yineleyin: oradaki yalnızca çalışan sürüm kendi operatörünü, Kueue'sunu ve Node Feature Discovery'sini taşır. `--wait` parametresini kullanmayın: Node Feature Discovery'nin silme sonrası temizleme hook'u, silme işleminin tamamlanmasını `--timeout` süresinden daha uzun süre bekletir. Bundan çıkmak için Helm'i zorla sonlandırmak, temizlemeye ait RBAC kaynaklarını sahipsiz bırakır (5. adımda bunlar elle kaldırılır; `--wait` kullanmasanız da 5. adımı uygulayın):

    ```bash
    helm list -n gpustack-system -q | grep -E '^gpustack(-server)?$' \
      | xargs -r -n 1 helm uninstall -n gpustack-system
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
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") | test("^gpustack(-server)?$"))
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

7. *(İsteğe bağlı)* Higress nesnelerini, bootstrap ConfigMap'i, ServiceAccount'u ve Secret kayıt belirtecini kendisiyle birlikte alan ad alanını silin. Bununla birlikte ne olacağını bilin: Ad alanı, sunucunun kalıcı verilerini (kullanıcıları, modelleri ve kümeleriyle birlikte yerleşik PostgreSQL veritabanı) taşıyan sunucunun PVC'sini (`gpustack-data-dir-<release-name>-server-0` olarak adlandırılır, örneğin varsayılan `gpustack-server` sürümü için `gpustack-data-dir-gpustack-server-server-0`) tutar. `Delete` geri alma politikasında (`local-path` varsayılan) temeldeki birim de silinir. Verilerin yeniden yüklenmesini istiyorsanız ad alanını koruyun; aksi halde ad alanını silin (ve geri alma politikası zaten kaldırmıyorsa önce PVC'yi):

    ```bash
    kubectl get pvc -n gpustack-system -o name | grep -E '^persistentvolumeclaim/gpustack-data-dir-.*-server-0$' \
      | xargs -r -n 1 kubectl delete -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
