---
translation_status: translated
---

# Helm aracılığıyla Yükseltme { #upgrade-via-helm }

Kubernetes tabanlı GPUStack kurulumunu `helm upgrade` ile yükseltebilirsiniz.

Aşağıdaki yükseltme talimatları yalnızca Helm aracılığıyla dağıtılabilen ilk sürüm olan GPUStack v2.2.0 ve sonraki sürümleri için geçerlidir.

!!! note

    Higress ile Kubernetes'te GPUStack'i dağıtmak ve yükseltmek şu anda **deneysel** olarak kabul ediliyor.

!!! warning

    **Önce Yedekleme:** Yükseltme işlemine devam etmeden önce veritabanınızı yedeklemeniz önemle tavsiye edilir.

    Paketlenmiş yerleşik PostgreSQL veritabanını kullanırsanız verileri, `/var/lib/gpustack`'deki sunucu bölmesinin içine monte edilen sunucunun PVC'sinde (`gpustack-data-dir-<release-name>-server-0` olarak adlandırılır, örneğin varsayılan `gpustack` sürümü için `gpustack-data-dir-gpustack-server-0` olarak adlandırılır) bulunur; gömülü PostgreSQL veri dizini `postgresql/data`'deki bu bağın altında bulunur. Yükseltmeden önce yedekleyin.

    Harici bir veritabanı (`server.externalDatabaseURL`) kullanıyorsanız, bunun yerine veritabanı sağlayıcınızın yedekleme prosedürünü izleyin.

## Sürümü Yükseltme { #upgrade-the-release }

Yükseltmenin bilinen, tekrarlanabilir bir sürüme ulaşması için hedef sürümü her zaman `--version <chart-version>` ile sabitleyin (chart sürümü GPUStack sürümüyle eşleşir, örneğin `2.2.0`). Başlangıçta kullandığınız aynı `-f values.yaml` (ve/veya `--set`) işaretlerini ileterek yükleme sırasında sağladığınız değerleri yeniden kullanın veya daha önce uygulanan değerleri korumak için `--reuse-values`:

```bash
helm upgrade gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system \
  --version <chart-version> \
  -f values.yaml
```

!!! warning

    `--version`'i atlarsanız Helm, en yeni kararlı chart sürümünü otomatik olarak çözer. Yükseltmeler için bundan kaçının: Hangi sürüme ulaşacağınızı kontrol edemezsiniz. Belirli bir sürümü hedeflemek için her zaman `--version` belirtin.

## Sunucu ve Worker'ları Güncelleme Sırası { #server-and-worker-ordering }

GPUStack sunucusu `StatefulSet` olarak ve chart yönetimli çalışanlar `DaemonSet` olarak dağıtılır, böylece her ikisi de sürüm yükseltmesinin bir parçası olarak yeni imaja aktarılır. Sunucu ve çalışanlar **aynı sürüm**'e ait olduğundan, "önce sunucu, sonra çalışanlar" sırası tek bir `helm upgrade` tarafından uygulanamaz. Ortamınızda katı sıralama önemliyse, çalışan bölmelerinin hareket etmesine izin vermeden önce sunucu bölmesinin hazır olmasını bekleyin:

```bash
kubectl rollout status statefulset/<release-name>-server -n gpustack-system
```

Chart dışında GPUStack kümeleri aracılığıyla kaydedilen çalışanlar (örneğin, kullanıcı arayüzü aracılığıyla eklenen Docker veya Kubernetes kümeleri) bu Helm sürümü tarafından yönetilmez ve [Küme Dağıtımını Yükseltme](cluster.md) adımları izlenerek yükseltilmeleri gerekir.
