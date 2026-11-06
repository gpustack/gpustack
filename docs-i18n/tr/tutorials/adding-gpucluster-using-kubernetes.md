---
translation_status: translated
---

# Kubernetes Kullanarak GPU Kümesi Ekleme { #adding-a-gpu-cluster-using-kubernetes }

GPUStack, Kubernetes kümesinin GPU kümesi olarak eklenmesini destekler.

## Hazırlık { #preparation }

Tüm düğümlerde çalışma zamanı sınıfının kurulu ve yapılandırılmış olduğu bir Kubernetes kümesi oluşturulmalıdır. Bu derste örnek olarak k3s kümesini kullanacağız.

## Kubernetes Kümesi Oluşturun { #create-kubernetes-cluster }

Lütfen [Kubernetes Kümesini Kaydetme](../user-guide/cluster-management.md#register-kubernetes-cluster) bölümündeki adımlara bakın.

Kubernetes kümesi oluştururken aşağıdaki seçenekleri yapılandırabilirsiniz:

![kubernetes-küme-seçenekleri](../assets/tutorials/adding-gpucluster-using-kubernetes/k8s-options.png)

Kümeyi oluşturmak için `Save`'e tıklayın.

`Kubernetes` sağlayıcı kümesi oluşturulduktan sonra ihtiyacınız olan `GPU Vendors`'i seçin. Birden fazla satıcı seçebilir veya hiçbirini seçebilirsiniz.

![nvidia-gpu-satıcısı](../assets/tutorials/adding-gpucluster-using-kubernetes/nvidia-gpu-vendor.png)

Kümenin kaydolmaya hazır olduğunu doğrulayan ortam kontrol komut dosyasını almak için `Next` öğesine tıklayın. Komut dosyasını k3s'in kurulu olduğu ana bilgisayarda çalıştırın.

![ortamı kontrol edin](../assets/tutorials/adding-gpucluster-using-kubernetes/check-environment.png)

Lütfen k8s'deki `runtimeclass` kaynağının yalnızca bir veri kaydı olduğunu ve her düğümde iyi yapılandırılmış konteyner çalışma zamanını temsil etmediğini unutmayın. Belirtilen GPU satıcısı için kapsayıcı çalışma zamanının yapılandırıldığından emin olun, aksi takdirde çalışan başlatılamaz.

`Run Command` adımındaki scripti kopyalayın ve k3s kurulu hostta çalıştırın.

![run-komutu](../assets/tutorials/adding-gpucluster-using-kubernetes/run-command.png)

![uygulama-manifestleri](../assets/tutorials/adding-gpucluster-using-kubernetes/apply-manifests.png)

## İşçilerin Tedarik Edilmesi Bekleniyor { #waiting-for-workers-to-be-provisioned }

Manifestleri Kubernetes kümesine uyguladıktan sonra, `Workers` Sayfasında çalışan düğümlerin doldurulmasını bekleyebiliriz.

![tedarik edilen işçiler](../assets/tutorials/adding-gpucluster-using-kubernetes/provisioned-workers.png)

Worker `Ready` durumuna ulaştığında modelleri onun üzerine dağıtabilirsiniz.
