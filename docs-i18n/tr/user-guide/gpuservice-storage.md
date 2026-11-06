---
translation_status: translated
---

# GPU Hizmet Depolama Alanı { #gpu-service-storage }

[GPU Hizmet Depolama Türleri](gpuservice-storage-types.md) temel alınarak, GPU Hizmet Depolama, kalıcı depolamayı ayırmanıza ve onu birden fazla GPU Hizmet Örneğine bağlamanıza olanak tanır.

Bu, örnekler arasında paylaşılması gereken veriler, modeller ve diğer dosyalar için kullanışlıdır.

GPUStack, birden fazla Kubernetes kümesini yönetir ve bunlar arasında GPU Hizmet Depolamasını paylaşmak için birleşik bir arayüz sağlar.

Kaputun altında, her depolama öğesi hedef kümede bir Kubernetes `PersistentVolumeClaim` olarak gerçekleştirilir.

## Depolama Alanına Göz Atın { #browse-storage }

Tüm depolama alanına ve ayrıntılarına göz atmak için `GPU Service` > `Storage` sayfasına gidin.

![Ekran görüntüsü: Depolama listesi](../assets/gpuservice/storage/list.png)

Depolamayı görünen ada veya ada göre filtreleyebilirsiniz; `Name` sütununun gösterebileceği değerlerden herhangi biri. `Name` sütun sıralarını görüntülediği etikete göre sıralama.

## Depolama Ekleme { #adding-storage }

`Storage` sayfasında, oluşturma formunu açmak için `Add Storage`'e tıklayın.

![Ekran görüntüsü: Depolama formu ekle](../assets/gpuservice/storage/add.png)

Bir `Name` girin, bir `Storage Type` seçin ([GPU Hizmet Depolama Türleri](gpuservice-storage-types.md) sayfasında tanımlanır) ve `Capacity (GB)`'i ayarlayın. Ardından `Save`'e tıklayın.

## Depolamayı Düzenleme { #editing-storage }

Oluşturulduktan sonra yalnızca görünen ad değiştirilebilir.

!!! note

    Gelecekteki bir sürüm için depolama yapılandırmasının daha fazla düzenlenmesi planlanmaktadır.

## Depolamayı Silme { #deleting-storage }

Bir depolama öğesi üzerinde `Delete` öğesine tıklayın ve onaylayın. Depolama daha sonra listeden kaldırılır.

!!! warning

    Depolama hala bir GPU Hizmet Örneğine bağlıysa, küme tarafındaki `PersistentVolumeClaim` hemen kaldırılmaz. Silme işlemi ertelenmiş bir politikayı takip ettiğinden, `PersistentVolumeClaim` ilgili örnek silinene kadar tutulur ve ancak o zaman temizlenir.
