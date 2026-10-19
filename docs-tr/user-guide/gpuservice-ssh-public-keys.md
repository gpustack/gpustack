---
translation_status: translated
---

# GPU Hizmeti SSH Genel Anahtarları { #gpu-service-ssh-public-keys }

GPU Hizmeti SSH Genel Anahtarları, GPU Hizmet Örneklerine erişmek için kullanılan SSH genel anahtarlarını yönetmenizi sağlayarak güvenli, parolasız erişim sağlar.

Bu anahtarları `SSH Access` yapılandırması altındaki bir örneğe atarsınız; bkz. [GPU Hizmet Örnekleri](gpuservice-instances.md).

## SSH Genel Anahtarlarına Göz Atın { #browse-ssh-public-keys }

Tüm kayıtlı anahtarlara göz atmak için `GPU Service` > `SSH Public Keys` sayfasına gidin.

![Ekran görüntüsü: SSH Genel Anahtarları listesi](../assets/gpuservice/ssh-public-keys/list.png)

SSH genel anahtarlarını görünen ada veya ada göre filtreleyebilirsiniz; `Name` sütununun gösterebileceği değerlerden herhangi biri. `Name` sütun sıralarını görüntülediği etikete göre sıralama.

## SSH Genel Anahtarı Ekleme { #adding-an-ssh-public-key }

`SSH Public Keys` sayfasında, oluşturma formunu açmak için `Add SSH Public Key`'e tıklayın.

![Ekran görüntüsü: SSH Genel Anahtar formunu ekleyin](../assets/gpuservice/ssh-public-keys/add.png)

Bir `Name` girin, ardından anahtarınızı `SSH Public Key` alanına yapıştırın. Her anahtar `ssh-` ile başlamalıdır ve her satıra bir tane girerek birden fazla anahtarı kaydedebilirsiniz. Bitirmek için `Save`'e tıklayın.

!!! tip

    Genel anahtarınızı yazdırmak için `cat ~/.ssh/id_rsa.pub` (RSA) veya `cat ~/.ssh/id_ed25519.pub` (Ed25519) komutunu çalıştırın.

## SSH Genel Anahtarını Düzenleme { #editing-an-ssh-public-key }

Bir tuşun yapılandırmasını açmak için `Edit` tuşuna tıklayın, ayarları güncelleyin ve `Save` tuşuna tıklayın.

![Ekran görüntüsü: SSH Genel Anahtar formunu düzenleyin](../assets/gpuservice/ssh-public-keys/edit.png)

## SSH Genel Anahtarını Silme { #deleting-an-ssh-public-key }

Bir tuşa `Delete` tıklayın ve onaylayın. Daha sonra anahtar listeden kaldırılır.

!!! note

    SSH ortak anahtarı zaten bir GPU Hizmet Örneği tarafından kullanılıyorsa, burada silinmesi onu o örnekten kaldırmaz. İçeriği zaten çalışan tarafındaki kümeyle senkronize edilmiştir ve tek bir giriş oradan geriye dönük olarak çıkarılamaz. Çalışan bir örnekten anahtarı iptal etmek için söz konusu örneği açın, `Edit` öğesine tıklayın ve `SSH Access` yapılandırmasının altındaki anahtarın seçimini kaldırın.
