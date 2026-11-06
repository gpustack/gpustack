---
translation_status: translated
---

# Bulut Kimlik Bilgisi Yönetimi { #cloud-credential-management }

GPUStack, bulut kimlik bilgisi yönetimini destekleyerek harici bulut sağlayıcılarına güvenli bağlantılara olanak tanır. Bulut kimlik bilgileri, API erişimi için gereken sağlayıcı bilgilerini, anahtarları ve seçenekleri içerir.

## Desteklenen Sağlayıcılar { #supported-providers }

`DigitalOcean` ve `SHUIHUA FUTURE` desteklenmektedir.

## Bulut Kimlik Bilgisi Oluşturun { #create-cloud-credential }

1. `Cloud Credentials` sayfasına gidin.
2. `Add Cloud Credential` açılır menüsüne tıklayın ve bir sağlayıcı seçin.
3. Aşağıdaki bilgileri girin:

   - `Name`: Benzersiz kimlik bilgisi adı.
   - Her ikisinin de ihtiyaç duyduğu tek kimlik bilgisi alanı olan sağlayıcının sırrı:
     - `DigitalOcean` — `Access Token`, DigitalOcean `Applications & API` sayfasında oluşturulan API belirteci. Belirteç kapsamı `Full Access` olmalıdır.
     - `SHUIHUA FUTURE` — `API Key`, [Shuihua konsolunda](https://hub.do.top/cn/signin) oluşturuldu ve önüne `amp_live_` eklendi.
   - `Description`: Bulut kimlik bilgileri için ek bilgiler.

4. `Save` düğmesine tıklayın.

## Bulut Kimlik Bilgilerini Güncelleyin { #update-cloud-credential }

1. `Cloud Credentials` sayfasına gidin.
2. Düzenlemek istediğiniz kimlik bilgisini bulun.
3. `Edit` düğmesine tıklayın.
4. `Name`, `Access Token` ve `Description`'yi gerektiği gibi güncelleyin.
5. `Save` düğmesine tıklayın.

## Bulut Kimlik Bilgilerini Sil { #delete-cloud-credential }

1. `Cloud Credentials` sayfasına gidin.
2. Silmek istediğiniz kimlik bilgisini bulun.
3. İşlemler sütunundaki üç nokta düğmesine tıklayın, ardından `Delete`'ı seçin.
4. Silme işlemini onaylayın.
