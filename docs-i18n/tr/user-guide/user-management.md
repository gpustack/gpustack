---
translation_status: translated
---

# Kullanıcı Yönetimi { #user-management }

GPUStack'in iki kullanıcı rolü vardır: `Admin` ve `User`.

Yöneticiler kümeleri, kaynakları, modelleri, kullanıcıları ve sistem ayarlarını yönetebilir.

Kullanıcılar kendi API anahtarlarını yönetebilir ve model API'lerine erişebilirler.

## Varsayılan Yönetici { #default-admin }

Bootstrap'te GPUStack varsayılan bir yönetici kullanıcı oluşturur.

Başlangıç şifresi `<data-dir>/initial_admin_password`'e kaydedilir.

Varsayılan kurulumda bu dosya sunucu kapsayıcısının içindeki `/var/lib/gpustack/initial_admin_password` konumunda bulunur.

`GPUStack`'i başlatırken `--bootstrap-password` işaretini kullanarak varsayılan yönetici için özel bir şifre belirleyebilirsiniz.

## Şifre Gereksinimleri { #password-requirements }

Kullanıcı arayüzü veya API aracılığıyla belirlenen bir şifre en az bir büyük harf, bir küçük harf, bir rakam ve aşağıdaki özel karakterlerden birini içermelidir:

```
! @ # $ % ^ & * _ + .
```

Kullanıcı arayüzü ayrıca şifreleri 6-64 karakterle sınırlandırır.

## Kullanıcı Oluştur { #create-user }

1. `Users` sayfasına gidin.
2. `Add User` düğmesine tıklayın.
3. Kullanıcı için `Name`, `Full Name`, `Password`'yi doldurun ve `Role`'i seçin.
4. `Save` düğmesine tıklayın.

## Kullanıcıyı Güncelle { #update-user }

1. `Users` sayfasına gidin.
2. Düzenlemek istediğiniz kullanıcıyı bulun.
3. `Operations` sütunundaki `Edit` düğmesine tıklayın.
4. Nitelikleri gerektiği gibi güncelleyin.
5. `Save` düğmesine tıklayın.

## Kullanıcıyı Devre Dışı Bırak { #deactivate-user }

1. `Users` sayfasına gidin.
2. Devre dışı bırakmak istediğiniz kullanıcıyı bulun.
3. `Operations` sütunundaki `Deactivate Account` düğmesine tıklayın.

## Kullanıcıyı Etkinleştir { #activate-user }

1. `Users` sayfasına gidin.
2. Etkinleştirmek istediğiniz kullanıcıyı bulun.
3. `Operations` sütunundaki `Activate Account` düğmesine tıklayın.

## Kullanıcıyı Sil { #delete-user }

1. `Users` sayfasına gidin.
2. Silmek istediğiniz kullanıcıyı bulun.
3. `Operations` sütunundaki üç nokta düğmesini tıklayın, ardından `Delete`'i seçin.
4. Silme işlemini onaylayın.

!!! note

    Yönetici kullanıcı kullanıcı arayüzünden devre dışı bırakılamaz veya silinemez.
