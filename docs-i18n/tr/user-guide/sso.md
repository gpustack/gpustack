---
translation_status: translated
---

# Tek Oturum Açma (SSO) Kimlik Doğrulaması { #single-sign-on-sso-authentication }

GPUStack, OIDC, SAML ve CAS gibi Tek Oturum Açma (SSO) kimlik doğrulama yöntemlerini destekler. Bu, kullanıcıların harici bir kimlik sağlayıcısından gelen mevcut kimlik bilgilerini kullanarak oturum açmasına olanak tanır.

Aynı anda bir SSO yöntemi etkindir ve oturum açma sayfası, SSO düğmesini kullanımdaki protokolle etiketler, ör. `Log in with OIDC`. Bunun yerine kimlik sağlayıcıyı adlandırmak için; `Log in with Okta`, `--external-auth-provider-name`'yi ayarlayın.

## OIDC { #oidc }

OIDC'yi destekleyen herhangi bir kimlik doğrulama sağlayıcısı yapılandırılabilir. Varsa `email`, `name` ve `picture` talepleri kullanılır. İzin verilen yönlendirme URI'si `<server-url>/auth/oidc/callback` içermelidir.

OIDC sağlayıcınız özel veya kurumsal bir CA tarafından verilen bir sertifika kullanıyorsa, CA sertifikalarını GPUStack kapsayıcısına nasıl bağlayacağınızı öğrenmek için [Ek Güvenilir CA'lar](../installation/installation.md#additional-trusted-cas) konusuna bakın.

OIDC yapılandırması için aşağıdaki CLI bayrakları mevcuttur:

| <div style="width:180px">Bayrak</div> | Açıklama |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--oidc-issuer` | OIDC veren kuruluş URL'si. `<issuer>/.well-known/openid-configuration` altındaki OIDC keşfi, OIDC yapılandırmasını keşfetmek için kullanılacaktır.                                   |
| `--oidc-client-id` | OIDC istemci kimliği.                                                                                                                                                      |
| `--oidc-client-secret` | OIDC istemci sırrı.                                                                                                                                                  |
| `--oidc-redirect-uri` | OIDC uygulamanızda yapılandırılan yönlendirme URI'si. Bu `<server-url>/auth/oidc/callback` olarak ayarlanmalıdır.                                                         |
| `--external-auth-provider-name` (İsteğe bağlı) | Giriş sayfasının SSO düğmesinde gösterilen kimlik sağlayıcının adı, ör. `Okta`. Ayarlanmadığında varsayılan olarak `OIDC` olur.                                                     |
| `--external-auth-name` (İsteğe bağlı) | OIDC kullanıcı bilgilerinin kullanıcı adına eşlenmesi, örneğin, `preferred_username`. Varsayılan olarak, varsa `email` talebi kullanılır.                                        |
| `--external-auth-full-name` (İsteğe bağlı) | OIDC kullanıcı bilgilerinin kullanıcının tam adıyla eşleştirilmesi. Birden fazla öğe birleştirilebilir, örneğin `name` veya `firstName+lastName`. Varsayılan olarak `name` talebi kullanılır. |
| `--external-auth-avatar-url` (İsteğe bağlı) | OIDC kullanıcı bilgilerinin kullanıcının avatar URL'sine eşlenmesi. Varsayılan olarak, varsa `picture` talebi kullanılır.                                                         |
| `--external-auth-default-inactive` (İsteğe bağlı) | Yeni SSO kullanıcılarının varsayılan olarak etkinleştirilmesini engeller.                                                                                                              |
| `--external-auth-post-logout-redirect-key` (İsteğe bağlı) | Farklı IdP'ler arasında oturum kapatma sonrası yeniden yönlendirme için genel parametre adı (ör. Auth0 `returnTo`). Hem OIDC hem de SAML'ye uygulanır.                                    |

Bu seçenekleri CLI bayrakları yerine ortam değişkenleri aracılığıyla da ayarlayabilirsiniz:

```bash
GPUSTACK_OIDC_ISSUER="your-oidc-issuer-url"
GPUSTACK_OIDC_CLIENT_ID="your-client-id"
GPUSTACK_OIDC_CLIENT_SECRET="your-client-secret"
GPUSTACK_OIDC_REDIRECT_URI="{your-server-url}/auth/oidc/callback"
# Optional
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Okta"
GPUSTACK_EXTERNAL_AUTH_NAME="email"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo"  # e.g., for Auth0
```

### Örnek: Auth0 OIDC ile entegrasyon { #example-integrate-with-auth0-oidc }

GPUStack'i OIDC sağlayıcısı olarak Auth0 ile yapılandırmak için:

1. [auth0](https://auth0.com)'e gidin ve `Regular Web Applications` türüyle yeni bir uygulama oluşturun.

![create-oidc-app](../assets/sso/create-oidc-app.png)

2. Uygulama ayarlarından `Domain`, `Client ID` ve `Client Secret`'yi alın.

![auth0-app](../assets/sso/auth0-app.png)

3. İzin Verilen Geri Arama URL'lerine `<your-server-url>/auth/oidc/callback` ekleyin. URL'yi sunucunuzun URL'siyle eşleşecek şekilde uyarlayın.

4. İzin Verilen Oturum Kapatma URL'leri'ne `<your-server-url>/` (veya istediğiniz oturum kapatma sonrası URL'sini) ekleyin.

![auth0-geri arama](../assets/sso/auth0-callback.png)

Daha sonra GPUStack'i ilgili OIDC yapılandırmasıyla çalıştırın. Aşağıdaki örnekte CUDA ile Docker kullanılmaktadır:

```bash
sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
    --runtime nvidia \
    -e GPUSTACK_OIDC_ISSUER="https://<your-auth0-domain>" \
    -e GPUSTACK_OIDC_CLIENT_ID="<your-client-id>" \
    -e GPUSTACK_OIDC_CLIENT_SECRET="<your-client-secret>" \
    -e GPUSTACK_OIDC_REDIRECT_URI="<your-server-url>/auth/oidc/callback" \
    gpustack/gpustack
```

!!! note

    Özel CA sertifikası montajı yalnızca OIDC sağlayıcınız özel bir CA tarafından imzalandığında gereklidir. Auth0 gibi genel OIDC sağlayıcıları genellikle buna ihtiyaç duymaz.

## SAML { #saml }

GPUStack, Tek Oturum Açma (SSO) için SAML kimlik doğrulamasını destekler. Bu, kullanıcıların SAML'yi destekleyen harici bir kimlik sağlayıcıdan gelen mevcut kimlik bilgilerini kullanarak oturum açmasına olanak tanır.

SAML yapılandırması için aşağıdaki CLI bayrakları mevcuttur:

| <div style="width:180px">Bayrak</div> | Açıklama |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--saml-idp-server-url` | SAML Kimlik Sağlayıcı sunucu URL'si.                                                                                                                                                                                                                             |
| `--saml-idp-entity-id` | SAML Kimlik Sağlayıcı varlık kimliği.                                                                                                                                                                                                                              |
| `--saml-idp-x509-cert` | SAML Kimlik Sağlayıcı X.509 sertifikası.                                                                                                                                                                                                                      |
| `--saml-sp-entity-id` | SAML Servis Sağlayıcı varlık kimliği.                                                                                                                                                                                                                               |
| `--saml-sp-acs-url` | SAML Servis Sağlayıcı Onayı Tüketici Hizmeti URL'si. `<gpustack-server-url>/auth/saml/callback` olarak ayarlanmalıdır.                                                                                                                                          |
| `--saml-sp-x509-cert` | SAML Servis Sağlayıcı X.509 sertifikası.                                                                                                                                                                                                                       |
| `--saml-sp-private-key` | SAML Servis Sağlayıcı özel anahtarı.                                                                                                                                                                                                                             |
| `--saml-idp-logout-url` (İsteğe bağlı) | SAML Kimlik Sağlayıcı Tek Oturum Kapatma uç noktası URL'si.                                                                                                                                                                                                             |
| `--saml-sp-slo-url` (İsteğe bağlı) | SAML Servis Sağlayıcı Tek Oturum Kapatma Hizmeti geri arama URL'si (ör. `<server-url>/auth/saml/logout/callback`).                                                                                                                                                     |
| `--saml-sp-attribute-prefix` (İsteğe bağlı) | --external-auth-\* tarafından belirtilen nitelikleri getirmek için kullanılan SAML Servis Sağlayıcı özellik öneki. örneğin, 'http://schemas.auth0.com/'. |
| `--saml-security` (İsteğe bağlı) | JSON formatında SAML güvenlik ayarları.                                                                                                                                                                                                                         |
| `--external-auth-provider-name` (İsteğe bağlı) | Giriş sayfasının SSO düğmesinde gösterilen kimlik sağlayıcının adı, örneğin, `Okta`. Ayarlanmadığında varsayılan olarak `SAML` olur.                                                                                                                                               |
| `--external-auth-name` (İsteğe bağlı) | SAML kullanıcı bilgilerinin kullanıcı adına eşlenmesi. Tam özellik adını 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress' gibi yapılandırmalı veya '--saml-sp-attribute-prefix' ile 'emailaddress' ile basitleştirmelisiniz.                            |
| `--external-auth-full-name` (İsteğe bağlı) | SAML kullanıcı bilgilerinin kullanıcının tam adıyla eşleştirilmesi. Birden fazla öğe birleştirilebilir. Tam özellik adını 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name' gibi yapılandırmalı veya '--saml-sp-attribute-prefix' ile 'name' ile basitleştirmelisiniz. |
| `--external-auth-avatar-url` (İsteğe bağlı) | SAML kullanıcı bilgilerinin kullanıcının avatar URL'sine eşlenmesi. Tam özellik adını 'http://schemas.auth0.com/picture' gibi yapılandırmalı veya '--saml-sp-attribute-prefix' ile 'resim' ile basitleştirmelisiniz.                                                          |
| `--external-auth-default-inactive` (İsteğe bağlı) | Yeni SSO kullanıcılarının varsayılan olarak etkinleştirilmesini engeller.                                                                                                                                                                                                        |

Bu seçenekleri CLI bayrakları yerine ortam değişkenleri aracılığıyla da ayarlayabilirsiniz:

```bash
GPUSTACK_SAML_IDP_SERVER_URL="https://idp.example.com"
GPUSTACK_SAML_IDP_ENTITY_ID="your-idp-entity-id"
GPUSTACK_SAML_IDP_X509_CERT="your-idp-x509-cert"
GPUSTACK_SAML_SP_ENTITY_ID="your-sp-entity-id"
GPUSTACK_SAML_SP_ACS_URL="{your-server-url}/auth/saml/callback"
GPUSTACK_SAML_SP_X509_CERT="your-sp-x509-cert"
GPUSTACK_SAML_SP_PRIVATE_KEY="your-sp-private-key"
# Optional
GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"
GPUSTACK_SAML_SECURITY="{}"
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Okta"
GPUSTACK_EXTERNAL_AUTH_NAME="emailaddress"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
GPUSTACK_SAML_IDP_LOGOUT_URL="https://idp.example.com/saml/slo"  # if IdP supports SLO
GPUSTACK_SAML_SP_SLO_URL="{your-server-url}/auth/saml/logout/callback"
GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo"  # optional, adds a post-logout parameter for compatible IdPs
```

### Örnek: Auth0 SAML ile entegrasyon { #example-integrate-with-auth0-saml }

GPUStack'i SAML sağlayıcısı olarak Auth0 ile yapılandırmak için:

1. [auth0](https://auth0.com)'e gidin ve `Regular Web Applications` türüyle yeni bir uygulama oluşturun.

![create-saml-app](../assets/sso/create-saml-app.png)

2. Uygulama ayarlarından `Domain` alın ve İzin Verilen Geri Arama URL'lerine `<your-server-url>/auth/saml/callback` ekleyin. URL'yi sunucunuzun URL'siyle eşleşecek şekilde uyarlayın.

![auth0-saml-geri arama](../assets/sso/auth0-saml-callback.png)

3. **Gelişmiş Ayarlar → Sertifikalar**'de IdP `X.509 Certificate`'yi kopyalayın.

![auth0-saml-cert](../assets/sso/auth0-saml-cert.png)

4. **Endpoints** sekmesinde, IdP sunucunuzun URL'si olan `SAML Protocol URL`'yi bulun.

![auth0-saml-url](../assets/sso/auth0-saml-url.png)

5. SP sertifikası ve özel anahtar oluşturun:

```bash
openssl req -x509 -newkey rsa:2048 -keyout myservice.key -out myservice.cert -days 365 -nodes -subj "/CN=myservice.example.com"
```

!!! note

    SP yapılandırması için myservice.cert ve myservice.key kullanılacaktır.

6. GPUStack'i ilgili SAML yapılandırmasıyla çalıştırın. Aşağıdaki örnekte CUDA ile Docker kullanılmaktadır:

```bash
SP_CERT="$(cat myservice.cert)"
SP_PRIVATE_KEY="$(cat myservice.key)"
SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"

sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    -e GPUSTACK_SAML_IDP_SERVER_URL="<auth0-saml-protocol-url>" \
    -e GPUSTACK_SAML_IDP_ENTITY_ID="urn:<auth0-domain>" \
    -e GPUSTACK_SAML_IDP_X509_CERT="<auth0-x509-cert>" \
    -e GPUSTACK_SAML_SP_ENTITY_ID="urn:gpustack:sp" \
    -e GPUSTACK_SAML_SP_ACS_URL="<your-gpustack-server-url>/auth/saml/callback" \
    -e GPUSTACK_SAML_SP_X509_CERT="$SP_CERT" \
    -e GPUSTACK_SAML_SP_PRIVATE_KEY="$SP_PRIVATE_KEY" \
    -e GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="$SP_ATTRIBUTE_PREFIX" \
    -e GPUSTACK_SAML_IDP_LOGOUT_URL="<idp-slo-url-if-available>" \
    -e GPUSTACK_SAML_SP_SLO_URL="<your-gpustack-server-url>/auth/saml/logout/callback" \
    -e GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo" \
    gpustack/gpustack
```

!!! note

    Tüm IdP'ler standart SAML Tek Oturum Kapatma (SLO) sağlamaz. Auth0 SAML bağlantıları genellikle `singleLogoutService`'i açığa çıkarmaz. Mevcut değilse GPUStack, oturum kapatıldığında yerel oturumları temizlemeye devam edecektir; Auth0 ile tarayıcıdan tam çıkış yapmak için, `client_id` ve `returnTo` izinli olarak OIDC `v2/logout`'yi kullanmayı düşünün.

## CAS { #cas }

!!! note

    Modern Apereo CAS sunucuları, yerel CAS protokolünün yanı sıra OIDC ve SAML2 için modüller sunar. CAS sunucunuzdan herhangi biri etkinleştirilmişse, yukarıdaki [OIDC](#oidc) veya [SAML](#saml) bölümü aracılığıyla da bağlantı kurabilirsiniz — yetenek aynıdır; sunucunuzun gerçekte ortaya çıkardığı şeyle eşleşeni seçin. Aşağıdaki yerel CAS protokolü, yalnızca CAS mevcut olduğunda (eski Apereo / eğitim / kamu dağıtımlarında yaygındır) veya en küçük entegrasyon yüzeyini istediğinizde doğru seçimdir.

GPUStack, CAS (Merkezi Kimlik Doğrulama Hizmeti) protokolü sürüm 2.0 ve 3.0'ı destekler. Oturum açma akışı tarayıcıyı `<cas-server-url>/login?service=<gpustack-callback>`'e yönlendirir; Başarı durumunda GPUStack, döndürülen hizmet biletini standart `/serviceValidate` uç noktası aracılığıyla değiştirir ve kullanıcıyı XML yanıtından hazırlar. CAS sunucusuna kayıtlı geri arama URL'si `<server-url>/auth/cas/callback`'i işaret etmelidir.

CAS yanıtları, `cas:` XML ad alanıyla veya bu ad alanı olmadan ayrıştırılır, bu nedenle vahşi ortamda çoğu sunucu, ekstra ayar gerektirmeden çalışır. CAS Tek Oturum Kapatma, `/auth/logout`'e bağlanır ve etkin oturum CAS'tan başlatıldığında tarayıcıyı `<cas-server-url>/logout?service=<server-external-url>`'ye gönderir.

CAS yapılandırması için aşağıdaki CLI bayrakları mevcuttur:

| <div style="width:200px">Bayrak</div> | Açıklama |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--cas-server-url` | CAS sunucusu temel URL'si, örneğin, `https://cas.example.com/cas`. Bunun ayarlanması CAS oturum açmayı etkinleştirir.                                                                                                  |
| `--cas-callback-url` (İsteğe bağlı) | CAS sunucusuna kayıtlı CAS geri arama URL'si. Ayarlanmadığında istek zamanında varsayılan olarak `<server-url>/auth/cas/callback` olur.                                                                  |
| `--cas-validate-endpoint` (İsteğe bağlı) | CAS bilet doğrulama uç noktası, `--cas-server-url`'e göre. Varsayılan: `/p3/serviceValidate` (CAS 3.0, kullanıcı özelliklerini döndürür). `/serviceValidate`'i yalnızca CAS-3.0 öncesi bir sunucuyu hedeflerken kullanın. |
| `--cas-username-attribute` (İsteğe bağlı) | GPUStack kullanıcı adı olarak kullanılacak CAS XML öznitelik adı. Ayarlanmadığında varsayılan olarak CAS `cas:user` öğesine döner.                                                                                 |
| `--cas-full-name-attribute` (İsteğe bağlı) | Kullanıcının tam adı olarak kullanılacak CAS XML öznitelik adı, örneğin, `displayName`. Varsa `displayName` özelliğine, aksi takdirde kullanıcı adına geri döner.                                    |
| `--cas-avatar-attribute` (İsteğe bağlı) | Kullanıcının avatar URL'si olarak kullanılacak CAS XML öznitelik adı.                                                                                                                                    |
| `--external-auth-provider-name` (İsteğe bağlı) | Giriş sayfasının SSO düğmesinde gösterilen kimlik sağlayıcının adı, örneğin, `Campus SSO`. Ayarlanmadığında varsayılan olarak `CAS` olur.                                                                       |
| `--external-auth-default-inactive` (İsteğe bağlı) | Yeni SSO kullanıcılarının varsayılan olarak etkinleştirilmesini engeller.                                                                                                                                    |

Bu seçenekleri CLI bayrakları yerine ortam değişkenleri aracılığıyla da ayarlayabilirsiniz:

```bash
GPUSTACK_CAS_SERVER_URL="https://cas.example.com/cas"
# Optional
GPUSTACK_CAS_CALLBACK_URL="{your-server-url}/auth/cas/callback"
GPUSTACK_CAS_VALIDATE_ENDPOINT="/p3/serviceValidate"
GPUSTACK_CAS_USERNAME_ATTRIBUTE="uid"
GPUSTACK_CAS_FULL_NAME_ATTRIBUTE="displayName"
GPUSTACK_CAS_AVATAR_ATTRIBUTE=""
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Campus SSO"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
```

### Örnek: Apereo CAS sunucusuyla entegrasyon { #example-integrate-with-an-apereo-cas-server }

1. GPUStack'i CAS sunucusunda `serviceId` ile `<your-server-url>/auth/cas/callback` ile eşleşen bir hizmet olarak kaydedin. Eşlemeyi düşündüğünüz özelliklere izin verin (ör. `uid`, `displayName`).

2. GPUStack'i CAS yapılandırmasıyla çalıştırın. Aşağıdaki örnekte CUDA ile Docker kullanılmaktadır:

```bash
sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    -e GPUSTACK_SERVER_EXTERNAL_URL="https://gpustack.example.com" \
    -e GPUSTACK_CAS_SERVER_URL="https://cas.example.com/cas" \
    -e GPUSTACK_CAS_USERNAME_ATTRIBUTE="uid" \
    -e GPUSTACK_CAS_FULL_NAME_ATTRIBUTE="displayName" \
    gpustack/gpustack
```

!!! note

    CAS sunucunuz özel veya kurumsal bir CA tarafından verilen bir sertifika kullanıyorsa, CA sertifikalarını GPUStack kapsayıcısına nasıl bağlayacağınızı öğrenmek için [Ek Güvenilir CA'lar](../installation/installation.md#additional-trusted-cas)'e bakın.
