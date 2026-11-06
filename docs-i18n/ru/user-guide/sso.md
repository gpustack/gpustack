---
translation_status: translated
---

# Аутентификация с помощью единого входа (SSO) { #single-sign-on-sso-authentication }

GPUStack поддерживает такие методы аутентификации с единым входом (SSO), как OIDC, SAML и CAS. Это позволяет пользователям входить в систему, используя существующие учетные данные от внешнего поставщика удостоверений.

Одновременно активен один метод единого входа, и на странице входа в кнопку единого входа помечается используемый протокол, например `Log in with OIDC`. Вместо этого можно назвать поставщика удостоверений, например. `Log in with Okta`, установите `--external-auth-provider-name`.

## OIDC { #oidc }

Можно настроить любого поставщика аутентификации, поддерживающего OIDC. Утверждения `email`, `name` и `picture` используются, если они доступны. Разрешенный URI перенаправления должен включать `<server-url>/auth/oidc/callback`.

Если ваш поставщик OIDC использует сертификат, выданный частным или корпоративным центром сертификации, см. раздел [Дополнительные доверенные центры сертификации](../installation/installation.md#additional-trusted-cas), чтобы узнать, как подключить сертификаты центра сертификации в контейнер GPUStack.

Для конфигурации OIDC доступны следующие флаги CLI:

| <div style="width:180px">Флаг</div> | Описание |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--oidc-issuer` | URL-адрес эмитента OIDC. Обнаружение OIDC в разделе `<issuer>/.well-known/openid-configuration` будет использоваться для обнаружения конфигурации OIDC.                                   |
| `--oidc-client-id` | Идентификатор клиента OIDC.                                                                                                                                                      |
| `--oidc-client-secret` | Секрет клиента OIDC.                                                                                                                                                  |
| `--oidc-redirect-uri` | URI перенаправления, настроенный в вашем приложении OIDC. Должно быть установлено значение `<server-url>/auth/oidc/callback`.                                                         |
| `--external-auth-provider-name` (Необязательно) | Имя провайдера удостоверений, отображаемое на кнопке SSO на странице входа, например, `Okta`. По умолчанию `OIDC`, если не установлено.                                                     |
| `--external-auth-name` (Необязательно) | Сопоставление информации пользователя OIDC с именем пользователя, например, `preferred_username`. По умолчанию используется утверждение `email`, если оно доступно.                                        |
| `--external-auth-full-name` (Необязательно) | Сопоставление информации о пользователе OIDC с полным именем пользователя. Можно комбинировать несколько элементов, например `name` или `firstName+lastName`. По умолчанию используется утверждение `name`. |
| `--external-auth-avatar-url` (Необязательно) | Сопоставление информации пользователя OIDC с URL-адресом аватара пользователя. По умолчанию используется утверждение `picture`, если оно доступно.                                                         |
| `--external-auth-default-inactive` (Необязательно) | По умолчанию запрещает активацию новых пользователей единого входа.                                                                                                              |
| `--external-auth-post-logout-redirect-key` (Необязательно) | Общее имя параметра для перенаправления после выхода из системы между разными IdP (например, Auth0 `returnTo`). Применяется как к OIDC, так и к SAML.                                    |

Вы также можете установить эти параметры с помощью переменных среды вместо флагов CLI:

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

### Пример: интеграция с Auth0 OIDC. { #example-integrate-with-auth0-oidc }

Чтобы настроить GPUStack с Auth0 в качестве поставщика OIDC:

1. Перейдите к [auth0](https://auth0.com) и создайте новое приложение с типом `Regular Web Applications`.

![create-oidc-app](../assets/sso/create-oidc-app.png)

2. Получите `Domain`, `Client ID` и `Client Secret` из настроек приложения.

![auth0-app](../assets/sso/auth0-app.png)

3. Добавьте `<your-server-url>/auth/oidc/callback` в разрешенные URL-адреса обратного вызова. Адаптируйте URL-адрес в соответствии с URL-адресом вашего сервера.

4. В разделе «Разрешенные URL-адреса выхода из системы» добавьте `<your-server-url>/` (или желаемый URL-адрес после выхода из системы).

![auth0-обратный вызов](../assets/sso/auth0-callback.png)

Затем запустите GPUStack с соответствующей конфигурацией OIDC. В следующем примере используется Docker с CUDA:

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

    Монтирование пользовательского сертификата ЦС требуется только в том случае, если ваш поставщик OIDC подписан частным ЦС. Публичные поставщики OIDC, такие как Auth0, обычно не требуют этого.

## SAML { #saml }

GPUStack поддерживает аутентификацию SAML для единого входа (SSO). Это позволяет пользователям входить в систему, используя существующие учетные данные от внешнего поставщика удостоверений, поддерживающего SAML.

Для конфигурации SAML доступны следующие флаги CLI:

| <div style="width:180px">Флаг</div> | Описание |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--saml-idp-server-url` | URL-адрес сервера поставщика удостоверений SAML.                                                                                                                                                                                                                             |
| `--saml-idp-entity-id` | Идентификатор объекта поставщика удостоверений SAML.                                                                                                                                                                                                                              |
| `--saml-idp-x509-cert` | Сертификат поставщика удостоверений SAML X.509.                                                                                                                                                                                                                      |
| `--saml-sp-entity-id` | Идентификатор объекта поставщика услуг SAML.                                                                                                                                                                                                                               |
| `--saml-sp-acs-url` | URL-адрес потребительской службы утверждений поставщика услуг SAML. Должно быть установлено значение `<gpustack-server-url>/auth/saml/callback`.                                                                                                                                          |
| `--saml-sp-x509-cert` | Сертификат поставщика услуг SAML X.509.                                                                                                                                                                                                                       |
| `--saml-sp-private-key` | Закрытый ключ поставщика услуг SAML.                                                                                                                                                                                                                             |
| `--saml-idp-logout-url` (Необязательно) | URL-адрес конечной точки единого выхода поставщика удостоверений SAML.                                                                                                                                                                                                             |
| `--saml-sp-slo-url` (необязательно) | URL-адрес обратного вызова службы единого выхода поставщика услуг SAML (например, `<server-url>/auth/saml/logout/callback`).                                                                                                                                                     |
| `--saml-sp-attribute-prefix` (Необязательно) | Префикс атрибута поставщика услуг SAML, который используется для выборки атрибутов, указанных --external-auth-\*. например, 'http://schemas.auth0.com/'. |
| `--saml-security` (Необязательно) | Настройки безопасности SAML в формате JSON.                                                                                                                                                                                                                         |
| `--external-auth-provider-name` (Необязательно) | Имя провайдера удостоверений, отображаемое на кнопке SSO на странице входа, например, `Okta`. По умолчанию `SAML`, если не установлено.                                                                                                                                               |
| `--external-auth-name` (Необязательно) | Сопоставление информации пользователя SAML с именем пользователя. Вы должны настроить полное имя атрибута, например «http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress'», или упростить его с помощью «адреса электронной почты» с помощью «--saml-sp-attribute-prefix».                            |
| `--external-auth-full-name` (Необязательно) | Сопоставление информации пользователя SAML с полным именем пользователя. Несколько элементов могут быть объединены. Вы должны настроить полное имя атрибута, например «http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name'», или упростить его с помощью «name» с помощью «--saml-sp-attribute-prefix». |
| `--external-auth-avatar-url` (Необязательно) | Сопоставление информации пользователя SAML с URL-адресом аватара пользователя. Вы должны настроить полное имя атрибута, например «http://schemas.auth0.com/picture'», или упростить его с помощью «изображения» с помощью «--saml-sp-attribute-prefix».                                                          |
| `--external-auth-default-inactive` (Необязательно) | По умолчанию запрещает активацию новых пользователей единого входа.                                                                                                                                                                                                        |

Вы также можете установить эти параметры с помощью переменных среды вместо флагов CLI:

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

### Пример: интеграция с Auth0 SAML. { #example-integrate-with-auth0-saml }

Чтобы настроить GPUStack с Auth0 в качестве поставщика SAML:

1. Перейдите к [auth0](https://auth0.com) и создайте новое приложение с типом `Regular Web Applications`.

![create-saml-app](../assets/sso/create-saml-app.png)

2. Получите `Domain` в настройках приложения и добавьте `<your-server-url>/auth/saml/callback` в разрешенные URL-адреса обратного вызова. Адаптируйте URL-адрес в соответствии с URL-адресом вашего сервера.

![auth0-saml-callback](../assets/sso/auth0-saml-callback.png)

3. В **Дополнительные настройки → Сертификаты** скопируйте IdP `X.509 Certificate`.

![auth0-saml-cert](../assets/sso/auth0-saml-cert.png)

4. На вкладке **Конечные точки** найдите `SAML Protocol URL`, который является URL-адресом вашего сервера IdP.

![auth0-saml-url](../assets/sso/auth0-saml-url.png)

5. Создайте сертификат SP и закрытый ключ:

```bash
openssl req -x509 -newkey rsa:2048 -keyout myservice.key -out myservice.cert -days 365 -nodes -subj "/CN=myservice.example.com"
```

!!! note

    myservice.cert и myservice.key будут использоваться для конфигурации SP.

6. Запустите GPUStack с соответствующей конфигурацией SAML. В следующем примере используется Docker с CUDA:

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

    Не все поставщики удостоверений предоставляют стандартный единый выход SAML (SLO). SAML-соединения Auth0 обычно не предоставляют `singleLogoutService`. Если он недоступен, GPUStack все равно будет очищать локальные сеансы при выходе из системы; для полного выхода из браузера с помощью Auth0 рассмотрите возможность использования его OIDC `v2/logout` с разрешенными `client_id` и `returnTo`.

## CAS { #cas }

!!! note

    Современные CAS-серверы Apereo поставляют модули для OIDC и SAML2 наряду с собственным протоколом CAS. Если ваш CAS-сервер включен, вы также можете подключить его через раздел [OIDC](#oidc) или [SAML](#saml) выше — возможности те же; выберите тот, который соответствует тому, что на самом деле предоставляет ваш сервер. Приведенный ниже собственный протокол CAS является правильным выбором, когда доступен только CAS (обычно в старых развертываниях Apereo/образования/правительственных учреждений) или когда вам нужна наименьшая поверхность интеграции.

GPUStack поддерживает протокол CAS (центральная служба аутентификации) версий 2.0 и 3.0. Процесс входа в систему перенаправляет браузер на `<cas-server-url>/login?service=<gpustack-callback>`; в случае успеха GPUStack обменивает возвращенный билет службы через стандартную конечную точку `/serviceValidate` и предоставляет пользователю доступ к XML-ответу. URL-адрес обратного вызова, зарегистрированный на сервере CAS, должен указывать на `<server-url>/auth/cas/callback`.

Ответы CAS анализируются с пространством имен `cas:` XML или без него, поэтому большинство действующих серверов работают без дополнительной настройки. Единый выход CAS подключен к `/auth/logout` и отправляет браузер на `<cas-server-url>/logout?service=<server-external-url>`, когда активный сеанс инициирован из CAS.

Для конфигурации CAS доступны следующие флаги CLI:

| <div style="width:200px">Флаг</div> | Описание |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--cas-server-url` | Базовый URL-адрес CAS-сервера, например, `https://cas.example.com/cas`. Установка этого параметра позволяет войти в CAS.                                                                                                  |
| `--cas-callback-url` (Необязательно) | URL-адрес обратного вызова CAS, зарегистрированный на сервере CAS. По умолчанию `<server-url>/auth/cas/callback` во время запроса, если не установлено.                                                                  |
| `--cas-validate-endpoint` (Необязательно) | Конечная точка проверки билета CAS относительно `--cas-server-url`. По умолчанию: `/p3/serviceValidate` (CAS 3.0, возвращает атрибуты пользователя). Используйте `/serviceValidate` только при настройке сервера до версии CAS-3.0. |
| `--cas-username-attribute` (Необязательно) | Имя атрибута CAS XML, которое будет использоваться в качестве имени пользователя GPUStack. По умолчанию используется элемент CAS `cas:user`, если он не установлен.                                                                                 |
| `--cas-full-name-attribute` (Необязательно) | Имя атрибута CAS XML, которое будет использоваться в качестве полного имени пользователя, например, `displayName`. Возвращается к атрибуту `displayName`, если он присутствует, в противном случае — к имени пользователя.                                    |
| `--cas-avatar-attribute` (Необязательно) | Имя атрибута CAS XML, которое будет использоваться в качестве URL-адреса аватара пользователя.                                                                                                                                    |
| `--external-auth-provider-name` (Необязательно) | Имя провайдера удостоверений, отображаемое на кнопке SSO на странице входа, например, `Campus SSO`. По умолчанию `CAS`, если не установлено.                                                                       |
| `--external-auth-default-inactive` (Необязательно) | По умолчанию запрещает активацию новых пользователей единого входа.                                                                                                                                    |

Вы также можете установить эти параметры с помощью переменных среды вместо флагов CLI:

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

### Пример: интеграция с сервером Apereo CAS. { #example-integrate-with-an-apereo-cas-server }

1. Зарегистрируйте GPUStack как службу на сервере CAS, указав `serviceId`, соответствующий `<your-server-url>/auth/cas/callback`. Разрешите атрибуты, которые вы собираетесь сопоставить (например, `uid`, `displayName`).

2. Запустите GPUStack с конфигурацией CAS. В следующем примере используется Docker с CUDA:

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

    Если ваш сервер CAS использует сертификат, выданный частным или корпоративным центром сертификации, см. [Дополнительные доверенные центры сертификации](../installation/installation.md#additional-trusted-cas), чтобы узнать, как подключить сертификаты CA в контейнер GPUStack.
