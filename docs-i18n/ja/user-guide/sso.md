# シングルサインオン（SSO）認証 { #single-sign-on-sso-authentication }

GPUStack は OIDC、SAML、CAS などのシングルサインオン（SSO）認証に対応しています。ユーザーは、外部の ID プロバイダーの既存の認証情報でログインできます。

同時に有効になる SSO 方式は 1 つです。ログインページの SSO ボタンには、`Log in with OIDC` など、使用中のプロトコル名を表示します。代わりに `Log in with Okta` のように ID プロバイダー名を表示するには、`--external-auth-provider-name` を設定します。

## OIDC { #oidc }

OIDC に対応する認証プロバイダーであれば構成できます。存在する場合は `email`、`name`、`picture` クレームを使用します。許可するリダイレクト URI に `<server-url>/auth/oidc/callback` を含めてください。

OIDC プロバイダーがプライベート CA または企業の CA が発行した証明書を使用する場合は、GPUStack コンテナに CA 証明書をマウントする方法について、[追加の信頼する CA](../installation/installation.md#additional-trusted-cas)を参照してください。

OIDC の構成には、次の CLI フラグを使用できます。

| <div style="width:180px">フラグ</div> | 説明 |
| --- | --- |
| `--oidc-issuer` | OIDC 発行者 URL。`<issuer>/.well-known/openid-configuration` で OIDC 構成を検出します。 |
| `--oidc-client-id` | OIDC クライアント ID。 |
| `--oidc-client-secret` | OIDC クライアントシークレット。 |
| `--oidc-redirect-uri` | OIDC アプリケーションに設定したリダイレクト URI。`<server-url>/auth/oidc/callback` に設定する必要があります。 |
| `--external-auth-provider-name`（任意） | ログインページの SSO ボタンに表示する ID プロバイダー名（`Okta` など）。未設定時は `OIDC` です。 |
| `--external-auth-name`（任意） | OIDC ユーザー情報からユーザー名へのマッピング（`preferred_username` など）。既定では、存在する場合は `email` クレームを使用します。 |
| `--external-auth-full-name`（任意） | OIDC ユーザー情報から氏名へのマッピング。`name` や `firstName+lastName` のように複数の要素を組み合わせられます。既定では `name` クレームを使用します。 |
| `--external-auth-avatar-url`（任意） | OIDC ユーザー情報からアバター URL へのマッピング。既定では、存在する場合は `picture` クレームを使用します。 |
| `--external-auth-default-inactive`（任意） | 新しい SSO ユーザーが既定で有効になることを防ぎます。 |
| `--external-auth-post-logout-redirect-key`（任意） | IdP ごとのログアウト後リダイレクトに使用する汎用パラメーター名（Auth0 の `returnTo` など）。OIDC と SAML の両方に適用します。 |

CLI フラグの代わりに、環境変数でこれらのオプションを設定することもできます。

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

### 例：Auth0 OIDC との連携 { #example-integrate-with-auth0-oidc }

Auth0 を OIDC プロバイダーとして GPUStack に構成する手順：

1. [auth0](https://auth0.com) で、種類が `Regular Web Applications` の新しいアプリケーションを作成します。

![OIDC アプリケーションの作成](../assets/sso/create-oidc-app.png)

2. アプリケーション設定から `Domain`、`Client ID`、`Client Secret` を取得します。

![Auth0 アプリケーション](../assets/sso/auth0-app.png)

3. Allowed Callback URLs に `<your-server-url>/auth/oidc/callback` を追加します。Server の URL に合わせて変更してください。

4. Allowed Logout URLs に `<your-server-url>/`、または必要なログアウト後 URL を追加します。

![Auth0 のコールバック設定](../assets/sso/auth0-callback.png)

次に、OIDC の構成を指定して GPUStack を実行します。以下は CUDA 対応の Docker を使用する例です。

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

    カスタム CA 証明書のマウントは、OIDC プロバイダーの証明書がプライベート CA によって署名されている場合だけ必要です。Auth0 などの公開 OIDC プロバイダーでは通常不要です。

## SAML { #saml }

GPUStack は、SSO 用の SAML 認証に対応しています。ユーザーは、SAML に対応する外部 ID プロバイダーの既存の認証情報でログインできます。

SAML の構成には、次の CLI フラグを使用できます。

| <div style="width:180px">フラグ</div> | 説明 |
| --- | --- |
| `--saml-idp-server-url` | SAML ID プロバイダーの Server URL。 |
| `--saml-idp-entity-id` | SAML ID プロバイダーのエンティティ ID。 |
| `--saml-idp-x509-cert` | SAML ID プロバイダーの X.509 証明書。 |
| `--saml-sp-entity-id` | SAML サービスプロバイダーのエンティティ ID。 |
| `--saml-sp-acs-url` | SAML サービスプロバイダーの Assertion Consumer Service URL。`<gpustack-server-url>/auth/saml/callback` に設定してください。 |
| `--saml-sp-x509-cert` | SAML サービスプロバイダーの X.509 証明書。 |
| `--saml-sp-private-key` | SAML サービスプロバイダーの秘密鍵。 |
| `--saml-idp-logout-url`（任意） | SAML ID プロバイダーのシングルログアウトのエンドポイント URL。 |
| `--saml-sp-slo-url`（任意） | SAML サービスプロバイダーの Single Logout Service コールバック URL（`<server-url>/auth/saml/logout/callback` など）。 |
| `--saml-sp-attribute-prefix`（任意） | --external-auth-\* で指定した属性を取得するための、SAML サービスプロバイダーの属性プレフィックス。例：'http://schemas.auth0.com/'。 |
| `--saml-security`（任意） | JSON 形式の SAML セキュリティ設定。 |
| `--external-auth-provider-name`（任意） | ログインページの SSO ボタンに表示する ID プロバイダー名（`Okta` など）。未設定時は `SAML` です。 |
| `--external-auth-name`（任意） | SAML ユーザー情報からユーザー名へのマッピング。'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress' のように完全な属性名を指定するか、'--saml-sp-attribute-prefix' で 'emailaddress' のように簡略化する必要があります。 |
| `--external-auth-full-name`（任意） | SAML ユーザー情報から氏名へのマッピング。複数の要素を組み合わせられます。'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name' のように完全な属性名を指定するか、'--saml-sp-attribute-prefix' で 'name' のように簡略化する必要があります。 |
| `--external-auth-avatar-url`（任意） | SAML ユーザー情報からアバター URL へのマッピング。'http://schemas.auth0.com/picture' のように完全な属性名を指定するか、'--saml-sp-attribute-prefix' で 'picture' のように簡略化する必要があります。 |
| `--external-auth-default-inactive`（任意） | 新しい SSO ユーザーが既定で有効になることを防ぎます。 |

CLI フラグの代わりに、環境変数でこれらのオプションを設定することもできます。

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

### 例：Auth0 SAML との連携 { #example-integrate-with-auth0-saml }

Auth0 を SAML プロバイダーとして GPUStack に構成する手順：

1. [auth0](https://auth0.com) で、種類が `Regular Web Applications` の新しいアプリケーションを作成します。

![SAML アプリケーションの作成](../assets/sso/create-saml-app.png)

2. アプリケーション設定から `Domain` を取得し、Allowed Callback URLs に `<your-server-url>/auth/saml/callback` を追加します。Server の URL に合わせて変更してください。

![Auth0 SAML のコールバック](../assets/sso/auth0-saml-callback.png)

3. **Advanced Settings → Certificates** で、IdP の `X.509 Certificate` をコピーします。

![Auth0 SAML 証明書](../assets/sso/auth0-saml-cert.png)

4. **Endpoints** タブで、IdP Server URL にあたる `SAML Protocol URL` を確認します。

![Auth0 SAML URL](../assets/sso/auth0-saml-url.png)

5. SP 証明書と秘密鍵を生成します。

```bash
openssl req -x509 -newkey rsa:2048 -keyout myservice.key -out myservice.cert -days 365 -nodes -subj "/CN=myservice.example.com"
```

!!! note

    myservice.cert と myservice.key を SP の構成に使用します。

6. SAML の構成を指定して GPUStack を実行します。以下は CUDA 対応の Docker を使用する例です。

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

    すべての IdP が標準の SAML シングルログアウト（SLO）を提供するわけではありません。Auth0 の SAML 接続では、通常 `singleLogoutService` は公開されません。利用できなくても、GPUStack はログアウト時にローカルセッションを消去します。Auth0 でブラウザーから完全にサインアウトするには、`client_id` と `returnTo` を許可した OIDC の `v2/logout` の使用を検討してください。

## CAS { #cas }

!!! note

    最近の Apereo CAS Server は、ネイティブな CAS プロトコルに加え、OIDC と SAML2 のモジュールを提供しています。どちらかを有効にしている場合は、上の [OIDC](#oidc) または [SAML](#saml) の手順で連携することもできます。機能は同じなので、Server が実際に公開しているものに合わせて選択してください。以下のネイティブ CAS プロトコルは、CAS だけが利用可能な場合（古い Apereo、教育機関、行政機関の環境で一般的）や、連携範囲を最小限にしたい場合に適しています。

GPUStack は CAS（Central Authentication Service）プロトコルの 2.0 と 3.0 に対応しています。ログイン時はブラウザーを `<cas-server-url>/login?service=<gpustack-callback>` にリダイレクトします。成功すると、返されたサービステケットを標準の `/serviceValidate` エンドポイントで交換し、XML 応答からユーザーを作成します。CAS Server に登録するコールバック URL は、`<server-url>/auth/cas/callback` を指す必要があります。

CAS 応答は、`cas:` XML 名前空間の有無にかかわらず解析するため、多くの Server では追加の調整なしで動作します。CAS のシングルサインアウトは `/auth/logout` に接続され、現在のセッションが CAS から開始された場合は、ブラウザーを `<cas-server-url>/logout?service=<server-external-url>` に送ります。

CAS の構成には、次の CLI フラグを使用できます。

| <div style="width:200px">フラグ</div> | 説明 |
| --- | --- |
| `--cas-server-url` | CAS Server のベース URL（`https://cas.example.com/cas` など）。設定すると CAS ログインが有効になります。 |
| `--cas-callback-url`（任意） | CAS Server に登録する CAS コールバック URL。未設定時は、リクエスト時に `<server-url>/auth/cas/callback` を使用します。 |
| `--cas-validate-endpoint`（任意） | `--cas-server-url` を基準とする、CAS チケット検証エンドポイント。既定は `/p3/serviceValidate`（CAS 3.0、ユーザー属性を返す）です。CAS 3.0 より前の Server を対象とする場合だけ、`/serviceValidate` を使用してください。 |
| `--cas-username-attribute`（任意） | GPUStack のユーザー名に使用する CAS XML 属性名。未設定時は CAS の `cas:user` 要素を使用します。 |
| `--cas-full-name-attribute`（任意） | ユーザーの氏名に使用する CAS XML 属性名（`displayName` など）。その属性がない場合は、`displayName` 属性があれば使用し、なければユーザー名を使用します。 |
| `--cas-avatar-attribute`（任意） | ユーザーのアバター URL に使用する CAS XML 属性名。 |
| `--external-auth-provider-name`（任意） | ログインページの SSO ボタンに表示する ID プロバイダー名（`Campus SSO` など）。未設定時は `CAS` です。 |
| `--external-auth-default-inactive`（任意） | 新しい SSO ユーザーが既定で有効になることを防ぎます。 |

CLI フラグの代わりに、環境変数でこれらのオプションを設定することもできます。

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

### 例：Apereo CAS Server との連携 { #example-integrate-with-an-apereo-cas-server }

1. CAS Server に GPUStack をサービスとして登録し、`serviceId` が `<your-server-url>/auth/cas/callback` に一致するようにします。マッピングする属性（`uid`、`displayName` など）を許可してください。

2. CAS の構成を指定して GPUStack を実行します。以下は CUDA 対応の Docker を使用する例です。

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

    CAS Server がプライベート CA または企業の CA が発行した証明書を使用する場合は、GPUStack コンテナに CA 証明書をマウントする方法について、[追加の信頼する CA](../installation/installation.md#additional-trusted-cas)を参照してください。
