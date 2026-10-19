---
hide:
  - toc
translation_status: translated
---

# gpustack start { #gpustack-start }

Запуск сервера или воркера GPUStack.

```bash
gpustack start [OPTIONS]
```

!!! note "Расположение аргументов CLI"

    В Docker фактическая команда, выполняемая `docker run`, состоит из **ENTRYPOINT** и **COMMAND**.

    В образе `gpustack/gpustack` задан `ENTRYPOINT`, который можно понимать как
    следующую команду (или считать ей эквивалентным):

        gpustack start

    Поэтому аргументы CLI для `gpustack start` следует указывать после имени образа,
    в конце команды `docker run`, а не как параметры самой команды `docker run`.

    Пример:

        docker run [docker options] gpustack/gpustack <start-args>

    В Kubernetes эти аргументы указываются в поле `args`
    спецификации контейнера.

## Настройки { #configurations }

### Общие параметры { #common-options }

| <div style="width:180px">Флаг</div>         | <div style="width:100px">По умолчанию</div> | Описание |
|---------------------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `--advertise-address` значение                 | (пусто)                                | IP-адрес для внешнего доступа.<br/>Если не задан, система автоматически определяет подходящий локальный IP-адрес. |
| `--config-file` значение                       | (пусто)                                | Путь к YAML-файлу конфигурации. |
| `-d` значение, `--debug` значение                 | `False`                                | Включить режим отладки. Короткий флаг -d не поддерживается в Windows, поскольку PowerShell резервирует его для CommonParameters. |
| `--data-dir` значение                          | (пусто)                                | Каталог хранения данных. Значение по умолчанию зависит от ОС. |
| `--cache-dir` значение                         | (пусто)                                | Каталог хранения кэша, например файлов моделей. По умолчанию — <data-dir>/cache. |
| `--huggingface-token` значение                 | (пусто)                                | Пользовательский токен доступа для аутентификации в Hugging Face Hub. Его также можно задать переменной среды `HF_TOKEN`. |
| `--insecure-tls`                            | `False`                                | Пропустить проверку TLS в собственных HTTPS-соединениях GPUStack: от воркера к серверу и от сервера к IdP внешней аутентификации. Предназначено для доверенных сетей, где сертификат сервера нельзя проверить иначе. |
| `--bin-dir` значение                           | (пусто)                                | Каталог для дополнительных бинарных файлов, например исполняемых файлов бэкендов разных версий. |
| `--pipx-path` значение                         | (пусто)                                | Путь к исполняемому файлу pipx для установки бэкендов разных версий. |
| `--system-default-container-registry` значение | `docker.io`                            | Реестр контейнеров по умолчанию, из которого GPUStack загружает системные образы и образы инференса.<br/>Все рабочие образы GPUStack должны находиться в пространстве имён `gpustack`; при многоуровневом пространстве имён `gpustack` должен оставаться последним уровнем, например `awesome.com/amazing-group/amazing-subgroup/gpustack`. Затем задайте этому флагу (либо ключу конфигурации сервера/воркера `system_default_container_registry`, либо переменной среды `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) родительский путь, например `awesome.com/amazing-group/amazing-subgroup`.<br/>Если образы нельзя разместить в пространстве имён `gpustack` (например, используется `gpustack-ai`), соответственно измените `--image-repo` и `--benchmark-image-repo`. |
| `--image-repo` значение                        | `gpustack/gpustack`                    | Переопределить репозиторий образа контейнера GPUStack по умолчанию. Не включайте тег в значение: GPUStack добавляет тег версии автоматически.<br/>Если образ `gpustack/gpustack:<tag>` находится вне пространства имён `gpustack`, например `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/gpustack:<tag>`, сначала задайте `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`, указав реестр с родительским пространством имён, а затем задайте этому флагу (либо ключу конфигурации сервера/воркера `image_repo`, либо переменной среды `GPUSTACK_IMAGE_REPO`) значение `gpustack-ai/gpustack`. |
| `--gateway-mode` значение                      | `auto`                                 | Режим работы шлюза. Варианты: embedded, in-cluster, external, disabled или auto (по умолчанию). |
| `--gateway-kubeconfig` значение                | (пусто)                                | Путь к kubeconfig для шлюза. Используется только при gateway-mode=external. |
| `--gateway-namespace` значение                 | `higress-system`                       | Пространство имён, в котором развёрнут компонент шлюза. |
| `--service-discovery-name` значение            | (пусто)                                | Имя сервиса обнаружения служб в DNS. Используется только при развёртывании в Kubernetes с обнаружением служб. |
| `--namespace` значение                         | (пусто)                                | Пространство имён Kubernetes, в котором GPUStack развёртывает правила маршрутизации шлюза и экземпляры моделей. |

### Параметры сервера { #server-options }

| <div style="width:180px">Флаг</div>              | <div style="width:100px">По умолчанию</div>           | Описание |
|--------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--port` значение                                   | `80`                                             | Порт, на котором сервер принимает соединения. |
| `--tls-port` значение                               | `443`                                            | Порт, на котором TLS-сервер принимает соединения. |
| `--api-port` значение                               | `30080`                                          | Порт, на котором API-сервер GPUStack принимает соединения. |
| `--proxy-port` значение                             | `30079`                                          | Порт, на котором прокси-сервер GPUStack принимает соединения. |
| `--database-port` значение                          | `5432`                                           | Порт встроенной базы данных PostgresSQL. |
| `--metrics-port` значение                           | `10161`                                          | Порт публикации метрик сервера. |
| `--disable-metrics`                              | `False`                                          | Отключить метрики сервера. |
| `--disable-worker`                               | (пусто)                                          | (УСТАРЕЛО) Отключить встроенный воркер сервера GPUStack. В новых установках встроенный воркер по умолчанию отсутствует. При необходимости включите его флагом '--enable-worker'. Если ни один флаг не задан, ради обратной совместимости встроенный воркер по умолчанию включается в старых установках, созданных до v2.0.1. |
| `--enable-worker`                                | `False`                                          | Включить встроенный воркер сервера GPUStack. |
| `--bootstrap-password` значение                     | Автоматически.                                  | Начальный пароль пользователя-администратора по умолчанию. |
| `--database-url` значение                           | Встроенная PostgreSQL.                             | URL базы данных. Поддерживаются PostgreSQL 13.0+, MySQL 8.0.36+ и совместимые базы данных, например openGauss (схема postgresql://) и OceanBase (схема mysql://). Пример: postgresql://username:password@host:port/dbname или mysql://username:password@host:port/dbname |
| `--redis-url` значение                              | None                                             | URL внешнего Redis для функций, которым требуется быстрое общее состояние (бэкенд общего состояния плагина LB). Формат: `redis://host[:port][/db]`. Учётные данные не допускаются, поскольку они сохранялись бы в CR шлюза; TLS (`rediss://`) не поддерживается. |
| `--ssl-keyfile` значение                            | (пусто)                                          | Путь к файлу SSL-ключа. |
| `--ssl-certfile` значение                           | (пусто)                                          | Путь к файлу SSL-сертификата. |
| `--ssl-ca-certfile` значение                        | (пусто)                                          | Путь к PEM-пакету CA, с помощью которого воркеры проверяют сервер GPUStack. Требуется, если сертификат сервера подписан частным CA. |
| `--force-auth-localhost`                         | (не задано)                                          | (Устарело, не действует) Запросы с localhost всегда проходят аутентификацию. Сохранено для обратной совместимости и будет удалено в будущем выпуске. |
| `--disable-update-check`                         | `False`                                          | Отключить проверку обновлений. |
| `--ota-server-url` значение                         | (пусто)                                          | URL OTA-сервера, с которого читаются официальные источники содержимого: индекс и файлы находятся непосредственно под этим URL, например https://ota.example.com/gpustack. По умолчанию используется публичный OTA-сервер. Укажите собственную копию, если он недоступен из вашей сети, или чтобы обслуживать внутреннюю установку из собственного репозитория. Настройка охватывает все три вида содержимого: они сохраняют статус `Official` и автоматическое обновление; действие `Reset to Official Source` в интерфейсе возвращает к этому адресу. См. [Обновления содержимого OTA](../user-guide/ota-content-updates.md). |
| `--disable-openapi-docs`                         | `False`                                          | Отключить автоматически создаваемые конечные точки документации OpenAPI: Swagger UI на /docs, ReDoc на /redoc и исходную спецификацию на /openapi.json. |
| `--model-catalog-file` значение                     | (пусто)                                          | Путь или URL файла каталога моделей. |
| `--enable-cors`                                  | `False`                                          | Включить Cross-Origin Resource Sharing (CORS) на сервере. |
| `--allow-credentials`                            | `False`                                          | Разрешить cookie и учётные данные в междоменных запросах. |
| `--allow-origins` значение                          | `["*"]`                                          | Источники, разрешённые для междоменных запросов. Для нескольких источников укажите флаг несколько раз. Пример: `--allow-origins https://example.com --allow-origins https://api.example.com` |
| `--allow-methods` значение                          | `["GET", "POST"]`                                | HTTP-методы, разрешённые в междоменных запросах. Для нескольких методов укажите флаг несколько раз. Пример: `--allow-methods GET --allow-methods POST` |
| `--allow-headers` значение                          | `["Authorization", "Content-Type", "X-API-Key"]` | HTTP-заголовки, разрешённые в междоменных запросах. Для нескольких заголовков укажите флаг несколько раз. Пример: `--allow-headers Authorization --allow-headers X-API-Key --allow-headers Content-Type` |
| `--oidc-issuer` значение                            | (пусто)                                          | URL издателя OpenID Connect. |
| `--oidc-client-id` значение                         | (пусто)                                          | Идентификатор клиента OpenID Connect. |
| `--oidc-client-secret` значение                     | (пусто)                                          | Секрет клиента OpenID Connect. |
| `--oidc-redirect-uri` значение                      | (пусто)                                          | URI перенаправления, настроенный в приложении OIDC. Должен иметь значение `<server-url>/auth/oidc/callback`. |
| `--external-auth-post-logout-redirect-key` значение | (пусто)                                          | Общее имя параметра для перенаправления после выхода у разных IdP. Применяется к OIDC и SAML. |
| `--oidc-skip-userinfo`                           | `False`                                          | Не запрашивать OIDC userinfo_endpoint, а попытаться прочитать сведения непосредственно из заголовка. |
| `--oidc-use-userinfo`                            | (пусто)                                          | [Устарело] Получать сведения о пользователе через конечную точку UserInfo после аутентификации. |
| `--saml-idp-server-url` значение                    | (пусто)                                          | URL сервера поставщика удостоверений SAML. |
| `--saml-idp-entity-id` значение                     | (пусто)                                          | Идентификатор сущности поставщика удостоверений SAML. |
| `--saml-idp-x509-cert` значение                     | (пусто)                                          | Сертификат X.509 поставщика удостоверений SAML. |
| `--saml-sp-entity-id` значение                      | (пусто)                                          | Идентификатор сущности поставщика услуг SAML. |
| `--saml-sp-acs-url` значение                        | (пусто)                                          | URL Assertion Consumer Service поставщика услуг SAML. Должен иметь значение `<server-url>/auth/saml/callback`. |
| `--saml-sp-x509-cert` значение                      | (пусто)                                          | Сертификат X.509 поставщика услуг SAML. |
| `--saml-sp-private-key` значение                    | (пусто)                                          | Закрытый ключ поставщика услуг SAML. |
| `--saml-idp-logout-url` значение                    | (пусто)                                          | URL конечной точки Single Logout поставщика удостоверений SAML. |
| `--saml-sp-slo-url` значение                        | (пусто)                                          | URL обратного вызова Single Logout Service поставщика услуг SAML (например, `<server-url>/auth/saml/logout/callback`). |
| `--saml-security` значение                          | (пусто)                                          | Параметры безопасности SAML в формате JSON. |
| `--cas-server-url` значение                         | (пусто)                                          | Базовый URL сервера CAS (Central Authentication Service), например `https://cas.example.com/cas`. Его настройка включает вход через CAS. |
| `--cas-callback-url` значение                       | (пусто)                                          | URL обратного вызова CAS, зарегистрированный на CAS-сервере. Если не задан, используется `<server-url>/auth/cas/callback`. |
| `--cas-validate-endpoint` значение                  | `/p3/serviceValidate`                            | Конечная точка проверки билетов CAS относительно `--cas-server-url`. Для серверов до CAS 3.0 используйте `/serviceValidate` (без поддержки атрибутов). |
| `--cas-username-attribute` значение                 | (пусто)                                          | Имя XML-атрибута CAS, используемого как имя пользователя GPUStack. Если не задано, используется элемент CAS `cas:user`. |
| `--cas-full-name-attribute` значение                | (пусто)                                          | Имя XML-атрибута CAS, используемого как полное имя пользователя. |
| `--cas-avatar-attribute` значение                   | (пусто)                                          | Имя XML-атрибута CAS, используемого как URL аватара пользователя. |
| `--external-auth-provider-name` значение            | (пусто)                                          | Имя поставщика удостоверений, отображаемое на кнопке SSO страницы входа, например `Okta`. Если не задано, используется имя протокола (OIDC/SAML/CAS). |
| `--external-auth-name` значение                     | (пусто)                                          | Сопоставление сведений внешней аутентификации с именем пользователя, например preferred_username. |
| `--external-auth-full-name` значение                | (пусто)                                          | Сопоставление сведений внешней аутентификации с полным именем пользователя. Можно объединять несколько элементов, например `name` или `firstName+lastName`. |
| `--external-auth-avatar-url` значение               | (пусто)                                          | Сопоставление сведений внешней аутентификации с URL аватара пользователя. |
| `--external-auth-default-inactive`               | `False`                                          | True, если новые пользователи должны быть по умолчанию деактивированы. |
| `--external-auth-insecure-skip-tls-verify`       | `False`                                          | Пропустить проверку TLS при установлении соединения с IdP внешней аутентификации (OIDC/CAS). Только для тестирования самоподписанных IdP; никогда не используйте в рабочей среде. |
| `--server-external-url` значение                    | (пусто)                                          | Внешний URL сервера для регистрации воркеров. Обязателен при подготовке воркеров через облачных провайдеров, чтобы они могли корректно подключиться к серверу. |
| `--trusted-hosts` значение                          | (пусто)                                          | Список разрешённых значений заголовка `the X-Forwarded-Host` за обратным прокси. Если не задан, формируется из `--server-external-url`. Если оба параметра не заданы, используется `*`, доверяющий любому хосту (не рекомендуется, если сервер доступен не только через доверенный прокси). |
| `--saml-sp-attribute-prefix` значение               | (пусто)                                          | Префикс атрибутов поставщика услуг SAML для получения атрибутов, заданных через --external-auth-\*. |
| `--gateway-concurrency` значение                    | `16`                                             | Число одновременных соединений встроенного шлюза. |
| `--disable-builtin-observability`                | `False`                                          | Отключить встроенные сервисы Grafana и Prometheus. |
| `--builtin-prometheus-port` значение                | `19090`                                          | Порт встроенного сервиса Prometheus. |
| `--builtin-grafana-port` значение                   | `13000`                                          | Порт встроенного сервиса Grafana. |
| `--prometheus-url` значение                         | (пусто)                                          | Базовый URL внешнего Prometheus для запросов. Если не задан, используется встроенный сервис, когда он включён. Задайте этот параметр, чтобы функции на основе метрик (панели моделей и PD, самопроверка пропускной способности) работали в установке с собственным стеком наблюдаемости. |
| `--grafana-url` значение                            | (пусто)                                          | Базовый URL Grafana для перенаправления к панелям и проксирования. Должен быть доступен из браузера, а не только по имени внутри контейнера. Если задан, встроенные Grafana и Prometheus отключаются. Требуется только для внешней Grafana. |
| `--grafana-worker-dashboard-uid` значение           | (пусто)                                          | UID панели Grafana для перенаправления к панели воркера. |
| `--grafana-model-dashboard-uid` значение            | (пусто)                                          | UID панели Grafana для перенаправления к панели модели. |

### Параметры воркера { #worker-options }

| <div style="width:180px">Флаг</div>      | <div style="width:100px">По умолчанию</div> | Описание |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-t` значение, `--token` значение              | Автоматически.                        | Общий секрет для регистрации воркера. |
| `-s` значение, `--server-url` значение         | (пусто)                                | Сервер для подключения. |
| `--worker-name` значение                    | (пусто)                                | Имя узла-воркера. По умолчанию используется имя хоста. |
| `--worker-ip` значение                      | (пусто)                                | IP-адрес узла-воркера. По умолчанию определяется автоматически. |
| `--disable-worker-metrics`               | `False`                                | Отключить метрики. |
| `--worker-metrics-port` значение            | `10151`                                | Порт публикации метрик. |
| `--worker-port` значение                    | `10150`                                | Порт, на котором воркер принимает соединения. Используйте одинаковое значение для всех воркеров. |
| `--service-port-range` значение             | `40000-40063`                          | Диапазон портов сервисов инференса в виде строки 'N1-N2'. Обе границы включаются. |
| `--ray-port-range` значение                 | `41000-41999`                          | Диапазон портов сервисов Ray (для распределённого развёртывания vLLM) в виде строки 'N1-N2'. Обе границы включаются. |
| `--log-dir` значение                        | (пусто)                                | Каталог хранения журналов. |
| `--system-reserved` значение                | (пусто)                                | Ресурсы, резервируемые системой для воркера при планировании, в GiB. По умолчанию ресурсы не резервируются. Пример: '{"ram": 2, "vram": 1}'. |
| `--enable-hf-transfer`                   | `False`                                | [Устарело] Поддержка hf_transfer удалена в huggingface_hub v1.0; этот флаг не действует. Теперь по умолчанию используется загрузчик hf_xet. |
| `--enable-hf-xet`                        | `False`                                | [Устарело] Включить загрузку файлов моделей через Hugging Face Xet. |
| `--worker-ifname` значение                  | (пусто)                                | Имя сетевого интерфейса узла-воркера. По умолчанию определяется автоматически. |
| `--kv-transfer-ifname` значение                      | (пусто)                                | Сетевой интерфейс для передачи KV-кэша при разделённом развёртывании prefill/decode. Определяется автоматически, если на хосте ровно один маршрутизируемый интерфейс. При нескольких интерфейсах обязателен, поскольку неверный выбор заставляет движок объявлять адрес, недоступный другому участнику. |
| `--proxy-mode` значение                     | (пусто)                                | Режим проксирования для доступа к экземплярам моделей: `direct`, `worker` или `tunnel` (воркер сам соединяется с сервером; для воркеров за межсетевым экраном/NAT). По умолчанию — `direct` для встроенного и `worker` для отдельного воркера. |
| `--benchmark-image-repo` значение           | `gpustack/benchmark-runner`            | Переопределить репозиторий образа контейнера бенчмарков GPUStack по умолчанию. В отличие от `--image-repo`, значение используется как есть, поэтому включайте тег.<br/>Если образ `gpustack/benchmark-runner:<tag>` находится вне пространства имён `gpustack`, например `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/benchmark-runner:<tag>`, сначала укажите реестр с родительским пространством имён через `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`, затем задайте этому флагу (либо ключу конфигурации воркера `benchmark_image_repo`, либо переменной среды `GPUSTACK_BENCHMARK_IMAGE_REPO`) значение `gpustack-ai/benchmark-runner:<tag>`. |
| `--benchmark-dir` значение                  | `<data-dir>/benchmarks`                | Каталог хранения результатов бенчмарков. |
| `--benchmark-max-duration-seconds` значение | (пусто)                                | Максимальная длительность бенчмарка до тайм-аута. Пустое значение отключает ограничение. |

### Доступные переменные среды { #available-environment-variables }

Большинство параметров командной строки также можно задать переменными среды с префиксом `GPUSTACK_`, записанными
в верхнем регистре (например, `--data-dir` можно задать через `GPUSTACK_DATA_DIR`).

Переменные среды, не соответствующие перечисленным выше параметрам командной строки, описаны
в [документации по переменным среды](../environment-variables.md).

## Файл конфигурации { #config-file }

При запуске сервера или воркера GPUStack параметры можно задать в YAML-файле конфигурации. Ниже приведён
полный пример:

```yaml
# Common Options
advertise_address: exposed_server_or_worker_ip
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
benchmark_dir: /path/to/benchmark_dir
benchmark_image_repo: gpustack/benchmark-runner
token: your_token
huggingface_token: your_huggingface_token
insecure_tls: false

# Server Options
port: 80
tls_port: 443
api_port: 30080
metrics_port: 10161
disable_metrics: false
enable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://username:password@host:port/dbname
# database_url: mysql://username:password@host:port/dbname
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
ssl_ca_certfile: /path/to/ca-bundle.pem
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
enable_cors: false
allow_credentials: false
allow_origins: [ "*" ]
allow_methods: [ "GET", "POST" ]
allow_headers: [ "Authorization", "Content-Type", "X-API-Key" ]
oidc_issuer: https://your_oidc_issuer
oidc_client_id: your_oidc_client_id
oidc_client_secret: your_oidc_client_secret
oidc_redirect_uri: http://your_gpustack_server_url/auth/oidc/callback
saml_idp_server_url: https://your_saml_idp_server_url
saml_idp_entity_id: your_saml_idp_entity_id
saml_idp_x509_cert: your_saml_idp_x509_cert_pem
saml_sp_entity_id: your_saml_sp_entity_id
saml_sp_acs_url: http://your_gpustack_server_url/auth/saml/callback
saml_sp_x509_cert: your_saml_sp_x509_cert_pem
saml_sp_private_key: your_saml_sp_private_key_pem
saml_security: '{"wantAssertionsSigned": true, "wantMessagesSigned": true}'
cas_server_url: https://your_cas_server_url
cas_username_attribute: uid
cas_full_name_attribute: displayName
external_auth_provider_name: Okta
external_auth_name: email
external_auth_full_name: name
external_auth_avatar_url: picture
server_external_url: http://your_gpustack_server_url_for_external_access
trusted_hosts: [ "your_reverse_proxy_hostname" ]
disable_builtin_observability: false
builtin_prometheus_port: 19090
builtin_grafana_port: 13000
# Per-plugin overrides for the API gateway, keyed by the plugin's manifest name
# -- which for several of them is not the name of the WasmPlugin resource they
# are deployed as (`transformer` is deployed as `gpustack-header-transformer`,
# `ai-statistics` as `gpustack-ai-statistics`). Keying by the wrong one matches
# nothing, silently.
#
# `url` / `sha256` / `image_pull_policy` say where Envoy pulls the module from,
# replacing the address derived from the bundled plugin manifest. `config` is
# the plugin's own settings and is validated against what that plugin accepts,
# so a field it does not declare fails startup rather than being ignored.
gateway_plugin:
  gpustack-ext-auth:
    config:
      local_auth:
        # Whether the gateway authenticates API keys itself. Turned off, every
        # credential is forwarded to the server instead, as it was before.
        enabled: true
      authz:
        # How long the gateway waits for the server to authorize a request.
        # Sized for a loaded server rather than a healthy one: a server that is
        # down refuses the connection and is detected at once, so this governs
        # only the case of one too busy to answer yet -- where waiting is what
        # keeps a load spike from being read as an outage.
        timeout: 30000
      # Let a caller the gateway authenticated itself through while the server
      # is unreachable -- a rolling upgrade included. Authorization is skipped
      # for the duration; unknown keys and anonymous callers are still refused.
      failure_mode_allow_authenticated: true
  ai-statistics:
    config:
      enable_content_types: [ application/json, text/event-stream ]
  # Fields are the plugin's own camelCase config keys; only whitelisted ones
  # are accepted, so a field the server does not vouch for fails startup
  # instead of being forwarded.
  gpustack-ai-proxy:
    config:
      # Reject tool-call messages that arrive without the assistant message
      # that initiated them ("strict"), or the plugin's default ("off").
      toolCallValidation: strict

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_worker_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
ray_port_range: 41000-41999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
enable_hf_transfer: false
proxy_mode: worker
```
