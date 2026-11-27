---
translation_status: translated
---

# Установка через Helm { #installation-via-helm }

Начиная с версии 2.2.0, GPUStack можно развертывать в Kubernetes с помощью внутрикластерного API-шлюза [Higress](https://higress.io/) с использованием официального чарта Helm.

!!! note

    Развертывание GPUStack в Kubernetes с помощью Higress в настоящее время считается **экспериментальным**. Прежде чем продолжить, ознакомьтесь с [ограничениями](#limitations).

## Предварительные условия { #prerequisites }

| Компонент | Версия |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

Кроме того:

- В кластере необходимо настроить значение по умолчанию `StorageClass` для тома данных сервера (в k3s значение по умолчанию — `local-path`). Альтернативно, установите `server.dataVolume.hostPath`, чтобы использовать том пути хоста вместо PVC.
- Для работников графического процессора убедитесь, что на узлах установлены соответствующие драйверы графического процессора и наборы инструментов для контейнеров (см. [Требования к установке](./requirements.md)), а также установлено [Обнаружение функций узла (NFD)](https://kubernetes-sigs.github.io/node-feature-discovery/), чтобы узлы графического процессора были помечены.

## Ограничения { #limitations }

- Сервер GPUStack развернут как `StatefulSet` и в настоящее время **не** поддерживает более одной реплики.
- По умолчанию используется встроенная база данных `PostgreSQL`. Для производства рекомендуется указать внешнюю базу данных через `server.externalDatabaseURL`.
- Плагины Higress обслуживаются специальным развертыванием (Deployment), установленным вместе с GPUStack; по умолчанию оно работает на образе сервера GPUStack, поэтому версии модулей плагинов синхронизированы с сервером. Шлюз Higress загружает плагины из этого сервиса при перезапуске; если сервис недоступен, запуск шлюза блокируется до тех пор, пока плагины не станут доступны.
- Входящий в комплект субчарт `higress-core` развертывает Higress в качестве входного контроллера кластера. Если другой входящий контроллер уже запущен, установите `higress-core.enabled=false` и укажите `gateway.ingressClassname` на существующий экземпляр Higress.

## Установите k3s (необязательно) { #install-k3s-optional }

В следующих шагах в качестве примера дистрибутива Kubernetes используется k3s. Другие CNCF-совместимые дистрибутивы (RKE2, кластеры на базе kubeadm или управляемые облачные Kubernetes) также работают, если они соответствуют указанным выше требованиям к версии.

Установите k3s с отключенным Traefik, поскольку в качестве входящего контроллера используется Higress. Информацию о кластерах k3s высокой доступности см. в документации [k3s](https://docs.k3s.io/datastore/ha).

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

Проверьте настройку:

```bash
kubectl version
```

## Установите GPUStack с помощью Helm. { #install-gpustack-with-helm }

Чарт публикуется как артефакт OCI в Docker Hub по адресу `oci://registry-1.docker.io/gpustack/gpustack-chart`. Версия чарта отслеживает выпуск GPUStack (например, чарт `2.2.0` поставляется с GPUStack `v2.2.0`).

Установите последнюю стабильную версию. Если `--version` опущено, Helm автоматически обрабатывает новейшую стабильную версию чарта и пропускает сборки для разработчиков и предварительные версии:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

!!! note

    Имя выпуска намеренно `gpustack-server`, а не `gpustack`: при регистрации собственного кластера сервера как рабочего (через **Register Cluster**) в том же пространстве имен `gpustack-system` устанавливается рабочий выпуск Helm с фиксированным именем `gpustack`. Отдельное имя для выпуска сервера позволяет избежать этого конфликта, поэтому регистрация кластера в том же пространстве имен и последующее обновление его воркеров не несут риска конфликта имен выпусков. Существующая установка с выпуском `gpustack` продолжает работать; команду `helm upgrade gpustack-server …` выполняйте только при установке с именем, указанным выше.

    Это справедливо для значения по умолчанию `worker.enabled=false`. Если вы включите управляемые чартом воркеры в выпуске сервера (`worker.enabled=true`), выпуск сервера также отрендерит объекты с фиксированными именами — ConfigMap `worker-config`, Service `worker` и Secret `registration-token` — и регистрация того же пространства имен будет отклонена предварительной проверкой bootstrap (после того как манифест регистрации уже заменит Secret `registration-token`, который читает pod сервера). В этом случае зарегистрируйте кластер в другом пространстве имен.

Чтобы закрепить конкретную версию, добавьте `--version <chart-version>` (версия чарта соответствует выпуску GPUStack, например `2.2.0`):

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

По умолчанию субчарт `higress-core` включён и развертывается вместе с GPUStack. Если в вашем кластере уже установлен Higress, отключите встроенный Higress и укажите GPUStack на существующий экземпляр:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

Чтобы настроить параметры, используйте `--set key=value` или `-f your-values.yaml` во время установки. См. [Параметры чарта](#chart-parameters) ниже.

### Установка из клонированного репозитория. { #installing-from-a-cloned-repository }

Альтернативно клонируйте репозиторий и установите его из локального каталога чарта (полезно для изолированных сред или при изменении чарта):

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack-server ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### Установка Higress отдельно { #installing-higress-separately }

Если вы установили `higress-core.enabled=false`, установите совместимый экземпляр Higress перед развертыванием GPUStack:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

Убедитесь, что IngressClass доступен и что его имя соответствует `gateway.ingressClassname`:

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

Для настройки Higress обратитесь к [документации Higress](https://higress.cn/en/docs/latest/ops/deploy-by-helm).

## Доступ к GPUStack { #accessing-gpustack }

Подождите, пока серверный модуль будет готов:

```bash
kubectl get pods -n gpustack-system -w
```

Получите первоначальный пароль администратора. Pod сервера называется `<release-name>-server-0` (`gpustack-server-server-0` для выпуска по умолчанию `gpustack-server`):

```bash
kubectl exec -it -n gpustack-system gpustack-server-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

Если вы не установили `server.ingress.hostname`, получите адрес пользовательского интерфейса GPUStack из входа, имя которого совпадает с именем выпуска (`gpustack-server` по умолчанию):

```bash
kubectl get ingress -n gpustack-system gpustack-server \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Откройте адрес в браузере и войдите в систему, используя имя пользователя `admin` и пароль, полученные выше.

## Общая конфигурация { #common-configuration }

### Включение рабочих демонсетов { #enabling-worker-daemonsets }

Worker DaemonSet по умолчанию отключены (`worker.enabled=false`). При включении чарт создаёт DaemonSet для CPU-воркеров (`<release>-worker`) и по одному DaemonSet для каждого производителя GPU из `worker.gpuVendors` (с именем `<release>-worker-<vendor>`).

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

Поддерживаемые значения `worker.gpuVendors`: `nvidia`, `mthreads`, `amd`, `ascend`, `hygon`, `metax`, `iluvatar`, `cambricon`, `thead`. Любое другое имя (например, опечатка) приводит к ошибке рендеринга со списком поддерживаемых значений.

!!! note

    Каждый набор демонов графического процессора получает автоматическую метку присутствия PCI `nodeSelector` (например, `feature.node.kubernetes.io/pci-10de.present: "true"` для NVIDIA), рекламируемую Node Feature Discovery. **NFD должен быть установлен** в кластере, иначе ни один узел не будет нести необходимые метки и все рабочие модули останутся `Pending`.

Всякий раз, когда в списке указан хотя бы один поставщик графического процессора (т. е. `worker.gpuVendors` не пусто), каждый рабочий модуль дополнительно получает обязательный `podAntiAffinity` (topologyKey=hostname), поэтому два рабочих не могут совместно использовать узел — это защищает порты `hostNetwork: true` от коллизий.

Рабочий DaemonSet ЦП охватывает узлы, на которые не претендует ни один поставщик графического процессора, через `nodeSelector` `feature.gpustack.ai/acceleratable: "false"`. Установите `worker.cpuEnabled=false`, чтобы оставить эти узлы в покое — обычно, когда плоскость управления разделяет кластер с узлами графического процессора и не должна привлекать рабочих:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

`worker.gpuVendors` должен содержать хотя бы одного поддерживаемого производителя; иначе чарт не создаст ни одного DaemonSet для воркеров и установка будет отклонена. GPU DaemonSet в любом случае сохраняют суффикс `-<vendor>`, поэтому отключение CPU DaemonSet никогда не переименовывает GPU DaemonSet в `<release>-worker`.

Альтернативно, добавьте кластеры графических процессоров и рабочие узлы через пользовательский интерфейс на страницах **Кластеры** и **Рабочие** после установки.

### Использование внешней базы данных { #using-an-external-database }

По умолчанию GPUStack использует встроенную базу данных PostgreSQL. Чтобы использовать внешнюю базу данных PostgreSQL или MySQL, установите `server.externalDatabaseURL`:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### Включение HTTPS с помощью специального сертификата. { #enabling-https-with-a-custom-certificate }

Предоставьте сертификат и содержимое ключа (PEM) через `server.ingress.tls`. Когда оба установлены, входная схема становится HTTPS:

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### Настройка ресурсов шлюза Higress { #sizing-the-higress-gateway }

Шлюз Higress (Envoy) обрабатывает весь клиентский трафик инференса. Его потребление памяти растет с частотой изменения маршрутов/конфигурации и числом одновременных соединений, а LLM-нагрузки долго удерживают соединения из-за потоковых ответов — поэтому при постоянной нагрузке память продолжает расти. Входящий в комплект субчарт `higress-core` по умолчанию задает и requests, и limits `2` CPU / `2Gi`, чего недостаточно для длительной эксплуатации; поэтому чарт GPUStack повышает **limits** шлюза до `4` CPU / `4096Mi` (requests остаются значениями субчарта по умолчанию — `2` CPU / `2048Mi`).

!!! warning

    OOMKill шлюза — это не только прерывание инференса. Все находящиеся в обработке запросы через шлюз обрываются, а если воркеры обращаются к серверу через адрес шлюза, все соединения воркеров обрываются одновременно, вызывая массовое переподключение по всему кластеру, из-за чего модельные сервисы могут стать недоступными. См. [Прямое подключение воркеров к серверу](#connecting-workers-directly-to-the-server) ниже.

Правила настройки:

- **Следите за памятью до того, как она упрётся в лимит.** Встроенный Prometheus собирает статистику Envoy шлюза, поэтому настройте алерты на `envoy_server_memory_allocated` и `envoy_server_memory_heap_size` (см. `gateway.namespace`, если Higress работает в другом пространстве имён). `container_memory_working_set_bytes` — метрика kubelet/cAdvisor, она доступна только из кластерного Prometheus. Если в обычной работе память приближается к лимиту, повышайте `higress-core.gateway.resources.limits.memory` — не дожидайтесь первого OOMKill.
- **Держите минимум 2 реплики для доступности.** У Deployment шлюза по умолчанию `2` реплики (`higress-core.gateway.replicas`), поэтому роллинг-перезапуск или отказ одного Pod не обрывает весь трафик. Реплики не снижают память на Pod от нагрузки на конфигурацию и соединения — каждая реплика хранит полный набор маршрутов, — но распределяют трафик запросов, так что на каждом Pod одновременно обрабатывается меньше крупных запросов (см. ниже). Обратите внимание, что `higress-core.gateway.service.externalTrafficPolicy` равен `Local`: для Service типа LoadBalancer/NodePort внешний LB должен проверять `healthCheckNodePort`, иначе узлы без Pod шлюза отбрасывают трафик.
- **Лимит памяти — то, что защищает от OOM.** В подчарте есть HPA по утилизации CPU. Он сглаживает нагрузку, но **не** защищает от OOM: память шлюза растёт с числом соединений и размером запросов, а не с CPU. Даже с автоскейлингом оставляйте запас в лимите памяти.
- **HPA переопределяет `replicas`.** При `higress-core.gateway.autoscaling.enabled=true` Deployment больше не использует `replicas`, а `autoscaling.minReplicas` подчарта по умолчанию равен `1`, поэтому одно лишь включение HPA позволяет шлюзу сжаться до одного Pod. Если HPA включён, задайте `autoscaling.minReplicas` не меньше `2`.

```yaml
# values.yaml
higress-core:
  gateway:
    replicas: 2
    resources:
      requests:
        cpu: 2000m
        memory: 4096Mi
      limits:
        cpu: 4000m
        memory: 8192Mi   # увеличьте ещё, если память продолжает расти
    autoscaling:
      enabled: true
      minReplicas: 2    # HPA игнорирует `replicas`; держите не меньше 2
      maxReplicas: 5
      targetCPUUtilizationPercentage: 80
```

#### Планирование ёмкости под запросы с картинками

Нагрузка от маршрутов и соединений — постоянная часть потребления; резкий рост даёт размер запроса. Плагины шлюза, которые просматривают или переписывают запрос, буферизуют тело запроса целиком, поэтому память растёт вместе с байтами запросов в обработке — число одновременных запросов, умноженное на размер тела. Чаще всего до этого доходят запросы с картинками в base64: в JSON каждая картинка занимает в 1,33 раза больше своего размера.

Цифры ниже получены на одном Pod шлюза (2 CPU / 4Gi) со стандартной цепочкой плагинов GPUStack и mock-апстриром: по 60 секунд стриминговой нагрузки на случай, каждый случай — с только что запущенного Pod. Это ориентир по порядку величины, а не формула для расчёта:

| Картинки | Тело запроса | Одновременно | В обработке | Память сверх холостого хода |
|---|---|---|---|---|
| 1 | 1,3 МБ | 64 | 85 МБ | около 0,1 GiB |
| 8 | 10,7 МБ | 4 | 43 МБ | около 0,2 GiB |
| 32 | 42,7 МБ | 4 | 171 МБ | около 1,8 GiB |
| 32 | 42,7 МБ | 16 | 683 МБ | **OOMKilled** |

Стоимость одного запроса растёт быстрее, чем само тело, и после окончания нагрузки память не возвращается сразу, поэтому планируйте по байтам в обработке, а не по числу запросов. Для Pod с 4Gi держите тела запросов в обработке суммарно ниже примерно 200 МБ или повышайте `higress-core.gateway.resources.limits.memory` (для трафика со многими картинками разумно начать с 8Gi); дополнительные реплики тоже помогают, поскольку распределяют запросы.

Для дальнейшей настройки Higress обратитесь к [документации Higress по Helm](https://higress.cn/en/docs/latest/ops/deploy-by-helm).

### Прямое подключение воркеров к серверу { #connecting-workers-directly-to-the-server }

Регистрация, хартбиты и watch моделей/экземпляров между воркером и сервером (а в режиме прокси `tunnel` — ещё и постоянный WebSocket) — это трафик плоскости управления, он не зависит от клиентского инференс-трафика и **не должен проходить через шлюз Higress**: иначе при перезапуске шлюза или OOMKill все соединения воркеров рвутся одновременно.

Для воркеров, управляемых чартом, всё уже настроено правильно: при `worker.enabled=true` и незаданном `worker.serverURL` воркеры регистрируются через внутрикластерный Service (`http://<release>-server.<namespace>.svc:<server.apiPort>`), полностью минуя шлюз. Если вы задаёте адрес сами, предпочитайте Service типа ClusterIP (`<release>-server-cluster-ip`), а не этот headless: headless-имя резолвится в IP Pod, которые могут устареть при переходе сервера на другую реплику.

Проверить стоит кластеры, зарегистрированные через **Register Cluster** (или любой воркер, запущенный с `--server-url`). `worker.serverURL` в манифесте регистрации выдаёт сервер, который разрешает адрес в таком порядке: собственный **Server URL** кластера, затем `server_external_url`, затем Host запроса. Значит, если у кластера нет переопределения, это *внешний* URL сервера. В Helm-развёртывании это обычно адрес ingress, то есть шлюз Higress. В такой схеме последствия OOM шлюза шире, чем потерянные запросы:

- Все воркеры одновременно теряют соединения с сервером. В режиме `tunnel` обрываются все инференс-потоки, передаваемые через туннели, а туннельный клиент переподключается с экспоненциальной задержкой (от 1 до 60 секунд с джиттером). В режимах `direct`/`worker` туннеля нет, но watch моделей и экземпляров — долгоживущие HTTP-потоки, они тоже обрываются, после чего происходит массовое переподключение и повторное получение списков; хартбиты — периодические вызовы, которые просто завершаются ошибкой и повторяются в следующем цикле.

- Если простой превышает льготный период хартбитов (`GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`, по умолчанию 150 с), воркеры помечаются офлайн и их маршруты моделей отзываются — модельные сервисы становятся недоступны, хотя процессы инференса продолжают работать.

Чтобы отвязать плоскость управления от шлюза, направьте воркеры на адрес, который достигает сервера напрямую:

- **Кластеры из Register Cluster:** задайте в кластере **Server URL** — прямой адрес (внутрикластерный Service, если воркеры в кластере сервера, иначе выделенный NodePort/LoadBalancer на Service сервера), затем заново получите и примените манифест регистрации. Делайте это в кластере, а не через `helm upgrade --set worker.serverURL=…`: при следующем применении манифеста bootstrap Job переустановит worker-релиз со значениями, отрендеренными сервером, и URL шлюза вернётся. Не добавляйте и второй worker-only релиз рядом с релизом из манифеста: получатся дублирующие DaemonSet воркеров (их Pod останутся в Pending из-за port anti-affinity при `hostNetwork`) и второй оператор.

- **Воркеры вне Kubernetes:** запускайте их с `--server-url`, указывающим на прямой адрес.

- **Воркеры, установленные Helm без манифеста регистрации:** worker-only релиз в кластере сервера. Он живёт в том же пространстве имён, что и релиз сервера, поэтому не должен рендерить Secret, которые уже принадлежат этому релизу: задайте `imagePullSecret.create: false` и укажите в `registrationTokenSecretName` Secret сервера `registration-token` (`worker.enabled` по умолчанию `false`, а `server` и `higress-core` по умолчанию включены, поэтому все три тоже нужно указать явно):

```yaml
# values.yaml для worker-only релиза в кластере сервера
server:
  enabled: false
higress-core:
  enabled: false
imagePullSecret:
  create: false
registrationTokenSecretName: registration-token
worker:
  enabled: true
  serverURL: http://gpustack-server-server-cluster-ip.gpustack-system.svc:30080
```

Тогда шлюз несёт только клиентский инференс-трафик, и его перезапуск больше не затрагивает регистрацию, хартбиты и watch воркеров. О самих режимах прокси см. [Режимы подключения воркеров](../user-guide/cluster-management.md#worker-connection-modes).

### Получение образов из частного реестра { #pulling-images-from-a-private-registry }

Чтобы загружать все образы (сервера/воркера GPUStack, higress-plugins и входящих в комплект компонентов higress-core: gateway/controller/pilot) из зеркального частного реестра, переопределите `global.hub`. Благодаря распространению глобальных значений Helm одна настройка применяется ко всем образам:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` также передается на сервер как `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`, гарантируя, что образы механизма вывода (например, vLLM, llama.cpp) извлекаются из одного и того же реестра.

Чтобы предоставить учетные данные извлечения, чарт может создать секрет `docker-registry` с именем `gpustack-image-pull-secret` и подключить его ко всем модулям:

```bash
helm install gpustack-server oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

Чтобы вместо этого ссылаться на свои собственные ранее существовавшие секреты, замените `global.imagePullSecrets`:

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## Параметры чарта { #chart-parameters }

Ниже перечислены наиболее часто используемые параметры. Полный и авторитетный список см. в [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) и [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md) таблицы.

| Параметр | По умолчанию | Описание |
| -------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `debug`                                | `false`                  | Включить режим отладки. |
| `registrationToken`                    | `null`                   | Регистрационный жетон работника; случайный генерируется и используется повторно, если `null`. |
| `clusterDomain`                        | `cluster.local`          | Kubernetes суффикс домена службы кластера. |
| `global.hub`                           | `docker.io`              | Хост реестра контейнеров; переопределить для частного реестра. |
| `global.imagePullSecrets`              | `[gpustack-image-pull-secret]` | Секреты извлечения образов, прикрепленные ко всем модулям и распространяемые на субчарты. |
| `global.nodeSelector`                  | `{}`                     | NodeSelector по умолчанию для каждого компонента; заменено значением на уровне компонента. |
| `image.repository`                     | `gpustack/gpustack`      | Репозиторий образов с пространством имен; итоговая ссылка — `{global.hub}/{repository}:{tag}`. |
| `image.tag`                            | `null`                   | Тег образа; по умолчанию используется значение `appVersion` чарта. |
| `image.pullPolicy`                     | `IfNotPresent`           | Политика получения образов. |
| `imagePullSecret.credentials.registry` | `docker.io`              | Хост реестра, используемый, когда чарт создает секрет извлечения. |
| `imagePullSecret.credentials.username` | `null`                   | Имя пользователя реестра; создает секрет при установке пароля. |
| `imagePullSecret.credentials.password` | `null`                   | Пароль реестра; создает секрет, если он установлен с именем пользователя. |
| `server.ingress.hostname`              | `null`                   | Входное имя хоста для сервера. |
| `server.ingress.tls.cert`              | `null`                   | Входной сертификат TLS (PEM); включает HTTPS при установке с помощью ключа. |
| `server.ingress.tls.key`               | `null`                   | Входной закрытый ключ TLS (PEM). |
| `server.externalDatabaseURL`           | `null`                   | Строка подключения к внешней базе данных (PostgreSQL или MySQL). |
| `server.dataVolume.hostPath`           | `null`                   | Путь к хосту для тома данных сервера; использует HostPath вместо PVC. |
| `server.dataVolume.size`               | `100Gi`                  | Размер тома данных сервера (PVC). |
| `server.apiPort`                       | `30080`                  | Порт службы API. |
| `server.metricsPort`                   | `10161`                  | Порт метрик сервера. |
| `server.environmentConfig`             | `{}`                     | Дополнительные переменные среды для сервера GPUStack. |
| `server.nodeSelector`                  | `{}`                     | Серверный модуль nodeSelector; заменяет `global.nodeSelector`, если не пусто. |
| `gateway.ingressClassname`             | `higress`                | Higress Имя IngressClass; при обнаружении включает режим шлюза внутри кластера. |
| `higress-core.enabled`                 | `true`                   | Разверните входящий в комплект шлюз Higress; отключить, если он уже установлен. |
| `higress-core.gateway.replicas`        | `2`                      | Число реплик шлюза Higress; держите >= 2 для доступности. |
| `higress-core.gateway.resources`       | limits `4` CPU / `4096Mi` | Ресурсы шлюза; limits повышены относительно значений субчарта. См. [Настройка ресурсов шлюза Higress](#sizing-the-higress-gateway). |
| `higress-core.gateway.autoscaling.enabled` | `false`              | HPA шлюза по CPU; не предотвращает OOM — задавайте лимиты памяти с запасом в любом случае. |
| `higress-core.gateway.autoscaling.minReplicas` | `1`                      | Минимум реплик при включённом HPA. HPA переопределяет `replicas`; ставьте не меньше 2. |
| `worker.serverURL`                     | `""`                     | Адрес сервера для регистрации воркеров; пусто — используется внутрикластерный Service сервера. Не маршрутизируйте через шлюз. |
| `worker.enabled`                       | `false`                  | Создавать манифесты DaemonSet для воркеров. |
| `worker.gpuVendors`                    | `[nvidia]`               | поставщики графических процессоров; один набор DaemonSet от каждого поставщика плюс набор демонов ЦП. |
| `worker.cpuEnabled`                    | `true`                   | Создавать манифест DaemonSet для CPU-воркеров; `false` требуется поставщик графического процессора. |
| `worker.nodeSelector`                  | `{}`                     | Базовый nodeSelector воркеров; заменяет `global.nodeSelector`, если не пусто. |
| `worker.port`                          | `10150`                  | Рабочий порт обслуживания. |
| `worker.metricsPort`                   | `10151`                  | Порт рабочих метрик. |
| `worker.dataDir`                       | `/var/lib/gpustack`      | Путь к хосту установлен в точке `/var/lib/gpustack` внутри каждого рабочего модуля. |
| `worker.environmentConfig`             | `{}`                     | Дополнительные переменные среды для работника GPUStack. |

## Удаление { #uninstallation }

В выпуске развертывается больше, чем просто сервер: субчарт `gpustack-operator` содержит контроллеры Kueue и Node Feature Discovery, а оператор создает пользовательские ресурсы, содержащие финализаторы. `helm uninstall` удаляет контроллеры вместе со всем остальным, поэтому пользовательские ресурсы должны быть удалены **сначала** — удалены, пока контроллеры еще работают — иначе объекты останутся в `Terminating` навсегда, а пространство имен `gpustack-system` никогда не покинет его. По порядку:

1. Сбросьте рабочие нагрузки, пока **каждый** контроллер все еще работает. Оператор очищает финализаторы своих объектов `Instance`, а Kueue отказывается освобождать `ClusterQueue`, пока допущенные `Workload` все еще занимают его — поэтому они должны удалиться, прежде чем что-либо будет отключено. `Instance` находятся в пространстве имен и живут в пространствах имен арендаторов, поэтому перечислите их все с помощью `-A`. Удалить модели с сервера (или их экземпляры напрямую); модуль каждого экземпляра завершается вместе с ним, и рабочая нагрузка отправляется вместе с экземпляром. Не удаляйте поды пространства имен целиком — это уничтожит оператора, контроллеры Kueue и NFD, и именно они завершат очистку:

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    Если Kueue используется совместно с другими приложениями, не используйте `--all` здесь или на шаге 3 — удалите только объекты GPUStack. Все, что оператор создает в Kueue, имеет префикс `gpustack` (`clusterqueue/gpustack--…`, `admissioncheck/gpustack-node-devices`). Виды в области кластера выбираются только по имени; для пространств имен необходимо, чтобы пространство имен сохранялось, потому что `-o name` его опускает:

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. Остановите собственные контроллеры оператора, но оставьте версию установленной, чтобы контроллеры Kueue продолжали работать — именно они завершают удаление на следующем этапе. Пока `gpustack-operator-worker` и `gpustack-operator-device-manager-*` DaemonSets работают, они воссоздают `LocalQueue`, `ClusterQueue` и `gpustack-node-devices` AdmissionCheck, как только вы их удаляете, поэтому счетчики никогда не достигают нуля:

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. Удалите ресурсы Kueue в порядке зависимости — `LocalQueue`s, затем `ClusterQueue`s, затем `AdmissionCheck`s, затем `ResourceFlavor`s. Kueue удерживает финализатор `kueue.x-k8s.io/resource-in-use` для объекта, пользователи которого все еще присутствуют: AdmissionCheck `gpustack-node-devices` остается в состоянии удаления до тех пор, пока не исчезнут все ClusterQueue, и каждый ClusterQueue до тех пор, пока не исчезнут его LocalQueues и допущенные рабочие нагрузки. Подождите, пока каждый вид опустеет, прежде чем переходить к следующему. В общем кластере примените к каждой команде префикс `gpustack` из шага 1 вместо `--all`:

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. Удалите **все** выпуски GPUStack в пространстве имен. В конфигурации, которую рекомендует эта страница — сервер установлен как `gpustack-server` со значением по умолчанию `worker.enabled=false`, а тот же кластер зарегистрирован через **Register Cluster**, что устанавливает рабочий выпуск с именем `gpustack` в `gpustack-system` — оператор, Kueue и Node Feature Discovery принадлежат рабочему выпуску, а не выпуску сервера, поэтому удаление только выпуска сервера оставляет рабочий выпуск и его контроллеры запущенными. Проверьте `helm list -n gpustack-system` и удалите каждый показанный выпуск GPUStack. Если вы зарегистрировали кластер в другом пространстве имен (например, при включенном `worker.enabled=true` в выпуске сервера — см. примечание выше), повторите удаление и в том пространстве имен: его рабочий выпуск несет собственные оператор, Kueue и Node Feature Discovery. Не передавайте `--wait`: хук очистки Node Feature Discovery после удаления блокирует ожидание завершения дольше, чем задано в `--timeout`. Если принудительно завершить Helm, созданные для очистки ресурсы RBAC останутся без владельца (на шаге 5 они удаляются вручную — выполните шаг 5 и в случае, когда `--wait` не использовался):

    ```bash
    helm list -n gpustack-system -q | grep -E '^gpustack(-server)?$' \
      | xargs -r -n 1 helm uninstall -n gpustack-system
    ```

5. Удалите то, что не удаляет `helm uninstall`. Helm никогда не удаляет CRD, а оператор регистрирует admission webhooks и APIServices, не принадлежащие ни одному релизу. После исчезновения Service оператора оба механизма блокируют запросы при ошибке обращения к нему (fail closed), а webhooks не позволяют снять финализаторы приведённой ниже командой, пока сами не будут удалены:

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    Группа `worker.gpustack.ai` всегда принадлежит GPUStack — ни один другой компонент ее не использует — поэтому ее CRD можно безопасно удалить без каких-либо условий.

6. Удалите CRD, оставшиеся после выпуска — **только те, которые установлены в этом выпуске**. При удалении CRD удаляются все объекты такого типа во всем кластере, поэтому, если в кластере уже выполнялись Kueue, Node Feature Discovery или Higress до GPUStack (см. `higress-core.enabled` выше) и что-то еще все еще использует их, оставьте их CRD на месте и пропустите этот шаг для этих групп. Узнайте, что есть в этом выпуске, а не гадайте:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") | test("^gpustack(-server)?$"))
      | .metadata.name'
    ```

    В стандартной установке, в которой указаны CRD Kueue, а также NFD и Higress, если они установлены в таблице:

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    Собственные CRD провайдера кластера никогда не являются объектами GPUStack для удаления — например, `*.k3s.cattle.io` и `helm.cattle.io` в кластере k3s принадлежат кластеру.

7. *(Необязательно)* Удалите пространство имен, которое включает в себя объекты Higress, загрузочную карту ConfigMap, ServiceAccount и регистрационный токен Secret. Знайте, что с этим связано: пространство имен содержит PVC сервера (с именем `gpustack-data-dir-<release-name>-server-0`, например `gpustack-data-dir-gpustack-server-server-0` для выпуска по умолчанию `gpustack-server`), в котором хранятся постоянные данные сервера — встроенная база данных PostgreSQL с ее пользователями, моделями и кластерами. В политике возврата `Delete` (по умолчанию `local-path`) базовый том также удаляется. Сохраните пространство имен, если вам нужны данные для переустановки; в противном случае удалите пространство имен (и сначала PVC, если политика восстановления еще не удалила его):

    ```bash
    kubectl get pvc -n gpustack-system -o name | grep -E '^persistentvolumeclaim/gpustack-data-dir-.*-server-0$' \
      | xargs -r -n 1 kubectl delete -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
